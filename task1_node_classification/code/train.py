from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dgl
import torch
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler
from torch.nn import functional as F

from common.models import GNNEncoder, SUPPORTED_GNN_MODELS
from common.utils import (
    Timer,
    count_parameters,
    default_num_workers,
    ensure_dgl_device,
    load_checkpoint,
    parse_int_list,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
    timestamped_path,
)


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
RESULT_DIRECTORY = Path(__file__).resolve().parents[1] / "results"


@dataclass
class NodeConfig:
    dataset: str = "cora"
    model: str = "gcn"
    mode: str = "full"
    epochs: int = 200
    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    fanout: str = "10,10"
    batch_size: int = 1024
    patience: int = 30
    seed: int = 42
    device: str = "auto"
    num_workers: int | None = None
    log_every: int = 10
    data_dir: str = str(DATA_DIRECTORY)
    output: str | None = None
    checkpoint_out: str | None = None


def load_node_dataset(name: str, raw_dir: str | Path) -> tuple[dgl.DGLGraph, int]:
    from dgl.data import CiteseerGraphDataset, CoraGraphDataset, FlickrDataset

    datasets: dict[str, Any] = {
        "cora": CoraGraphDataset,
        "citeseer": CiteseerGraphDataset,
        "flickr": FlickrDataset,
    }
    normalized = name.lower()
    if normalized not in datasets:
        raise ValueError(f"Unknown dataset {name!r}; choose from {tuple(datasets)}.")
    dataset = datasets[normalized](raw_dir=str(raw_dir), verbose=True)
    graph = dataset[0]
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    num_classes = int(getattr(dataset, "num_classes", graph.ndata["label"].max().item() + 1))
    return graph, num_classes


def structure_only(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Copy topology without duplicating dataset features on the accelerator."""
    source, destination = graph.edges()
    return dgl.graph(
        (source, destination),
        num_nodes=graph.num_nodes(),
        idtype=graph.idtype,
        device=graph.device,
    )


@torch.no_grad()
def evaluate(
    model: GNNEncoder,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    logits = model(graph, features)
    loss = F.cross_entropy(logits[mask], labels[mask]).item()
    accuracy = (logits[mask].argmax(dim=-1) == labels[mask]).float().mean().item()
    return loss, accuracy


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def run_training(config: NodeConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    ensure_dgl_device(device)
    graph, num_classes = load_node_dataset(config.dataset, config.data_dir)
    features_cpu = graph.ndata["feat"].float()
    labels_cpu = graph.ndata["label"].long()
    train_ids = torch.nonzero(graph.ndata["train_mask"], as_tuple=False).squeeze(-1)

    model = GNNEncoder(
        model=config.model,
        in_dim=features_cpu.shape[1],
        hidden_dim=config.hidden_dim,
        out_dim=num_classes,
        num_layers=config.layers,
        dropout=config.dropout,
        heads=config.heads,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    eval_graph = structure_only(graph).to(device)
    eval_features = features_cpu.to(device)
    eval_labels = labels_cpu.to(device)
    train_mask = graph.ndata["train_mask"].to(device)
    val_mask = graph.ndata["val_mask"].to(device)
    test_mask = graph.ndata["test_mask"].to(device)

    loader: DataLoader | None = None
    if config.mode == "sampled":
        fanouts = parse_int_list(config.fanout, expected=config.layers)
        sampler = MultiLayerNeighborSampler(fanouts)
        workers = (
            default_num_workers(device) if config.num_workers is None else config.num_workers
        )
        loader = DataLoader(
            graph,
            train_ids,
            sampler,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=workers,
        )
    elif config.mode != "full":
        raise ValueError("mode must be either 'full' or 'sampled'")

    best_state = copy.deepcopy(_cpu_state_dict(model))
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    total_train_seconds = 0.0
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        with Timer(device) as timer:
            if config.mode == "full":
                optimizer.zero_grad(set_to_none=True)
                logits = model(eval_graph, eval_features)
                train_loss = F.cross_entropy(logits[train_mask], eval_labels[train_mask])
                train_loss.backward()
                optimizer.step()
                mean_train_loss = float(train_loss.item())
            else:
                assert loader is not None
                weighted_loss = 0.0
                examples = 0
                for input_nodes, output_nodes, blocks in loader:
                    blocks = [block.to(device) for block in blocks]
                    batch_features = features_cpu[input_nodes].to(device)
                    batch_labels = labels_cpu[output_nodes].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(blocks, batch_features)
                    loss = F.cross_entropy(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    weighted_loss += float(loss.item()) * output_nodes.numel()
                    examples += output_nodes.numel()
                mean_train_loss = weighted_loss / max(examples, 1)
        total_train_seconds += timer.seconds

        val_loss, val_accuracy = evaluate(
            model, eval_graph, eval_features, eval_labels, val_mask
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "train_seconds": timer.seconds,
            }
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % config.log_every == 0:
            print(
                f"epoch={epoch:04d} train_loss={mean_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
                f"train_s={timer.seconds:.3f}"
            )
        if config.patience > 0 and stale_epochs >= config.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate(
        model, eval_graph, eval_features, eval_labels, test_mask
    )
    _, final_val_accuracy = evaluate(model, eval_graph, eval_features, eval_labels, val_mask)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    run_stem = (
        f"{config.dataset}-{config.model}-{config.mode}-seed{config.seed}"
    )
    result_path = (
        Path(config.output)
        if config.output
        else timestamped_path(RESULT_DIRECTORY, run_stem)
    )
    checkpoint_path = (
        Path(config.checkpoint_out)
        if config.checkpoint_out
        else result_path.with_suffix(".pt")
    )
    result: dict[str, Any] = {
        "task": "node_classification",
        "config": asdict(config),
        "dataset": {
            "num_nodes": graph.num_nodes(),
            "num_edges": graph.num_edges(),
            "num_features": features_cpu.shape[1],
            "num_classes": num_classes,
            "train_nodes": int(train_mask.sum().item()),
            "val_nodes": int(val_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
        },
        "device": str(device),
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_val_accuracy": final_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "total_train_seconds": total_train_seconds,
        "mean_epoch_train_seconds": total_train_seconds / max(len(history), 1),
        "peak_cuda_memory_mb": peak_memory_mb,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    save_checkpoint(checkpoint_path, model, optimizer, result)
    save_json(result, result_path)
    print(json.dumps({key: result[key] for key in (
        "best_epoch", "best_val_accuracy", "test_accuracy", "total_train_seconds", "checkpoint"
    )}, ensure_ascii=False, indent=2))
    print(f"result={result_path}")
    return result


def evaluate_checkpoint(checkpoint: str | Path, device_name: str = "auto") -> dict[str, Any]:
    device = resolve_device(device_name)
    ensure_dgl_device(device)
    payload = load_checkpoint(checkpoint, device)
    metadata = payload["metadata"]
    raw_config = metadata["config"]
    valid_fields = NodeConfig.__dataclass_fields__.keys()
    config = NodeConfig(**{key: value for key, value in raw_config.items() if key in valid_fields})
    graph, num_classes = load_node_dataset(config.dataset, config.data_dir)
    features = graph.ndata["feat"].float().to(device)
    labels = graph.ndata["label"].long().to(device)
    test_mask = graph.ndata["test_mask"].to(device)
    model = GNNEncoder(
        model=config.model,
        in_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        out_dim=num_classes,
        num_layers=config.layers,
        dropout=config.dropout,
        heads=config.heads,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    test_loss, test_accuracy = evaluate(
        model, structure_only(graph).to(device), features, labels, test_mask
    )
    result = {
        "checkpoint": str(checkpoint),
        "dataset": config.dataset,
        "model": config.model,
        "mode": config.mode,
        "device": str(device),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGL node classification benchmark")
    parser.add_argument("--dataset", choices=("cora", "citeseer", "flickr"), default="cora")
    parser.add_argument("--model", choices=SUPPORTED_GNN_MODELS, default="gcn")
    parser.add_argument("--mode", choices=("full", "sampled"), default="full")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--fanout", default="10,10")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--data-dir", default=str(DATA_DIRECTORY))
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval-only requires --checkpoint")
        evaluate_checkpoint(args.checkpoint, args.device)
        return
    config_keys = NodeConfig.__dataclass_fields__.keys()
    run_training(NodeConfig(**{key: value for key, value in vars(args).items() if key in config_keys}))


if __name__ == "__main__":
    main()
