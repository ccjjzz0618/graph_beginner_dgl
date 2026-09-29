from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler, as_edge_prediction_sampler
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from common.models import DotProductPredictor, GNNEncoder, SUPPORTED_GNN_MODELS
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
class LinkConfig:
    dataset: str = "cora"
    model: str = "gcn"
    mode: str = "full"
    epochs: int = 200
    hidden_dim: int = 64
    embedding_dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.3
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    fanout: str = "15,10"
    batch_size: int = 2048
    negative_ratio: int = 1
    val_ratio: float = 0.05
    test_ratio: float = 0.10
    patience: int = 30
    seed: int = 42
    device: str = "auto"
    num_workers: int | None = None
    log_every: int = 10
    data_dir: str = str(DATA_DIRECTORY)
    output: str | None = None
    checkpoint_out: str | None = None


@dataclass
class EdgeSplit:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    forbidden_keys: np.ndarray


def load_link_dataset(name: str, raw_dir: str | Path) -> dgl.DGLGraph:
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
    graph = dgl.remove_self_loop(dataset[0])
    return graph


def split_undirected_edges(
    graph: dgl.DGLGraph,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> EdgeSplit:
    """Split unique unordered pairs so a reverse edge cannot leak across splits."""
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than one")
    source, destination = graph.edges()
    source_np = source.cpu().numpy().astype(np.int64, copy=False)
    destination_np = destination.cpu().numpy().astype(np.int64, copy=False)
    keep = source_np != destination_np
    lower = np.minimum(source_np[keep], destination_np[keep])
    upper = np.maximum(source_np[keep], destination_np[keep])
    num_nodes = graph.num_nodes()
    keys = lower * num_nodes + upper
    _, first_indices = np.unique(keys, return_index=True)
    pairs = np.stack((lower[first_indices], upper[first_indices]), axis=1)
    if len(pairs) < 3:
        raise ValueError("At least three unique non-self edges are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    validation_count = max(1, int(len(pairs) * val_ratio))
    test_count = max(1, int(len(pairs) * test_ratio))
    train_count = len(pairs) - validation_count - test_count
    if train_count < 1:
        raise ValueError("The requested split leaves no training edges")
    train = torch.from_numpy(pairs[:train_count].copy()).long()
    validation = torch.from_numpy(
        pairs[train_count : train_count + validation_count].copy()
    ).long()
    test = torch.from_numpy(pairs[train_count + validation_count :].copy()).long()
    forbidden = np.unique(
        np.concatenate(
            (
                pairs[:, 0] * num_nodes + pairs[:, 1],
                pairs[:, 1] * num_nodes + pairs[:, 0],
            )
        )
    )
    return EdgeSplit(train, validation, test, forbidden)


def sample_non_edges(
    num_nodes: int,
    count: int,
    forbidden_keys: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Vectorized rejection sampling of directed non-edges."""
    chunks: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        candidate_count = max(remaining * 2, 1024)
        source = rng.integers(0, num_nodes, size=candidate_count, dtype=np.int64)
        destination = rng.integers(0, num_nodes, size=candidate_count, dtype=np.int64)
        keys = source * num_nodes + destination
        valid = (source != destination) & ~np.isin(keys, forbidden_keys, assume_unique=False)
        accepted = np.stack((source[valid], destination[valid]), axis=1)
        if len(accepted):
            accepted = accepted[:remaining]
            chunks.append(accepted)
            remaining -= len(accepted)
    return torch.from_numpy(np.concatenate(chunks, axis=0)).long()


def build_training_graph(
    num_nodes: int, train_pairs: torch.Tensor
) -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    pair_count = train_pairs.shape[0]
    source = torch.cat((train_pairs[:, 0], train_pairs[:, 1]))
    destination = torch.cat((train_pairs[:, 1], train_pairs[:, 0]))
    graph = dgl.graph((source, destination), num_nodes=num_nodes)
    graph = dgl.add_self_loop(graph)
    edge_ids = torch.arange(pair_count, dtype=torch.int64)
    reverse_ids = torch.cat(
        (
            torch.arange(pair_count, 2 * pair_count, dtype=torch.int64),
            torch.arange(0, pair_count, dtype=torch.int64),
            torch.arange(2 * pair_count, 2 * pair_count + num_nodes, dtype=torch.int64),
        )
    )
    return graph, edge_ids, reverse_ids


class NonEdgeSampler:
    def __init__(
        self,
        num_nodes: int,
        ratio: int,
        forbidden_keys: np.ndarray,
        seed: int,
    ) -> None:
        self.num_nodes = num_nodes
        self.ratio = ratio
        self.forbidden_keys = forbidden_keys
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, graph: dgl.DGLGraph, edge_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del graph
        pairs = sample_non_edges(
            self.num_nodes,
            int(edge_ids.numel()) * self.ratio,
            self.forbidden_keys,
            self.rng,
        )
        return pairs[:, 0], pairs[:, 1]


def link_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        positive_scores, torch.ones_like(positive_scores)
    ) + F.binary_cross_entropy_with_logits(
        negative_scores, torch.zeros_like(negative_scores)
    )


@torch.no_grad()
def evaluate_pairs(
    model: GNNEncoder,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    positive_pairs: torch.Tensor,
    negative_pairs: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    embeddings = model(graph, features)
    positive = positive_pairs.to(device)
    negative = negative_pairs.to(device)
    positive_scores = DotProductPredictor.score_pairs(
        embeddings, positive[:, 0], positive[:, 1]
    )
    negative_scores = DotProductPredictor.score_pairs(
        embeddings, negative[:, 0], negative[:, 1]
    )
    scores = torch.cat((positive_scores, negative_scores)).detach().cpu().numpy()
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def run_training(config: LinkConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    ensure_dgl_device(device)
    original_graph = load_link_dataset(config.dataset, config.data_dir)
    features_cpu = original_graph.ndata["feat"].float()
    split = split_undirected_edges(
        original_graph, config.val_ratio, config.test_ratio, config.seed
    )
    train_graph, train_edge_ids, reverse_edge_ids = build_training_graph(
        original_graph.num_nodes(), split.train
    )
    rng = np.random.default_rng(config.seed + 1)
    validation_negatives = sample_non_edges(
        original_graph.num_nodes(), len(split.validation), split.forbidden_keys, rng
    )
    test_negatives = sample_non_edges(
        original_graph.num_nodes(), len(split.test), split.forbidden_keys, rng
    )

    model = GNNEncoder(
        model=config.model,
        in_dim=features_cpu.shape[1],
        hidden_dim=config.hidden_dim,
        out_dim=config.embedding_dim,
        num_layers=config.layers,
        dropout=config.dropout,
        heads=config.heads,
    ).to(device)
    predictor = DotProductPredictor()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    graph_device = train_graph.to(device)
    features_device = features_cpu.to(device)

    loader: DataLoader | None = None
    if config.mode == "sampled":
        fanouts = parse_int_list(config.fanout, expected=config.layers)
        sampler = MultiLayerNeighborSampler(fanouts)
        edge_sampler = as_edge_prediction_sampler(
            sampler,
            exclude="reverse_id",
            reverse_eids=reverse_edge_ids,
            negative_sampler=NonEdgeSampler(
                original_graph.num_nodes(),
                config.negative_ratio,
                split.forbidden_keys,
                config.seed + 2,
            ),
        )
        workers = (
            default_num_workers(device) if config.num_workers is None else config.num_workers
        )
        loader = DataLoader(
            train_graph,
            train_edge_ids,
            edge_sampler,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=workers,
        )
    elif config.mode != "full":
        raise ValueError("mode must be either 'full' or 'sampled'")

    best_state = copy.deepcopy(_cpu_state_dict(model))
    best_val_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    total_train_seconds = 0.0
    history: list[dict[str, float | int]] = []
    train_rng = np.random.default_rng(config.seed + 3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        with Timer(device) as timer:
            if config.mode == "full":
                negatives = sample_non_edges(
                    original_graph.num_nodes(),
                    len(split.train) * config.negative_ratio,
                    split.forbidden_keys,
                    train_rng,
                ).to(device)
                positive = split.train.to(device)
                optimizer.zero_grad(set_to_none=True)
                embeddings = model(graph_device, features_device)
                positive_scores = DotProductPredictor.score_pairs(
                    embeddings, positive[:, 0], positive[:, 1]
                )
                negative_scores = DotProductPredictor.score_pairs(
                    embeddings, negatives[:, 0], negatives[:, 1]
                )
                loss = link_loss(positive_scores, negative_scores)
                loss.backward()
                optimizer.step()
                train_loss = float(loss.item())
            else:
                assert loader is not None
                weighted_loss = 0.0
                examples = 0
                for input_nodes, positive_graph, negative_graph, blocks in loader:
                    blocks = [block.to(device) for block in blocks]
                    positive_graph = positive_graph.to(device)
                    negative_graph = negative_graph.to(device)
                    batch_features = features_cpu[input_nodes].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    embeddings = model(blocks, batch_features)
                    positive_scores = predictor(positive_graph, embeddings)
                    negative_scores = predictor(negative_graph, embeddings)
                    loss = link_loss(positive_scores, negative_scores)
                    loss.backward()
                    optimizer.step()
                    batch_examples = positive_graph.num_edges()
                    weighted_loss += float(loss.item()) * batch_examples
                    examples += batch_examples
                train_loss = weighted_loss / max(examples, 1)
        total_train_seconds += timer.seconds

        validation = evaluate_pairs(
            model,
            graph_device,
            features_device,
            split.validation,
            validation_negatives,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": validation["auc"],
                "val_average_precision": validation["average_precision"],
                "train_seconds": timer.seconds,
            }
        )
        if validation["auc"] > best_val_auc:
            best_val_auc = validation["auc"]
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % config.log_every == 0:
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.4f} "
                f"val_auc={validation['auc']:.4f} "
                f"val_ap={validation['average_precision']:.4f} train_s={timer.seconds:.3f}"
            )
        if config.patience > 0 and stale_epochs >= config.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    final_validation = evaluate_pairs(
        model,
        graph_device,
        features_device,
        split.validation,
        validation_negatives,
        device,
    )
    test_metrics = evaluate_pairs(
        model,
        graph_device,
        features_device,
        split.test,
        test_negatives,
        device,
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    run_stem = f"{config.dataset}-{config.model}-{config.mode}-seed{config.seed}"
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
        "task": "link_prediction",
        "config": asdict(config),
        "dataset": {
            "num_nodes": original_graph.num_nodes(),
            "original_edges": original_graph.num_edges(),
            "num_features": features_cpu.shape[1],
            "train_pairs": len(split.train),
            "validation_pairs": len(split.validation),
            "test_pairs": len(split.test),
        },
        "device": str(device),
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "validation": final_validation,
        "test": test_metrics,
        "total_train_seconds": total_train_seconds,
        "mean_epoch_train_seconds": total_train_seconds / max(len(history), 1),
        "peak_cuda_memory_mb": peak_memory_mb,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    save_checkpoint(checkpoint_path, model, optimizer, result)
    save_json(result, result_path)
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "validation": final_validation,
                "test": test_metrics,
                "total_train_seconds": total_train_seconds,
                "checkpoint": str(checkpoint_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"result={result_path}")
    return result


def evaluate_checkpoint(checkpoint: str | Path, device_name: str = "auto") -> dict[str, Any]:
    device = resolve_device(device_name)
    ensure_dgl_device(device)
    payload = load_checkpoint(checkpoint, device)
    metadata = payload["metadata"]
    raw_config = metadata["config"]
    valid_fields = LinkConfig.__dataclass_fields__.keys()
    config = LinkConfig(**{key: value for key, value in raw_config.items() if key in valid_fields})
    graph = load_link_dataset(config.dataset, config.data_dir)
    split = split_undirected_edges(graph, config.val_ratio, config.test_ratio, config.seed)
    train_graph, _, _ = build_training_graph(graph.num_nodes(), split.train)
    features = graph.ndata["feat"].float().to(device)
    model = GNNEncoder(
        model=config.model,
        in_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        out_dim=config.embedding_dim,
        num_layers=config.layers,
        dropout=config.dropout,
        heads=config.heads,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    rng = np.random.default_rng(config.seed + 1)
    _ = sample_non_edges(graph.num_nodes(), len(split.validation), split.forbidden_keys, rng)
    test_negatives = sample_non_edges(
        graph.num_nodes(), len(split.test), split.forbidden_keys, rng
    )
    metrics = evaluate_pairs(
        model, train_graph.to(device), features, split.test, test_negatives, device
    )
    result = {
        "checkpoint": str(checkpoint),
        "dataset": config.dataset,
        "model": config.model,
        "mode": config.mode,
        "device": str(device),
        "test": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGL link prediction benchmark")
    parser.add_argument("--dataset", choices=("cora", "citeseer", "flickr"), default="cora")
    parser.add_argument("--model", choices=SUPPORTED_GNN_MODELS, default="gcn")
    parser.add_argument("--mode", choices=("full", "sampled"), default="full")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--fanout", default="15,10")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.10)
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
    config_keys = LinkConfig.__dataclass_fields__.keys()
    run_training(LinkConfig(**{key: value for key, value in vars(args).items() if key in config_keys}))


if __name__ == "__main__":
    main()

