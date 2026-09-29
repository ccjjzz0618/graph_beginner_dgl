from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dgl
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from torch.nn import functional as F
from torch.utils.data import Dataset

from common.models import GraphClassifier, SUPPORTED_GNN_MODELS, SUPPORTED_POOLING
from common.utils import (
    Timer,
    count_parameters,
    default_num_workers,
    ensure_dgl_device,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
    timestamped_path,
)


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
RESULT_DIRECTORY = Path(__file__).resolve().parents[1] / "results"


@dataclass
class GraphConfig:
    dataset: str = "mutag"
    model: str = "gcn"
    pooling: str = "avg"
    mode: str = "mini"
    epochs: int = 200
    hidden_dim: int = 64
    layers: int = 3
    heads: int = 4
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 64
    eval_batch_size: int = 256
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    patience: int = 30
    seed: int = 42
    device: str = "auto"
    num_workers: int | None = None
    log_every: int = 10
    data_dir: str = str(DATA_DIRECTORY)
    output: str | None = None
    checkpoint_out: str | None = None


class AdaptedGraphDataset(Dataset[tuple[dgl.DGLGraph, torch.Tensor]]):
    def __init__(
        self,
        base: Any,
        indices: np.ndarray,
        feature_mode: Literal["existing", "categorical", "degree"],
        input_dim: int,
        task_type: Literal["classification", "regression"],
        label_mapping: dict[int, int] | None = None,
    ) -> None:
        self.base = base
        self.indices = indices.astype(np.int64, copy=False)
        self.feature_mode = feature_mode
        self.input_dim = input_dim
        self.task_type = task_type
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[dgl.DGLGraph, torch.Tensor]:
        graph, raw_label = self.base[int(self.indices[position])]
        graph = graph.clone()
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        if self.feature_mode == "categorical":
            categorical = graph.ndata["feat"].long().reshape(-1)
            features = F.one_hot(categorical, num_classes=self.input_dim).float()
        elif self.feature_mode == "existing":
            features = graph.ndata["feat"].float()
            if features.ndim == 1:
                features = features.unsqueeze(-1)
            else:
                features = features.reshape(graph.num_nodes(), -1)
        else:
            # log-degree is defined for every TUDataset, including those without attributes.
            features = torch.log1p(graph.in_degrees().float()).unsqueeze(-1)
        graph.ndata["x"] = features
        for key in list(graph.ndata.keys()):
            if key != "x":
                graph.ndata.pop(key)
        for key in list(graph.edata.keys()):
            graph.edata.pop(key)

        if self.task_type == "classification":
            assert self.label_mapping is not None
            label = int(torch.as_tensor(raw_label).reshape(-1)[0].item())
            return graph, torch.tensor(self.label_mapping[label], dtype=torch.long)
        return graph, torch.as_tensor(raw_label, dtype=torch.float32).reshape(1)


@dataclass
class DatasetBundle:
    train: AdaptedGraphDataset
    validation: AdaptedGraphDataset
    test: AdaptedGraphDataset
    task_type: Literal["classification", "regression"]
    input_dim: int
    output_dim: int


def _random_split_indices(
    length: int, val_ratio: float, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than one")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(length)
    validation_count = max(1, int(length * val_ratio))
    test_count = max(1, int(length * test_ratio))
    train_count = length - validation_count - test_count
    if train_count < 1:
        raise ValueError("The requested split leaves no training graphs")
    return (
        indices[:train_count],
        indices[train_count : train_count + validation_count],
        indices[train_count + validation_count :],
    )


def load_graph_dataset(config: GraphConfig) -> DatasetBundle:
    from dgl.data import TUDataset, ZINCDataset

    name = config.dataset.lower()
    if name == "zinc":
        train_base = ZINCDataset(mode="train", raw_dir=config.data_dir)
        validation_base = ZINCDataset(mode="valid", raw_dir=config.data_dir)
        test_base = ZINCDataset(mode="test", raw_dir=config.data_dir)
        input_dim = int(train_base.num_atom_types)
        return DatasetBundle(
            train=AdaptedGraphDataset(
                train_base,
                np.arange(len(train_base)),
                "categorical",
                input_dim,
                "regression",
            ),
            validation=AdaptedGraphDataset(
                validation_base,
                np.arange(len(validation_base)),
                "categorical",
                input_dim,
                "regression",
            ),
            test=AdaptedGraphDataset(
                test_base,
                np.arange(len(test_base)),
                "categorical",
                input_dim,
                "regression",
            ),
            task_type="regression",
            input_dim=input_dim,
            output_dim=1,
        )

    allowed_tu = {"mutag", "proteins", "enzymes"}
    if name not in allowed_tu:
        raise ValueError(f"Unknown dataset {config.dataset!r}; choose from {sorted(allowed_tu | {'zinc'})}.")
    base = TUDataset(name.upper(), raw_dir=config.data_dir)
    first_graph, _ = base[0]
    if "feat" in first_graph.ndata:
        first_features = first_graph.ndata["feat"]
        input_dim = 1 if first_features.ndim == 1 else int(np.prod(first_features.shape[1:]))
        feature_mode: Literal["existing", "categorical", "degree"] = "existing"
    else:
        input_dim = 1
        feature_mode = "degree"
    raw_labels = [int(torch.as_tensor(base[index][1]).reshape(-1)[0].item()) for index in range(len(base))]
    unique_labels = sorted(set(raw_labels))
    mapping = {label: index for index, label in enumerate(unique_labels)}
    train_indices, validation_indices, test_indices = _random_split_indices(
        len(base), config.val_ratio, config.test_ratio, config.seed
    )
    adapter_kwargs = {
        "base": base,
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "task_type": "classification",
        "label_mapping": mapping,
    }
    return DatasetBundle(
        train=AdaptedGraphDataset(indices=train_indices, **adapter_kwargs),
        validation=AdaptedGraphDataset(indices=validation_indices, **adapter_kwargs),
        test=AdaptedGraphDataset(indices=test_indices, **adapter_kwargs),
        task_type="classification",
        input_dim=input_dim,
        output_dim=len(unique_labels),
    )


def make_loader(
    dataset: AdaptedGraphDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> GraphDataLoader:
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )


def batch_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
) -> torch.Tensor:
    if task_type == "classification":
        return F.cross_entropy(predictions, labels.long().reshape(-1))
    return F.mse_loss(predictions.reshape(-1), labels.float().reshape(-1))


@torch.no_grad()
def evaluate(
    model: GraphClassifier,
    loader: GraphDataLoader,
    task_type: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for graph, labels in loader:
        graph = graph.to(device)
        labels = labels.to(device)
        predictions = model(graph, graph.ndata["x"])
        loss = batch_loss(predictions, labels, task_type)
        examples = labels.shape[0]
        total_loss += float(loss.item()) * examples
        total_examples += examples
        predictions_all.append(predictions.detach().cpu())
        labels_all.append(labels.detach().cpu())
    predictions_cpu = torch.cat(predictions_all)
    labels_cpu = torch.cat(labels_all)
    metrics = {"loss": total_loss / max(total_examples, 1)}
    if task_type == "classification":
        metrics["accuracy"] = float(
            (predictions_cpu.argmax(dim=-1) == labels_cpu.long().reshape(-1)).float().mean().item()
        )
    else:
        metrics["mae"] = float(
            (predictions_cpu.reshape(-1) - labels_cpu.reshape(-1)).abs().mean().item()
        )
    return metrics


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def run_training(config: GraphConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    ensure_dgl_device(device)
    bundle = load_graph_dataset(config)
    workers = default_num_workers(device) if config.num_workers is None else config.num_workers
    if config.mode == "full":
        train_batch_size = len(bundle.train)
    elif config.mode == "mini":
        train_batch_size = config.batch_size
    else:
        raise ValueError("mode must be either 'full' or 'mini'")
    train_loader = make_loader(bundle.train, train_batch_size, True, workers)
    validation_loader = make_loader(
        bundle.validation, min(config.eval_batch_size, len(bundle.validation)), False, workers
    )
    test_loader = make_loader(
        bundle.test, min(config.eval_batch_size, len(bundle.test)), False, workers
    )

    model = GraphClassifier(
        model=config.model,
        in_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        out_dim=bundle.output_dim,
        num_layers=config.layers,
        dropout=config.dropout,
        heads=config.heads,
        pooling=config.pooling,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_state = copy.deepcopy(_cpu_state_dict(model))
    best_selection = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    total_train_seconds = 0.0
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss = 0.0
        examples_seen = 0
        with Timer(device) as timer:
            for graph, labels in train_loader:
                graph = graph.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(graph, graph.ndata["x"])
                loss = batch_loss(predictions, labels, bundle.task_type)
                loss.backward()
                optimizer.step()
                examples = labels.shape[0]
                weighted_loss += float(loss.item()) * examples
                examples_seen += examples
        total_train_seconds += timer.seconds
        train_loss = weighted_loss / max(examples_seen, 1)
        validation = evaluate(model, validation_loader, bundle.task_type, device)
        selection = (
            validation["accuracy"]
            if bundle.task_type == "classification"
            else -validation["mae"]
        )
        entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "train_seconds": timer.seconds,
        }
        entry.update({f"val_{key}": value for key, value in validation.items() if key != "loss"})
        history.append(entry)
        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % config.log_every == 0:
            metric_name = "accuracy" if bundle.task_type == "classification" else "mae"
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.4f} "
                f"val_loss={validation['loss']:.4f} val_{metric_name}={validation[metric_name]:.4f} "
                f"train_s={timer.seconds:.3f}"
            )
        if config.patience > 0 and stale_epochs >= config.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    final_validation = evaluate(model, validation_loader, bundle.task_type, device)
    test_metrics = evaluate(model, test_loader, bundle.task_type, device)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    run_stem = (
        f"{config.dataset}-{config.model}-{config.pooling}-{config.mode}-seed{config.seed}"
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
        "task": "graph_prediction",
        "task_type": bundle.task_type,
        "config": asdict(config),
        "dataset": {
            "train_graphs": len(bundle.train),
            "validation_graphs": len(bundle.validation),
            "test_graphs": len(bundle.test),
            "input_dim": bundle.input_dim,
            "output_dim": bundle.output_dim,
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
                "task_type": bundle.task_type,
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
    valid_fields = GraphConfig.__dataclass_fields__.keys()
    config = GraphConfig(**{key: value for key, value in raw_config.items() if key in valid_fields})
    bundle = load_graph_dataset(config)
    workers = default_num_workers(device) if config.num_workers is None else config.num_workers
    test_loader = make_loader(
        bundle.test, min(config.eval_batch_size, len(bundle.test)), False, workers
    )
    model = GraphClassifier(
        model=config.model,
        in_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        out_dim=bundle.output_dim,
        num_layers=config.layers,
        dropout=config.dropout,
        heads=config.heads,
        pooling=config.pooling,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    metrics = evaluate(model, test_loader, bundle.task_type, device)
    result = {
        "checkpoint": str(checkpoint),
        "dataset": config.dataset,
        "model": config.model,
        "pooling": config.pooling,
        "mode": config.mode,
        "task_type": bundle.task_type,
        "device": str(device),
        "test": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGL graph classification/regression benchmark")
    parser.add_argument("--dataset", choices=("mutag", "proteins", "enzymes", "zinc"), default="mutag")
    parser.add_argument("--model", choices=SUPPORTED_GNN_MODELS, default="gcn")
    parser.add_argument("--pooling", choices=SUPPORTED_POOLING, default="avg")
    parser.add_argument("--mode", choices=("full", "mini"), default="mini")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
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
    config_keys = GraphConfig.__dataclass_fields__.keys()
    run_training(GraphConfig(**{key: value for key, value in vars(args).items() if key in config_keys}))


if __name__ == "__main__":
    main()
