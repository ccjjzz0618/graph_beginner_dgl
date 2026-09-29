from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dgl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.utils import (
    Timer,
    count_parameters,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
    timestamped_path,
)
from task4_knowledge_graph.code.kge_models import (
    KGEModel,
    SUPPORTED_KGE_MODELS,
    make_kge_model,
)


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
RESULT_DIRECTORY = Path(__file__).resolve().parents[1] / "results"


@dataclass
class KGConfig:
    dataset: str = "fb15k237"
    model: str = "transe"
    epochs: int = 100
    embedding_dim: int = 200
    batch_size: int = 1024
    negative_samples: int = 32
    learning_rate: float = 0.0005
    weight_decay: float = 0.0
    regularization: float = 1e-6
    margin: float = 9.0
    dropout: float = 0.2
    embedding_height: int = 10
    conve_channels: int = 32
    conve_kernel_size: int = 3
    eval_batch_size: int = 64
    candidate_chunk_size: int = 2048
    eval_every: int = 5
    eval_limit: int = 2000
    patience: int = 5
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 1
    data_dir: str = str(DATA_DIRECTORY)
    output: str | None = None
    checkpoint_out: str | None = None


@dataclass
class KGDatasetBundle:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    num_entities: int
    num_relations: int


def _find_mask(graph: dgl.DGLGraph, split: str) -> torch.Tensor:
    candidates = {
        "train": ("train_edge_mask", "train_mask"),
        "validation": (
            "valid_edge_mask",
            "val_edge_mask",
            "validation_edge_mask",
            "valid_mask",
            "val_mask",
        ),
        "test": ("test_edge_mask", "test_mask"),
    }[split]
    for key in candidates:
        if key in graph.edata:
            return graph.edata[key].bool()
    raise KeyError(f"Could not find a {split} edge mask; available keys: {list(graph.edata.keys())}")


def _triples_from_mask(graph: dgl.DGLGraph, mask: torch.Tensor) -> torch.Tensor:
    edge_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    source, destination = graph.find_edges(edge_ids)
    # DGL 2.2 Windows uses ``etype`` while newer generated docs show ``e_type``.
    relation_key = "e_type" if "e_type" in graph.edata else "etype"
    relation = graph.edata[relation_key][edge_ids].long()
    return torch.stack((source.long(), relation, destination.long()), dim=1)


def load_kg_dataset(name: str, raw_dir: str | Path) -> KGDatasetBundle:
    from dgl.data import FB15k237Dataset, FB15kDataset, WN18Dataset

    datasets: dict[str, Any] = {
        "fb15k237": FB15k237Dataset,
        "fb15k": FB15kDataset,
        "wn18": WN18Dataset,
    }
    normalized = name.lower()
    if normalized not in datasets:
        raise ValueError(f"Unknown dataset {name!r}; choose from {tuple(datasets)}.")
    dataset = datasets[normalized](reverse=False, raw_dir=str(raw_dir), verbose=True)
    graph = dataset[0]
    return KGDatasetBundle(
        train=_triples_from_mask(graph, _find_mask(graph, "train")),
        validation=_triples_from_mask(graph, _find_mask(graph, "validation")),
        test=_triples_from_mask(graph, _find_mask(graph, "test")),
        num_entities=int(dataset.num_nodes),
        num_relations=int(dataset.num_rels),
    )


def add_reciprocal_triples(triples: torch.Tensor, num_relations: int) -> torch.Tensor:
    inverse = torch.stack(
        (triples[:, 2], triples[:, 1] + num_relations, triples[:, 0]), dim=1
    )
    return torch.cat((triples, inverse), dim=0)


def build_filter_map(
    bundle: KGDatasetBundle,
) -> dict[tuple[int, int], set[int]]:
    filters: dict[tuple[int, int], set[int]] = defaultdict(set)
    all_triples = torch.cat((bundle.train, bundle.validation, bundle.test), dim=0)
    base_relations = bundle.num_relations
    for head, relation, tail in all_triples.tolist():
        filters[(head, relation)].add(tail)
        filters[(tail, relation + base_relations)].add(head)
    return dict(filters)


def _filtered_ranks(
    scores: torch.Tensor,
    query_entities: torch.Tensor,
    relations: torch.Tensor,
    targets: torch.Tensor,
    filters: dict[tuple[int, int], set[int]],
) -> torch.Tensor:
    target_scores = scores.gather(1, targets[:, None]).squeeze(1).clone()
    for row, (query, relation, target) in enumerate(
        zip(query_entities.tolist(), relations.tolist(), targets.tolist())
    ):
        filtered = filters.get((query, relation))
        if filtered:
            filtered_without_target = [entity for entity in filtered if entity != target]
            if filtered_without_target:
                indices = torch.tensor(
                    filtered_without_target, dtype=torch.long, device=scores.device
                )
                scores[row, indices] = -torch.inf
    return 1 + (scores > target_scores[:, None]).sum(dim=1)


@torch.no_grad()
def evaluate_filtered(
    model: KGEModel,
    triples: torch.Tensor,
    filters: dict[tuple[int, int], set[int]],
    base_relations: int,
    device: torch.device,
    batch_size: int,
    candidate_chunk_size: int,
    limit: int,
    seed: int,
) -> dict[str, float | int]:
    model.eval()
    if limit > 0 and len(triples) > limit:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(len(triples), generator=generator)[:limit]
        triples = triples[selected]
    ranks_all: list[torch.Tensor] = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start : start + batch_size].to(device)
        head, relation, tail = batch.unbind(dim=1)

        tail_scores = model.score_all_tails(
            head, relation, candidate_chunk_size=candidate_chunk_size
        )
        tail_ranks = _filtered_ranks(
            tail_scores, head, relation, tail, filters
        )

        inverse_relation = relation + base_relations
        head_scores = model.score_all_tails(
            tail, inverse_relation, candidate_chunk_size=candidate_chunk_size
        )
        head_ranks = _filtered_ranks(
            head_scores, tail, inverse_relation, head, filters
        )
        ranks_all.extend((tail_ranks.cpu(), head_ranks.cpu()))

    ranks = torch.cat(ranks_all).float()
    return {
        "evaluated_triples": len(triples),
        "mean_rank": float(ranks.mean().item()),
        "mrr": float((1.0 / ranks).mean().item()),
        "hits_at_1": float((ranks <= 1).float().mean().item()),
        "hits_at_3": float((ranks <= 3).float().mean().item()),
        "hits_at_10": float((ranks <= 10).float().mean().item()),
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _build_model(config: KGConfig, bundle: KGDatasetBundle, device: torch.device) -> KGEModel:
    # Reciprocal training doubles the relation vocabulary and turns head prediction into tail prediction.
    return make_kge_model(
        config.model,
        num_entities=bundle.num_entities,
        num_relations=2 * bundle.num_relations,
        embedding_dim=config.embedding_dim,
        margin=config.margin,
        embedding_height=config.embedding_height,
        conve_channels=config.conve_channels,
        conve_kernel_size=config.conve_kernel_size,
        dropout=config.dropout,
    ).to(device)


def run_training(config: KGConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    bundle = load_kg_dataset(config.dataset, config.data_dir)
    filters = build_filter_map(bundle)
    train_triples = add_reciprocal_triples(bundle.train, bundle.num_relations)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_triples),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = _build_model(config, bundle, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_state = copy.deepcopy(_cpu_state_dict(model))
    best_mrr = float("-inf")
    best_epoch = 0
    stale_evaluations = 0
    total_train_seconds = 0.0
    history: list[dict[str, float | int]] = []
    final_validation: dict[str, float | int] | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        weighted_loss = 0.0
        examples_seen = 0
        with Timer(device) as timer:
            for (batch_cpu,) in loader:
                batch = batch_cpu.to(device, non_blocking=True)
                head, relation, tail = batch.unbind(dim=1)
                negative_tail = torch.randint(
                    0,
                    bundle.num_entities,
                    (len(batch), config.negative_samples),
                    device=device,
                )
                repeated_head = head[:, None].expand_as(negative_tail).reshape(-1)
                repeated_relation = relation[:, None].expand_as(negative_tail).reshape(-1)
                optimizer.zero_grad(set_to_none=True)
                positive_scores = model.score_triples(head, relation, tail)
                negative_scores = model.score_triples(
                    repeated_head, repeated_relation, negative_tail.reshape(-1)
                ).reshape(len(batch), config.negative_samples)
                loss = F.softplus(-positive_scores).mean() + F.softplus(negative_scores).mean()
                if config.regularization > 0:
                    loss = loss + config.regularization * model.regularization_loss()
                loss.backward()
                optimizer.step()
                model.post_step()
                weighted_loss += float(loss.item()) * len(batch)
                examples_seen += len(batch)
        total_train_seconds += timer.seconds
        train_loss = weighted_loss / max(examples_seen, 1)
        entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": timer.seconds,
        }

        should_evaluate = epoch % config.eval_every == 0 or epoch == config.epochs
        if should_evaluate:
            final_validation = evaluate_filtered(
                model,
                bundle.validation,
                filters,
                bundle.num_relations,
                device,
                config.eval_batch_size,
                config.candidate_chunk_size,
                config.eval_limit,
                config.seed,
            )
            entry.update({f"val_{key}": value for key, value in final_validation.items()})
            current_mrr = float(final_validation["mrr"])
            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_epoch = epoch
                best_state = _cpu_state_dict(model)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
        history.append(entry)

        if epoch == 1 or epoch % config.log_every == 0:
            validation_text = (
                f" val_mrr={final_validation['mrr']:.4f}"
                if should_evaluate and final_validation is not None
                else ""
            )
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.4f} "
                f"train_s={timer.seconds:.3f}{validation_text}"
            )
        if (
            should_evaluate
            and config.patience > 0
            and stale_evaluations >= config.patience
        ):
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    final_validation = evaluate_filtered(
        model,
        bundle.validation,
        filters,
        bundle.num_relations,
        device,
        config.eval_batch_size,
        config.candidate_chunk_size,
        config.eval_limit,
        config.seed,
    )
    test_metrics = evaluate_filtered(
        model,
        bundle.test,
        filters,
        bundle.num_relations,
        device,
        config.eval_batch_size,
        config.candidate_chunk_size,
        config.eval_limit,
        config.seed + 1,
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    run_stem = f"{config.dataset}-{config.model}-seed{config.seed}"
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
        "task": "knowledge_graph_completion",
        "config": asdict(config),
        "dataset": {
            "num_entities": bundle.num_entities,
            "num_relations": bundle.num_relations,
            "train_triples": len(bundle.train),
            "validation_triples": len(bundle.validation),
            "test_triples": len(bundle.test),
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


def evaluate_checkpoint(
    checkpoint: str | Path,
    device_name: str = "auto",
    split_name: str = "test",
    eval_limit: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    payload = load_checkpoint(checkpoint, device)
    metadata = payload["metadata"]
    raw_config = metadata["config"]
    valid_fields = KGConfig.__dataclass_fields__.keys()
    config = KGConfig(**{key: value for key, value in raw_config.items() if key in valid_fields})
    if eval_limit is not None:
        config.eval_limit = eval_limit
    bundle = load_kg_dataset(config.dataset, config.data_dir)
    model = _build_model(config, bundle, device)
    model.load_state_dict(payload["model_state"])
    triples = bundle.test if split_name == "test" else bundle.validation
    metrics = evaluate_filtered(
        model,
        triples,
        build_filter_map(bundle),
        bundle.num_relations,
        device,
        config.eval_batch_size,
        config.candidate_chunk_size,
        config.eval_limit,
        config.seed + (1 if split_name == "test" else 0),
    )
    result = {
        "checkpoint": str(checkpoint),
        "dataset": config.dataset,
        "model": config.model,
        "split": split_name,
        "device": str(device),
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGL-dataset knowledge-graph completion")
    parser.add_argument("--dataset", choices=("fb15k237", "fb15k", "wn18"), default="fb15k237")
    parser.add_argument("--model", choices=SUPPORTED_KGE_MODELS, default="transe")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--negative-samples", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--margin", type=float, default=9.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding-height", type=int, default=10)
    parser.add_argument("--conve-channels", type=int, default=32)
    parser.add_argument("--conve-kernel-size", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=2000,
        help="Triples per validation/test split; 0 evaluates the complete split.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--data-dir", default=str(DATA_DIRECTORY))
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-split", choices=("validation", "test"), default="test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval-only requires --checkpoint")
        evaluate_checkpoint(
            args.checkpoint, args.device, args.eval_split, eval_limit=args.eval_limit
        )
        return
    config_keys = KGConfig.__dataclass_fields__.keys()
    run_training(KGConfig(**{key: value for key, value in vars(args).items() if key in config_keys}))


if __name__ == "__main__":
    main()
