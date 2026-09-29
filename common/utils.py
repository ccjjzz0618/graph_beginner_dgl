from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch and DGL without requiring a CUDA build."""
    os.environ.setdefault("DGLBACKEND", "pytorch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import dgl

        dgl.seed(seed)
        dgl.random.seed(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a CUDA device. "
            "Run scripts/check_environment.py and install requirements-gpu-cu121.txt."
        )
    return torch.device(requested)


def ensure_dgl_device(device: torch.device) -> None:
    """Fail early when PyTorch is CUDA-enabled but the installed DGL wheel is not."""
    if device.type != "cuda":
        return
    import dgl

    try:
        dgl.graph(([0], [0])).to(device)
    except Exception as exc:  # DGL raises different exception types across versions.
        raise RuntimeError(
            "The installed DGL wheel has no usable CUDA backend. Install the CUDA "
            "wheel with requirements-gpu-cu121.txt, or pass --device cpu."
        ) from exc


def parse_int_list(value: str | Iterable[int], expected: int | None = None) -> list[int]:
    if isinstance(value, str):
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        result = [int(item) for item in value]
    if not result:
        raise ValueError("Expected at least one integer.")
    if expected is not None:
        if len(result) == 1:
            result = result * expected
        elif len(result) != expected:
            raise ValueError(f"Expected 1 or {expected} values, got {len(result)}.")
    return result


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.started_at = 0.0

    def __enter__(self) -> "Timer":
        synchronize(self.device)
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        synchronize(self.device)
        self.seconds = time.perf_counter() - self.started_at


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_json(payload: dict[str, Any], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def timestamped_path(directory: str | Path, stem: str, suffix: str = ".json") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(directory) / f"{stem}-{stamp}{suffix}"


def save_checkpoint(
    output: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metadata: dict[str, Any],
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "metadata": _jsonable(metadata),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)


def default_num_workers(device: torch.device) -> int:
    # DGL graph loaders are most portable on Windows with worker processes disabled.
    return 0 if os.name == "nt" else min(4, os.cpu_count() or 1)

