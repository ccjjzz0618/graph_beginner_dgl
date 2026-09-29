"""Shared utilities and DGL model components for all four tasks."""

import os
import warnings

os.environ.setdefault("DGLBACKEND", "pytorch")
warnings.filterwarnings(
    "ignore", message="Dataloader CPU affinity opt is not enabled.*"
)

from .models import GNNEncoder, GraphClassifier, DotProductPredictor
from .utils import resolve_device, save_json, seed_everything

__all__ = [
    "DotProductPredictor",
    "GNNEncoder",
    "GraphClassifier",
    "resolve_device",
    "save_json",
    "seed_everything",
]
