from __future__ import annotations

from collections.abc import Sequence

import dgl
import torch
from dgl.nn.pytorch import (
    AvgPooling,
    GATConv,
    GINConv,
    GraphConv,
    MaxPooling,
    SAGEConv,
)
from torch import nn
from torch.nn import functional as F


SUPPORTED_GNN_MODELS = ("gcn", "gat", "graphsage", "gin")
SUPPORTED_POOLING = ("avg", "max", "min")


class GraphLayer(nn.Module):
    """One layer with a common output shape for GCN/GAT/SAGE/GIN."""

    def __init__(
        self,
        kind: str,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kind = kind.lower()
        if kind not in SUPPORTED_GNN_MODELS:
            raise ValueError(f"Unknown GNN model {kind!r}; choose from {SUPPORTED_GNN_MODELS}.")
        self.kind = kind
        if kind == "gcn":
            self.layer = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
        elif kind == "gat":
            self.layer = GATConv(
                in_dim,
                out_dim,
                num_heads=heads,
                feat_drop=dropout,
                attn_drop=dropout,
                allow_zero_in_degree=True,
            )
        elif kind == "graphsage":
            self.layer = SAGEConv(in_dim, out_dim, aggregator_type="mean", feat_drop=dropout)
        else:
            apply_func = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            self.layer = GINConv(apply_func, aggregator_type="sum", init_eps=0.0, learn_eps=True)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        output = self.layer(graph, features)
        if self.kind == "gat":
            output = output.mean(dim=1)
        return output


class GNNEncoder(nn.Module):
    """DGL encoder that supports both a full graph and sampled message-flow blocks."""

    def __init__(
        self,
        model: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.model_name = model.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        dimensions = [in_dim]
        if num_layers > 1:
            dimensions.extend([hidden_dim] * (num_layers - 1))
        dimensions.append(out_dim)
        self.layers = nn.ModuleList(
            GraphLayer(
                self.model_name,
                dimensions[index],
                dimensions[index + 1],
                heads=heads,
                dropout=dropout,
            )
            for index in range(num_layers)
        )

    def forward(
        self,
        graph_or_blocks: dgl.DGLGraph | Sequence[dgl.DGLGraph],
        features: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(graph_or_blocks, Sequence):
            if len(graph_or_blocks) != self.num_layers:
                raise ValueError(
                    f"Expected {self.num_layers} sampled blocks, got {len(graph_or_blocks)}."
                )
            graphs = list(graph_or_blocks)
        else:
            graphs = [graph_or_blocks] * self.num_layers

        hidden = features
        for index, (layer, graph) in enumerate(zip(self.layers, graphs)):
            hidden = layer(graph, hidden)
            if index + 1 != self.num_layers:
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden


class MinPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_pool = MaxPooling()

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        return -self.max_pool(graph, -features)


def make_pooling(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "avg":
        return AvgPooling()
    if kind == "max":
        return MaxPooling()
    if kind == "min":
        return MinPooling()
    raise ValueError(f"Unknown pooling {kind!r}; choose from {SUPPORTED_POOLING}.")


class GraphClassifier(nn.Module):
    def __init__(
        self,
        model: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        heads: int = 4,
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(
            model=model,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
        )
        self.pool = make_pooling(pooling)
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        node_embeddings = self.encoder(graph, features)
        graph_embeddings = self.pool(graph, node_embeddings)
        return self.output(graph_embeddings)


class DotProductPredictor(nn.Module):
    def forward(self, graph: dgl.DGLGraph, embeddings: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata["h"] = embeddings
            graph.apply_edges(dgl.function.u_dot_v("h", "h", "score"))
            return graph.edata["score"].squeeze(-1)

    @staticmethod
    def score_pairs(
        embeddings: torch.Tensor, source: torch.Tensor, destination: torch.Tensor
    ) -> torch.Tensor:
        return (embeddings[source] * embeddings[destination]).sum(dim=-1)

