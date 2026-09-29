from __future__ import annotations

import dgl
import pytest
import torch
from dgl.dataloading import MultiLayerNeighborSampler

from common.models import GNNEncoder, GraphClassifier, SUPPORTED_GNN_MODELS


@pytest.fixture
def graph() -> dgl.DGLGraph:
    source = torch.tensor([0, 1, 2, 3, 0, 2])
    destination = torch.tensor([1, 2, 3, 0, 2, 0])
    return dgl.add_self_loop(dgl.graph((source, destination), num_nodes=4))


@pytest.mark.parametrize("model_name", SUPPORTED_GNN_MODELS)
def test_encoder_full_and_sampled_shapes(graph: dgl.DGLGraph, model_name: str) -> None:
    features = torch.randn(4, 5)
    model = GNNEncoder(model_name, 5, 8, 3, num_layers=2, dropout=0.0, heads=2)
    assert model(graph, features).shape == (4, 3)

    sampler = MultiLayerNeighborSampler([2, 2])
    input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, torch.tensor([0, 1]))
    sampled_output = model(blocks, features[input_nodes])
    assert sampled_output.shape == (len(output_nodes), 3)


@pytest.mark.parametrize("pooling", ("avg", "max", "min"))
def test_graph_classifier_pooling_shape(graph: dgl.DGLGraph, pooling: str) -> None:
    batched = dgl.batch([graph, graph])
    features = torch.randn(batched.num_nodes(), 5)
    model = GraphClassifier(
        "gcn", 5, 8, 2, num_layers=2, dropout=0.0, pooling=pooling
    )
    assert model(batched, features).shape == (2, 2)

