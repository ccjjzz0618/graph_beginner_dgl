from __future__ import annotations

import dgl
import torch

from task2_link_prediction.code.train import build_training_graph, split_undirected_edges


def pair_set(pairs: torch.Tensor) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in pairs.tolist()}


def test_split_keeps_reverse_edges_together() -> None:
    undirected = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 4)]
    source = torch.tensor([a for a, b in undirected] + [b for a, b in undirected])
    destination = torch.tensor([b for a, b in undirected] + [a for a, b in undirected])
    graph = dgl.graph((source, destination), num_nodes=5)
    split = split_undirected_edges(graph, 0.2, 0.2, seed=7)
    train, validation, test = map(pair_set, (split.train, split.validation, split.test))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(undirected)


def test_reverse_edge_ids_cover_graph() -> None:
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3]])
    graph, target_ids, reverse_ids = build_training_graph(4, pairs)
    assert len(reverse_ids) == graph.num_edges()
    assert torch.equal(reverse_ids[reverse_ids], torch.arange(graph.num_edges()))
    assert target_ids.tolist() == [0, 1, 2]

