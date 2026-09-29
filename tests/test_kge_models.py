from __future__ import annotations

import pytest
import torch

from task4_knowledge_graph.code.kge_models import SUPPORTED_KGE_MODELS, make_kge_model


@pytest.mark.parametrize("model_name", SUPPORTED_KGE_MODELS)
def test_kge_score_shapes(model_name: str) -> None:
    model = make_kge_model(
        model_name,
        num_entities=7,
        num_relations=6,
        embedding_dim=12,
        embedding_height=3,
        conve_channels=4,
        dropout=0.0,
    )
    model.eval()
    head = torch.tensor([0, 1])
    relation = torch.tensor([0, 4])
    tail = torch.tensor([2, 3])
    assert model.score_triples(head, relation, tail).shape == (2,)
    assert model.score_all_tails(head, relation, candidate_chunk_size=3).shape == (2, 7)

