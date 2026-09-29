from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_KGE_MODELS = ("transe", "rotate", "conve")


class KGEModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

    def score_triples(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def score_candidate_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def score_all_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        candidate_chunk_size: int = 2048,
    ) -> torch.Tensor:
        chunks = []
        for start in range(0, self.num_entities, candidate_chunk_size):
            candidates = torch.arange(
                start,
                min(start + candidate_chunk_size, self.num_entities),
                device=head.device,
            )
            chunks.append(self.score_candidate_tails(head, relation, candidates))
        return torch.cat(chunks, dim=1)

    def regularization_loss(self) -> torch.Tensor:
        parameters = [parameter for parameter in self.parameters() if parameter.ndim >= 2]
        return sum(parameter.abs().pow(3).mean() for parameter in parameters)

    def post_step(self) -> None:
        return


class TransE(KGEModel):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 9.0,
        norm: int = 1,
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.norm = norm
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        bound = 6.0 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def score_triples(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        distance = self.entity(head) + self.relation(relation) - self.entity(tail)
        return self.margin - torch.linalg.vector_norm(distance, ord=self.norm, dim=-1)

    def score_candidate_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        query = self.entity(head) + self.relation(relation)
        candidates = self.entity(candidate_ids)
        distance = torch.cdist(query, candidates, p=float(self.norm))
        return self.margin - distance

    def post_step(self) -> None:
        with torch.no_grad():
            self.entity.weight.data = F.normalize(self.entity.weight.data, p=2, dim=-1)


class RotatE(KGEModel):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 12.0,
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.entity = nn.Embedding(num_entities, 2 * embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        bound = (margin + 2.0) / embedding_dim
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -math.pi, math.pi)

    def _rotate(
        self, entity: torch.Tensor, relation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real, imaginary = entity.chunk(2, dim=-1)
        phase = math.pi * torch.tanh(relation)
        cosine, sine = torch.cos(phase), torch.sin(phase)
        return real * cosine - imaginary * sine, real * sine + imaginary * cosine

    def score_triples(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        rotated_real, rotated_imaginary = self._rotate(
            self.entity(head), self.relation(relation)
        )
        tail_real, tail_imaginary = self.entity(tail).chunk(2, dim=-1)
        distance = torch.sqrt(
            (rotated_real - tail_real).pow(2)
            + (rotated_imaginary - tail_imaginary).pow(2)
            + 1e-9
        ).sum(dim=-1)
        return self.margin - distance

    def score_candidate_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        rotated_real, rotated_imaginary = self._rotate(
            self.entity(head), self.relation(relation)
        )
        candidate_real, candidate_imaginary = self.entity(candidate_ids).chunk(2, dim=-1)
        real_delta = rotated_real[:, None, :] - candidate_real[None, :, :]
        imaginary_delta = rotated_imaginary[:, None, :] - candidate_imaginary[None, :, :]
        distance = torch.sqrt(real_delta.pow(2) + imaginary_delta.pow(2) + 1e-9).sum(dim=-1)
        return self.margin - distance


class ConvE(KGEModel):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        embedding_height: int = 10,
        channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim)
        if embedding_dim % embedding_height != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by embedding_height={embedding_height}"
            )
        self.embedding_height = embedding_height
        self.embedding_width = embedding_dim // embedding_height
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout2d(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        self.convolution = nn.Conv2d(
            1, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        flattened = channels * 2 * self.embedding_height * self.embedding_width
        self.projection = nn.Linear(flattened, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.bias = nn.Parameter(torch.zeros(num_entities))
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def _query(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        head_embedding = self.entity(head).view(
            -1, 1, self.embedding_height, self.embedding_width
        )
        relation_embedding = self.relation(relation).view(
            -1, 1, self.embedding_height, self.embedding_width
        )
        stacked = torch.cat((head_embedding, relation_embedding), dim=2)
        hidden = self.input_dropout(stacked)
        hidden = F.relu(self.convolution(hidden))
        hidden = self.feature_dropout(hidden)
        hidden = hidden.flatten(start_dim=1)
        hidden = self.hidden_dropout(self.projection(hidden))
        return F.relu(self.layer_norm(hidden))

    def score_triples(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        query = self._query(head, relation)
        return (query * self.entity(tail)).sum(dim=-1) + self.bias[tail]

    def score_candidate_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        query = self._query(head, relation)
        return query @ self.entity(candidate_ids).T + self.bias[candidate_ids]


def make_kge_model(
    name: str,
    num_entities: int,
    num_relations: int,
    embedding_dim: int,
    *,
    margin: float = 9.0,
    embedding_height: int = 10,
    conve_channels: int = 32,
    conve_kernel_size: int = 3,
    dropout: float = 0.2,
) -> KGEModel:
    normalized = name.lower()
    if normalized == "transe":
        return TransE(num_entities, num_relations, embedding_dim, margin=margin)
    if normalized == "rotate":
        return RotatE(num_entities, num_relations, embedding_dim, margin=margin)
    if normalized == "conve":
        return ConvE(
            num_entities,
            num_relations,
            embedding_dim,
            embedding_height=embedding_height,
            channels=conve_channels,
            kernel_size=conve_kernel_size,
            dropout=dropout,
        )
    raise ValueError(f"Unknown KGE model {name!r}; choose from {SUPPORTED_KGE_MODELS}.")

