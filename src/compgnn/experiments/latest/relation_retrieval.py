"""Explicit relation-conditioned node retrieval and local graph extraction.

The central path implemented here is::

    (query, relation) -> transformed query -> concat with raw query
    -> attention over candidate nodes -> selected node(s) -> graph expansion

Relations can be fixed labels or oracle labels.  The module deliberately keeps
the graph traversal separate from the neural retriever so that experiments can
measure retrieval quality without conflating it with graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Edge:
    """A directed labelled edge in the candidate graph."""

    source: int
    target: int
    relation: str


@dataclass
class RetrievalStep:
    """One relation-conditioned retrieval decision."""

    relation: str
    selected: torch.Tensor
    scores: torch.Tensor
    attention: torch.Tensor
    transformed_query: torch.Tensor


@dataclass
class RetrievedSubgraph:
    """The local graph returned after retrieval and relation-aware expansion."""

    nodes: list[int]
    edges: list[Edge]
    steps: list[RetrievalStep]


class RelationConditionedRetriever(nn.Module):
    """Retrieve nodes using both semantic and explicit relation information.

    ``query`` and every row of ``candidate_nodes`` must have the same input
    dimension.  The raw query is retained in the attention input so the model
    can fall back to ordinary semantic retrieval when a relation is noisy.
    """

    def __init__(
        self,
        input_dim: int,
        relation_names: Sequence[str],
        hidden_dim: int | None = None,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if not relation_names:
            raise ValueError("relation_names must not be empty")
        if len(set(relation_names)) != len(relation_names):
            raise ValueError("relation_names must be unique")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        hidden_dim = hidden_dim or input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.relation_to_id = {name: i for i, name in enumerate(relation_names)}
        self.relation_embedding = nn.Embedding(len(relation_names), hidden_dim)
        self.query_transform = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # The attention query sees [raw query, relation-transformed query].
        self.attention_query = nn.Linear(input_dim * 2, hidden_dim, bias=False)
        self.node_key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.node_value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim)

    def _relation_id(self, relation: str | int, device: torch.device) -> torch.Tensor:
        if isinstance(relation, str):
            try:
                relation = self.relation_to_id[relation]
            except KeyError as exc:
                known = ", ".join(self.relation_to_id)
                raise ValueError(f"unknown relation {relation!r}; expected one of: {known}") from exc
        if not isinstance(relation, int) or not 0 <= relation < len(self.relation_to_id):
            raise ValueError("relation id is out of range")
        return torch.tensor(relation, device=device)

    def transform_query(self, query: torch.Tensor, relation: str | int) -> torch.Tensor:
        """Produce the relation-specific query representation ``f_r(q)``."""
        if query.ndim != 1 or query.shape[0] != self.input_dim:
            raise ValueError(f"query must have shape [{self.input_dim}]")
        relation_id = self._relation_id(relation, query.device)
        relation_vector = self.relation_embedding(relation_id)
        transformed = self.query_transform(torch.cat((query, relation_vector), dim=-1))
        return F.normalize(transformed, dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        relation: str | int,
        candidate_nodes: torch.Tensor,
        top_k: int = 1,
    ) -> RetrievalStep:
        """Score candidates and return a differentiable attention decision."""
        if candidate_nodes.ndim != 2 or candidate_nodes.shape[1] != self.input_dim:
            raise ValueError(f"candidate_nodes must have shape [N, {self.input_dim}]")
        if candidate_nodes.shape[0] == 0:
            raise ValueError("candidate_nodes must not be empty")
        if not 1 <= top_k <= candidate_nodes.shape[0]:
            raise ValueError("top_k must be between 1 and the number of candidates")

        transformed = self.transform_query(query, relation)
        combined = torch.cat((F.normalize(query, dim=-1), transformed), dim=-1)
        attention_query = self.attention_query(combined)
        keys = self.node_key(F.normalize(candidate_nodes, dim=-1))
        scores = (attention_query @ keys.T) / (keys.shape[-1] ** 0.5 * self.temperature)
        attention = scores.softmax(dim=-1)
        selected = scores.topk(top_k, dim=-1).indices
        return RetrievalStep(
            relation=str(relation),
            selected=selected,
            scores=scores,
            attention=attention,
            transformed_query=transformed,
        )

    def retrieve_sequence(
        self,
        query: torch.Tensor,
        relations: Sequence[str | int],
        candidate_nodes: torch.Tensor,
        top_k: int = 1,
        exclude_selected: bool = True,
    ) -> list[RetrievalStep]:
        """Run successive relation-conditioned retrieval rounds.

        The query is updated with the selected node representation after each
        round.  This models the intended ``find node -> use it next round``
        loop while retaining a differentiable soft attention path.
        """
        if not relations:
            return []
        available = torch.ones(candidate_nodes.shape[0], dtype=torch.bool, device=candidate_nodes.device)
        current_query = query
        steps: list[RetrievalStep] = []
        for relation in relations:
            indices = available.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                break
            step = self(current_query, relation, candidate_nodes[indices], min(top_k, indices.numel()))
            step.selected = indices[step.selected]
            step.scores = step.scores
            steps.append(step)
            selected_nodes = candidate_nodes[step.selected]
            current_query = F.normalize(
                0.5 * current_query + 0.5 * selected_nodes.mean(dim=0), dim=-1
            )
            if exclude_selected:
                available[step.selected] = False
        return steps


def extract_k_hop_subgraph(
    seed_nodes: Iterable[int],
    edges: Sequence[Edge],
    max_hops: int = 3,
    relation_hops: Mapping[str, int] | None = None,
) -> tuple[list[int], list[Edge]]:
    """Extract a labelled local graph using relation-specific hop costs.

    ``relation_hops`` makes the semantic convention explicit.  For example,
    ``{"combine": 1, "support": 1}`` treats both as one graph step, while a
    relation assigned cost 2 reaches the next layer only after two units.
    Edges are traversed in both directions because the discovered graph is
    used as an undirected local evidence group, while original edge labels are
    preserved.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")
    relation_hops = dict(relation_hops or {})
    adjacency: dict[int, list[tuple[int, Edge]]] = {}
    for edge in edges:
        cost = relation_hops.get(edge.relation, 1)
        if cost <= 0:
            raise ValueError("relation hop costs must be positive")
        adjacency.setdefault(edge.source, []).append((edge.target, edge))
        adjacency.setdefault(edge.target, []).append((edge.source, edge))

    distances = {int(node): 0 for node in seed_nodes}
    queue = list(distances)
    while queue:
        source = queue.pop(0)
        for target, edge in adjacency.get(source, []):
            next_distance = distances[source] + relation_hops.get(edge.relation, 1)
            if next_distance <= max_hops and next_distance < distances.get(target, max_hops + 1):
                distances[target] = next_distance
                queue.append(target)

    selected = set(distances)
    selected_edges = [edge for edge in edges if edge.source in selected and edge.target in selected]
    return sorted(selected), selected_edges


def build_retrieved_subgraph(
    retriever: RelationConditionedRetriever,
    query: torch.Tensor,
    relations: Sequence[str | int],
    candidate_nodes: torch.Tensor,
    edges: Sequence[Edge],
    top_k: int = 1,
    max_hops: int = 3,
    relation_hops: Mapping[str, int] | None = None,
) -> RetrievedSubgraph:
    """Retrieve seeds, then return the relation-labelled local evidence graph."""
    steps = retriever.retrieve_sequence(query, relations, candidate_nodes, top_k)
    seeds = {int(node) for step in steps for node in step.selected.tolist()}
    nodes, selected_edges = extract_k_hop_subgraph(seeds, edges, max_hops, relation_hops)
    return RetrievedSubgraph(nodes=nodes, edges=selected_edges, steps=steps)

