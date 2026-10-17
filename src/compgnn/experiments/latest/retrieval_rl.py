"""REINFORCE training for relation-conditioned node selection.

This is deliberately separate from supervised retrieval.  The environment
provides graph ground truth for reward calculation, while the policy samples
the next node from the relation-conditioned attention distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.distributions import Categorical

from .relation_retrieval import Edge, RelationConditionedRetriever


@dataclass
class RetrievalRLExample:
    query: torch.Tensor
    candidate_nodes: torch.Tensor
    relations: Sequence[str]
    # The oracle node selected at each relation step.
    gold_nodes: Sequence[int]
    edges: Sequence[Edge]
    # Optional multi-objective reward targets.
    target_intent: torch.Tensor | None = None
    target_group: Sequence[int] | None = None
    defense_scorer: Callable[[torch.Tensor], float] | None = None


@dataclass(frozen=True)
class RewardBreakdown:
    node: float
    relation: float
    irrelevant: float
    duplicate: float
    intent: float = 0.0
    group: float = 0.0
    defense: float = 0.0

    @property
    def total(self) -> float:
        return (self.node + self.relation + self.irrelevant + self.duplicate
                + self.intent + self.group + self.defense)


def _group_f1(selected: set[int], target: set[int]) -> float:
    if not target:
        return 0.0
    true_positive = len(selected & target)
    precision = true_positive / max(len(selected), 1)
    recall = true_positive / len(target)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def multi_objective_components(
    selected_nodes: set[int],
    candidate_nodes: torch.Tensor,
    target_intent: torch.Tensor | None,
    target_group: Sequence[int] | None,
    previous_intent_score: float,
    previous_group_score: float,
    defense_reward: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Return incremental intent, group, and terminal-defense rewards."""
    intent_score = previous_intent_score
    if target_intent is not None and selected_nodes:
        representation = candidate_nodes[sorted(selected_nodes)].mean(dim=0)
        intent_score = float(torch.nn.functional.cosine_similarity(
            representation, target_intent.to(candidate_nodes.device), dim=0
        ))
    group_score = previous_group_score
    if target_group is not None:
        group_score = _group_f1(selected_nodes, set(target_group))
    return intent_score - previous_intent_score, group_score - previous_group_score, defense_reward, intent_score, group_score


def score_node_action(
    selected_node: int,
    expected_node: int,
    previous_node: int | None,
    expected_relation: str,
    edges: Sequence[Edge],
) -> RewardBreakdown:
    """Score one node action against graph ground truth.

    The reward is decomposed so W&B/RLAIF can inspect why an action was good or
    bad.  Exact node identity is strongest; relation correctness is rewarded
    independently, which allows useful partial credit.
    """
    node = 1.0 if selected_node == expected_node else 0.0
    relevant_edges = [
        edge for edge in edges
        if previous_node is not None
        and ((edge.source == previous_node and edge.target == selected_node)
             or (edge.target == previous_node and edge.source == selected_node))
    ]
    relation = 0.5 if any(edge.relation == expected_relation for edge in relevant_edges) else 0.0
    irrelevant = -1.0 if selected_node != expected_node and not relevant_edges else 0.0
    duplicate = -0.5 if previous_node == selected_node else 0.0
    return RewardBreakdown(node, relation, irrelevant, duplicate)


def _discounted_returns(rewards: Sequence[torch.Tensor], gamma: float) -> torch.Tensor:
    returns = []
    running = torch.zeros((), device=rewards[0].device)
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return torch.stack(list(reversed(returns)))


def run_rl_epoch(
    model: RelationConditionedRetriever,
    examples: Sequence[RetrievalRLExample],
    optimizer: torch.optim.Optimizer | None = None,
    gamma: float = 0.95,
    scorer: Callable[[int, int, int | None, str, Sequence[Edge]], RewardBreakdown] = score_node_action,
    wandb_run=None,
    step_offset: int = 0,
) -> dict[str, float]:
    """Run one policy-gradient epoch over relation-labelled graph episodes.

    ``scorer`` can be replaced by an RLAIF-backed evaluator.  It must return
    the same decomposed reward object, allowing the training loop and W&B
    metrics to remain unchanged.
    """
    training = optimizer is not None
    model.train(training)
    metrics = {"loss": [], "reward": [], "node_reward": [], "relation_reward": [],
               "irrelevant_penalty": [], "duplicate_penalty": [], "intent_reward": [],
               "group_reward": [], "defense_reward": [], "node_accuracy": []}
    for example_index, example in enumerate(examples):
        if len(example.relations) != len(example.gold_nodes):
            raise ValueError("relations and gold_nodes must have equal length")
        if training:
            optimizer.zero_grad()
        log_probs: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        previous_node = None
        selected_nodes: set[int] = set()
        previous_intent_score = 0.0
        previous_group_score = 0.0
        for relation, expected_node in zip(example.relations, example.gold_nodes):
            available = [i for i in range(example.candidate_nodes.shape[0]) if i not in selected_nodes]
            if not available:
                break
            indices = torch.tensor(available, device=example.candidate_nodes.device)
            step = model(example.query, relation, example.candidate_nodes[indices], top_k=1)
            distribution = Categorical(logits=step.scores)
            action = distribution.sample() if training else step.scores.argmax()
            selected_node = int(indices[action].item())
            reward_parts = scorer(selected_node, int(expected_node), previous_node,
                                  relation, example.edges)
            selected_nodes.add(selected_node)
            intent_reward, group_reward, _, intent_score, group_score = multi_objective_components(
                selected_nodes, example.candidate_nodes, example.target_intent,
                example.target_group, previous_intent_score, previous_group_score
            )
            previous_intent_score, previous_group_score = intent_score, group_score
            reward_parts = RewardBreakdown(
                reward_parts.node, reward_parts.relation, reward_parts.irrelevant,
                reward_parts.duplicate, intent_reward, group_reward, 0.0,
            )
            reward = torch.tensor(reward_parts.total, dtype=step.scores.dtype, device=step.scores.device)
            if training:
                log_probs.append(distribution.log_prob(action))
            rewards.append(reward)
            previous_node = selected_node
            metrics["reward"].append(reward)
            metrics["node_reward"].append(torch.tensor(reward_parts.node))
            metrics["relation_reward"].append(torch.tensor(reward_parts.relation))
            metrics["irrelevant_penalty"].append(torch.tensor(reward_parts.irrelevant))
            metrics["duplicate_penalty"].append(torch.tensor(reward_parts.duplicate))
            metrics["intent_reward"].append(torch.tensor(reward_parts.intent))
            metrics["group_reward"].append(torch.tensor(reward_parts.group))
            metrics["defense_reward"].append(torch.tensor(0.0))
            metrics["node_accuracy"].append(torch.tensor(float(selected_node == expected_node)))
        if example.defense_scorer is not None and selected_nodes:
            defense_reward = float(example.defense_scorer(example.candidate_nodes[sorted(selected_nodes)]))
            terminal = torch.tensor(defense_reward, dtype=example.candidate_nodes.dtype,
                                    device=example.candidate_nodes.device)
            # Attach the terminal outcome to the final action so the number of
            # rewards still matches the number of sampled policy actions.
            rewards[-1] = rewards[-1] + terminal
            metrics["reward"][-1] = metrics["reward"][-1] + terminal
            metrics["defense_reward"].append(torch.tensor(defense_reward))
        if training and log_probs:
            returns = _discounted_returns(rewards, gamma)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            loss = -(torch.stack(log_probs) * returns.detach()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics["loss"].append(loss.detach())
        if wandb_run is not None and rewards:
            wandb_run.log({
                "rl/episode_reward": float(torch.stack(rewards).sum()),
                "rl/episode_node_accuracy": float(torch.stack(metrics["node_accuracy"][-len(rewards):]).mean()),
                "rl/episode": step_offset + example_index,
            }, step=step_offset + example_index)
    if not metrics["reward"]:
        raise ValueError("examples produced no retrieval steps")
    result = {}
    for key, values in metrics.items():
        result[key] = float(torch.stack(values).mean()) if values else 0.0
    if wandb_run is not None:
        wandb_run.log({f"rl/{key}": value for key, value in result.items()}, step=step_offset + len(examples))
    return result

