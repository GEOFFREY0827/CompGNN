"""Retrieve a source decomposition group from a noisy 25-node memory.

This is a controlled retrieval experiment, not a claim that the current
dataset contains clean hidden attack groups. Each source example contributes
the positive node group; unrelated subtasks are sampled as distractors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...core.intent_reconstruction import load_split


def make_memory(examples, targets, index, memory_size, rng):
    positives = examples[index]
    candidates = [i for i in range(len(examples)) if i != index]
    distractor_nodes = []
    while len(distractor_nodes) < memory_size - len(positives):
        source = candidates[int(rng.integers(len(candidates)))]
        node = examples[source][int(rng.integers(len(examples[source])))]
        distractor_nodes.append(node)
    nodes = torch.cat([positives, torch.stack(distractor_nodes)], dim=0)
    positive = torch.zeros(memory_size, dtype=torch.float32)
    positive[: len(positives)] = 1.0
    permutation = torch.from_numpy(rng.permutation(memory_size))
    nodes = nodes[permutation]
    positive = positive[permutation]
    return nodes, positive, targets[index]


class EdgePriorRetriever(nn.Module):
    def __init__(self, input_dim, target_dim, relation_slots=8):
        super().__init__()
        self.key = nn.Linear(input_dim, target_dim, bias=False)
        self.value = nn.Linear(input_dim, target_dim, bias=False)
        self.context = nn.Linear(input_dim, target_dim, bias=False)
        self.relation_prior = nn.Parameter(torch.randn(relation_slots, target_dim) * 0.02)
        self.gate = nn.Linear(target_dim, 1)
        self.output = nn.Linear(target_dim, target_dim)

    def forward(self, nodes):
        context = self.context(nodes.mean(dim=0))
        queries = self.relation_prior + context.unsqueeze(0)
        keys = self.key(nodes)
        values = self.value(nodes)
        logits = queries @ keys.T / (keys.shape[-1] ** 0.5)
        attention = logits.softmax(dim=-1)
        retrieved = attention @ values
        gates = self.gate(queries).squeeze(-1).softmax(dim=0)
        representation = F.normalize(self.output((gates[:, None] * retrieved).sum(dim=0)), dim=-1)
        node_scores = logits.max(dim=0).values
        return representation, node_scores, attention


def baseline_scores(nodes):
    query = F.normalize(nodes.mean(dim=0), dim=-1)
    return F.normalize(nodes, dim=-1) @ query


def run_epoch(model, examples, targets, memory_size, optimizer, seed, device, train):
    rng = np.random.default_rng(seed)
    model.train(train)
    total = {"loss": [], "precision": [], "recall": [], "all_recall": [], "intent_cosine": []}
    order = rng.permutation(len(examples))
    for index in order:
        nodes, positive, target = make_memory(examples, targets, index, memory_size, rng)
        nodes, positive, target = nodes.to(device), positive.to(device), F.normalize(target.to(device), dim=-1)
        if train:
            optimizer.zero_grad()
        representation, scores, _ = model(nodes)
        k = int(positive.sum().item())
        top = scores.topk(k).indices
        selected = torch.zeros_like(positive)
        selected[top] = 1.0
        bce = F.binary_cross_entropy_with_logits(scores, positive, pos_weight=torch.tensor(
            (len(positive) - k) / max(k, 1), device=device
        ))
        cosine_loss = 1.0 - F.cosine_similarity(representation, target, dim=0)
        loss = bce + 0.2 * cosine_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tp = (selected * positive).sum()
        total["loss"].append(loss.detach())
        total["precision"].append(tp / selected.sum().clamp_min(1.0))
        total["recall"].append(tp / positive.sum().clamp_min(1.0))
        total["all_recall"].append((tp == positive.sum()).float())
        total["intent_cosine"].append(1.0 - cosine_loss.detach())
    return {key: float(torch.stack(value).mean()) for key, value in total.items()}


def evaluate_baseline(examples, targets, memory_size, seed):
    rng = np.random.default_rng(seed)
    values = {"precision": [], "recall": [], "all_recall": []}
    for index in range(len(examples)):
        nodes, positive, _ = make_memory(examples, targets, index, memory_size, rng)
        k = int(positive.sum().item())
        selected = torch.zeros_like(positive)
        selected[baseline_scores(nodes).topk(k).indices] = 1.0
        tp = (selected * positive).sum()
        values["precision"].append(tp / selected.sum().clamp_min(1.0))
        values["recall"].append(tp / positive.sum().clamp_min(1.0))
        values["all_recall"].append((tp == positive.sum()).float())
    return {key: float(torch.stack(value).mean()) for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/qa_embeddings"))
    parser.add_argument("--memory-size", type=int, default=25)
    parser.add_argument("--relations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="compgnn-decompharm")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_x, train_t, train_y = load_split(args.data, "train")
    valid_x, valid_t, valid_y = load_split(args.data, "validation")
    test_x, test_t, test_y = load_split(args.data, "test")
    model = EdgePriorRetriever(train_x[0].shape[-1], train_t.shape[-1], args.relations).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               config={**vars(args), "model": "edge-prior-memory-retriever"},
                               tags=["memory-retrieval", "graph-prior", "distractor-robustness"])
    print(json.dumps({"mean_baseline_test": evaluate_baseline(test_x, test_t, args.memory_size, args.seed + 10000)}))
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_x, train_t, args.memory_size, optimizer, args.seed + epoch, device, True)
        with torch.no_grad():
            valid_stats = run_epoch(model, valid_x, valid_t, args.memory_size, None, args.seed + epoch, device, False)
        print(json.dumps({"epoch": epoch, "train": train_stats, "validation": valid_stats}))
        if wandb_run is not None:
            wandb_run.log({**{f"train/{k}": v for k, v in train_stats.items()},
                           **{f"validation/{k}": v for k, v in valid_stats.items()}, "epoch": epoch}, step=epoch)
    with torch.no_grad():
        test_stats = run_epoch(model, test_x, test_t, args.memory_size, None, args.seed + 10000, device, False)
    print(json.dumps({"final_test": test_stats}))
    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in test_stats.items()}, step=args.epochs + 1)
        wandb_run.finish()


if __name__ == "__main__":
    main()
