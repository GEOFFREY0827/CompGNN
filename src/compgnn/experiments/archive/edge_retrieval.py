"""Edge-prior-conditioned retrieval from a subtask memory.

This is a prototype for testing whether learned relation slots can retrieve
useful nodes before intent reconstruction. The current dataset has no gold
edge labels, so relation slots are latent and trained against frozen
original-task Sentence Transformer embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...core.intent_reconstruction import load_split, mean_baseline


class EdgeGuidedRetriever(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, relation_slots: int = 8, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.node_key = nn.Linear(input_dim, target_dim, bias=False)
        self.node_value = nn.Linear(input_dim, target_dim, bias=False)
        self.query = nn.Linear(input_dim, target_dim, bias=False)
        self.relation_prior = nn.Parameter(torch.randn(relation_slots, target_dim) * 0.02)
        self.relation_gate = nn.Linear(target_dim, 1)
        self.output = nn.Linear(target_dim, target_dim)

    def forward(self, nodes: torch.Tensor, return_attention: bool = False):
        # A global context initializes relation-conditioned retrieval.
        context = nodes.mean(dim=0)
        base_query = self.query(context)
        relation_queries = base_query.unsqueeze(0) + self.relation_prior
        keys = self.node_key(nodes)
        values = self.node_value(nodes)
        scores = relation_queries @ keys.T / (keys.shape[-1] ** 0.5 * self.temperature)
        attention = scores.softmax(dim=-1)
        retrieved = attention @ values
        gates = self.relation_gate(relation_queries).squeeze(-1).softmax(dim=0)
        pooled = (gates.unsqueeze(-1) * retrieved).sum(dim=0)
        output = F.normalize(self.output(pooled), dim=-1)
        if return_attention:
            return output, attention, gates
        return output


def run_epoch(model, examples, targets, labels, optimizer=None, seed=0, device=None):
    generator = np.random.default_rng(seed)
    training = optimizer is not None
    model.train(training)
    device = device or next(model.parameters()).device
    predictions, target_values, label_values, entropies = [], [], [], []
    order = generator.permutation(len(examples))
    for index in order:
        nodes = examples[index].to(device)
        nodes = nodes[torch.from_numpy(generator.permutation(len(nodes)))]
        target = F.normalize(targets[index].to(device), dim=-1)
        if training:
            optimizer.zero_grad()
        prediction, attention, _ = model(nodes, return_attention=True)
        loss = 1.0 - F.cosine_similarity(prediction, target, dim=0)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        predictions.append(prediction.detach())
        target_values.append(target.detach())
        label_values.append(labels[index])
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)
        entropies.append(entropy.mean().detach())

    predicted = torch.stack(predictions)
    target_tensor = torch.stack(target_values)
    cosine = (predicted * target_tensor).sum(dim=-1)
    retrieval = predicted @ target_tensor.T
    top1 = retrieval.argmax(dim=1)
    labels_tensor = torch.stack(label_values).to(device)
    return {
        "cosine": float(cosine.mean()),
        "retrieval_top1": float((top1 == torch.arange(len(top1), device=device)).float().mean()),
        "harmful_cosine": float(cosine[labels_tensor == 1].mean()),
        "benign_cosine": float(cosine[labels_tensor == 0].mean()),
        "attention_entropy": float(torch.stack(entropies).mean()),
    }


def evaluate_mean(examples, targets, labels):
    predictions = torch.stack([mean_baseline(nodes) for nodes in examples])
    targets = F.normalize(targets, dim=-1)
    cosine = (predictions * targets).sum(dim=-1)
    top1 = (predictions @ targets.T).argmax(dim=1)
    return {
        "cosine": float(cosine.mean()),
        "retrieval_top1": float((top1 == torch.arange(len(top1))).float().mean()),
        "harmful_cosine": float(cosine[labels == 1].mean()),
        "benign_cosine": float(cosine[labels == 0].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/qa_embeddings"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--relations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/edge_retrieval_checkpoints"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="compgnn-decompharm")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_x, train_t, train_y = load_split(args.data, "train")
    valid_x, valid_t, valid_y = load_split(args.data, "validation")
    test_x, test_t, test_y = load_split(args.data, "test")
    model = EdgeGuidedRetriever(train_x[0].shape[-1], train_t.shape[-1], args.relations).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "model": "edge-conditioned-retrieval"},
            tags=["decompHarm", "qa", "retrieval", "graph-prior", "intent-reconstruction"],
        )

    print(json.dumps({"mean_baseline_test": evaluate_mean(test_x, test_t, test_y)}))
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_x, train_t, train_y, optimizer, args.seed + epoch, device)
        with torch.no_grad():
            valid_stats = run_epoch(model, valid_x, valid_t, valid_y, None, args.seed + epoch, device)
        checkpoint_path = args.checkpoint_dir / f"epoch-{epoch:03d}.pt"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train": train_stats, "validation": valid_stats, "config": vars(args)}, checkpoint_path)
        print(json.dumps({"epoch": epoch, "train": train_stats, "validation": valid_stats}))
        if wandb_run is not None:
            wandb_run.log({**{f"train/{k}": v for k, v in train_stats.items()},
                           **{f"validation/{k}": v for k, v in valid_stats.items()}, "epoch": epoch}, step=epoch)
            artifact = wandb.Artifact(f"edge-retrieval-checkpoint-epoch-{epoch:03d}", type="model")
            artifact.add_file(str(checkpoint_path))
            wandb_run.log_artifact(artifact)
    with torch.no_grad():
        test_stats = run_epoch(model, test_x, test_t, test_y, None, args.seed + 10000, device)
    print(json.dumps({"final_test": test_stats}))
    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in test_stats.items()}, step=args.epochs + 1)
        wandb_run.finish()


if __name__ == "__main__":
    main()
