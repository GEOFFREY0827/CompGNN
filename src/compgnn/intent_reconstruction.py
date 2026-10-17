"""Reconstruct the original-task embedding from an unordered subtask set.

The original-task embeddings are frozen teacher targets produced by the
Sentence Transformer encoder.  The GNN only receives subtask embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gnn_rl import RelationalSetGNN


def load_split(root: str | Path, split: str):
    path = Path(root) / split
    arrays = np.load(path / "embeddings.npz")
    subtasks = torch.from_numpy(arrays["subtask_embeddings"].astype(np.float32))
    targets = torch.from_numpy(arrays["original_embeddings"].astype(np.float32))
    offsets = arrays["subtask_offsets"].astype(np.int64)
    examples = [subtasks[offsets[i] : offsets[i + 1]] for i in range(len(targets))]
    records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
    labels = torch.tensor(
        [int(record["data"].get("label") == "harmful") for record in records],
        dtype=torch.long,
    )
    return examples, targets, labels


class IntentReconstructor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, target_dim: int):
        super().__init__()
        self.encoder = RelationalSetGNN(input_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.encoder(nodes)), dim=-1)


def mean_baseline(nodes: torch.Tensor) -> torch.Tensor:
    return F.normalize(nodes.mean(dim=0), dim=-1)


def run_epoch(model, examples, targets, labels, optimizer=None, seed=0, device=None):
    generator = np.random.default_rng(seed)
    training = optimizer is not None
    model.train(training)
    device = device or next(model.parameters()).device
    predictions, target_values, label_values = [], [], []
    order = generator.permutation(len(examples))
    for index in order:
        nodes = examples[index].to(device)
        nodes = nodes[torch.from_numpy(generator.permutation(len(nodes)))]
        target = F.normalize(targets[index].to(device), dim=-1)
        if training:
            optimizer.zero_grad()
        prediction = model(nodes)
        loss = 1.0 - F.cosine_similarity(prediction, target, dim=0)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        predictions.append(prediction.detach())
        target_values.append(target.detach())
        label_values.append(labels[index])

    predicted = torch.stack(predictions)
    targets_tensor = torch.stack(target_values)
    cosine = (predicted * targets_tensor).sum(dim=-1)
    similarities = predicted @ targets_tensor.T
    retrieval = similarities.argmax(dim=1)
    labels_tensor = torch.stack(label_values).to(device)
    return {
        "cosine": float(cosine.mean()),
        "retrieval_top1": float((retrieval == torch.arange(len(retrieval), device=device)).float().mean()),
        "harmful_cosine": float(cosine[labels_tensor == 1].mean()),
        "benign_cosine": float(cosine[labels_tensor == 0].mean()),
    }


def evaluate_mean_baseline(examples, targets, labels):
    predictions = torch.stack([mean_baseline(nodes) for nodes in examples])
    targets = F.normalize(targets, dim=-1)
    cosine = (predictions * targets).sum(dim=-1)
    retrieval = (predictions @ targets.T).argmax(dim=1)
    return {
        "cosine": float(cosine.mean()),
        "retrieval_top1": float((retrieval == torch.arange(len(retrieval))).float().mean()),
        "harmful_cosine": float(cosine[labels == 1].mean()),
        "benign_cosine": float(cosine[labels == 0].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/qa_embeddings"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/intent_checkpoints"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="compgnn-decompharm")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_x, train_t, train_y = load_split(args.data, "train")
    valid_x, valid_t, valid_y = load_split(args.data, "validation")
    test_x, test_t, test_y = load_split(args.data, "test")
    model = IntentReconstructor(train_x[0].shape[-1], args.hidden_dim, train_t.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "model": "RelationalSetGNN-intent-reconstruction"},
            tags=["decompHarm", "qa", "gnn", "intent-reconstruction", "leakage-audit"],
        )

    baseline = evaluate_mean_baseline(test_x, test_t, test_y)
    print(json.dumps({"mean_baseline_test": baseline}))
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
            artifact = wandb.Artifact(f"intent-reconstruction-epoch-{epoch:03d}", type="model")
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
