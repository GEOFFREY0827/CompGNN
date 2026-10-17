"""Small order-invariant GNN + policy-gradient prototype for DecomposedHarm.

The module intentionally uses only PyTorch and NumPy.  It treats each QA
example as an unordered growing set: a permutation changes the arrival path,
but not the graph representation of the currently observed subtasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def load_split(root: str | Path, split: str):
    root = Path(root) / split
    arrays = np.load(root / "embeddings.npz")
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    labels = torch.tensor(
        [int(record["data"].get("label") == "harmful") for record in records],
        dtype=torch.float32,
    )
    embeddings = torch.from_numpy(arrays["subtask_embeddings"].astype(np.float32))
    offsets = arrays["subtask_offsets"].astype(np.int64)
    examples = [embeddings[offsets[i] : offsets[i + 1]] for i in range(len(records))]
    return examples, labels


class RelationalSetGNN(nn.Module):
    """GNN whose graph representation is invariant to node permutation."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, relations: int = 3):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        # cosine, mean absolute difference, mean product
        self.edge = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, relations))
        self.updates = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(relations)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 2 or nodes.shape[0] == 0:
            raise ValueError("nodes must have shape [num_subtasks, embedding_dim]")
        h = self.node(nodes)
        if len(nodes) == 1:
            return self.norm(h).mean(dim=0)

        x = nodes / nodes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosine = x @ x.T
        diff = (nodes[:, None, :] - nodes[None, :, :]).abs().mean(dim=-1)
        product = (nodes[:, None, :] * nodes[None, :, :]).mean(dim=-1)
        edge_features = torch.stack((cosine, diff, product), dim=-1)
        diagonal = torch.eye(len(nodes), dtype=torch.bool, device=nodes.device)
        edge_features = edge_features.masked_fill(diagonal.unsqueeze(-1), 0)
        relation_weights = torch.sigmoid(self.edge(edge_features))
        relation_weights = relation_weights.masked_fill(diagonal.unsqueeze(-1), 0)

        messages = []
        for relation, update in enumerate(self.updates):
            weights = relation_weights[:, :, relation]
            degree = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            aggregated = weights @ h / degree
            messages.append(update(aggregated))
        h = self.norm(h + torch.stack(messages).sum(dim=0))
        return h.mean(dim=0)


class GNNPolicy(nn.Module):
    """Risk head plus 3-action policy: allow, flag, block."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.encoder = RelationalSetGNN(input_dim, hidden_dim)
        self.risk = nn.Linear(hidden_dim, 1)
        self.action = nn.Linear(hidden_dim, 3)

    def forward(self, nodes: torch.Tensor):
        representation = self.encoder(nodes)
        return self.risk(representation).squeeze(), self.action(representation)


def reward_for(action: torch.Tensor, harmful: torch.Tensor) -> torch.Tensor:
    """Conservative first-pass reward; action ids are allow/flag/block."""
    rewards = torch.zeros_like(harmful)
    benign = harmful < 0.5
    rewards = torch.where(benign & (action == 0), torch.ones_like(rewards), rewards)
    rewards = torch.where(benign & (action == 1), -0.5 * torch.ones_like(rewards), rewards)
    rewards = torch.where(benign & (action == 2), -torch.ones_like(rewards), rewards)
    rewards = torch.where((~benign) & (action == 0), -torch.ones_like(rewards), rewards)
    rewards = torch.where((~benign) & (action == 1), 0.5 * torch.ones_like(rewards), rewards)
    rewards = torch.where((~benign) & (action == 2), torch.ones_like(rewards), rewards)
    return rewards


def run_epoch(model, examples, labels, optimizer=None, seed=0, device=None):
    generator = np.random.default_rng(seed)
    training = optimizer is not None
    model.train(training)
    device = device or next(model.parameters()).device
    risk_values, policy_values, reward_values, correct = [], [], [], []
    harmful_seen = harmful_blocked = benign_seen = benign_blocked = 0
    order = generator.permutation(len(examples))
    for index in order:
        nodes = examples[index].to(device)
        # Random arrival order tests the intended order robustness.
        permutation = generator.permutation(len(nodes))
        nodes = nodes[torch.from_numpy(permutation)]
        harmful = labels[index].to(device)
        current_risk_losses = []
        log_probs, episode_rewards = [], []
        optimizer.zero_grad() if training else None
        for end in range(1, len(nodes) + 1):
            risk_logit, action_logits = model(nodes[:end])
            risk_loss = nn.functional.binary_cross_entropy_with_logits(risk_logit, harmful)
            current_risk_losses.append(risk_loss)
            risk_values.append(risk_loss.detach())
            distribution = Categorical(logits=action_logits)
            action = distribution.sample() if training else action_logits.argmax()
            log_probs.append(distribution.log_prob(action))
            reward = reward_for(action, harmful)
            episode_rewards.append(reward)
            correct.append(float((action == 2) == (harmful > 0.5)))
            if harmful > 0.5:
                harmful_seen += 1
                harmful_blocked += int(action == 2)
            else:
                benign_seen += 1
                benign_blocked += int(action == 2)
        if training:
            returns = torch.stack(episode_rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            policy_loss = -(torch.stack(log_probs) * returns.detach()).mean()
            loss = torch.stack(current_risk_losses).mean() + 0.2 * policy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            policy_values.append(policy_loss.detach())
        reward_values.extend(episode_rewards)
    return {
        "risk_loss": float(torch.stack(risk_values).mean()),
        "policy_loss": float(torch.stack(policy_values).mean()) if policy_values else 0.0,
        "mean_reward": float(torch.stack(reward_values).mean()),
        "action_accuracy": float(np.mean(correct)),
        "harmful_detection_rate": harmful_blocked / max(harmful_seen, 1),
        "benign_false_positive_rate": benign_blocked / max(benign_seen, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/qa_embeddings"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true", help="Log metrics and gradients to Weights & Biases")
    parser.add_argument("--wandb-project", default="compgnn-decompharm")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/gnn_rl_checkpoints"),
        help="Directory for per-epoch model checkpoints",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_x, train_y = load_split(args.data, "train")
    valid_x, valid_y = load_split(args.data, "validation")
    test_x, test_y = load_split(args.data, "test")
    model = GNNPolicy(train_x[0].shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("W&B logging requires `pip install 'compgnn[tracking]'`") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "data": str(args.data), "epochs": args.epochs,
                "hidden_dim": args.hidden_dim, "seed": args.seed,
                "device": str(device), "model": "RelationalSetGNN+REINFORCE",
                "train_examples": len(train_x), "validation_examples": len(valid_x),
                "test_examples": len(test_x),
            },
            tags=["decompHarm", "qa", "gnn", "rl", "permutation-invariant"],
        )
        wandb_run.watch(model, log="gradients", log_freq=100)
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_x, train_y, optimizer, args.seed + epoch, device)
        with torch.no_grad():
            valid_stats = run_epoch(model, valid_x, valid_y, None, args.seed + epoch, device)
        print(json.dumps({"epoch": epoch, "train": train_stats, "validation": valid_stats}))
        checkpoint_path = args.checkpoint_dir / f"epoch-{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train": train_stats,
                "validation": valid_stats,
                "config": vars(args),
            },
            checkpoint_path,
        )
        if wandb_run is not None:
            wandb_run.log(
                {**{f"train/{key}": value for key, value in train_stats.items()},
                 **{f"validation/{key}": value for key, value in valid_stats.items()},
                 "epoch": epoch,
                 "checkpoint/epoch": epoch},
                step=epoch,
            )
            wandb_run.save(str(checkpoint_path), policy="now")
            checkpoint_artifact = wandb.Artifact(
                f"gnn-rl-checkpoint-epoch-{epoch:03d}",
                type="model",
                metadata={"epoch": epoch, "validation": valid_stats},
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            wandb_run.log_artifact(checkpoint_artifact)
    with torch.no_grad():
        test_stats = run_epoch(model, test_x, test_y, None, args.seed + 10000, device)
    print(json.dumps({"final_test": test_stats}))
    if wandb_run is not None:
        wandb_run.log({f"test/{key}": value for key, value in test_stats.items()}, step=args.epochs + 1)
        wandb_run.finish()


if __name__ == "__main__":
    main()
