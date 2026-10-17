"""Generate auditable weak relation labels from decomposition records.

The current DecomposedHarm export has no gold edge annotations.  This module
therefore creates weak labels with explicit provenance:

* ``support``: consecutive decomposition steps, preserving their order;
* ``combine``: semantically similar steps above a configurable threshold.

These labels are useful for a first experiment, but are not substitutes for
human/oracle edge annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def infer_relation_edges(
    embeddings: np.ndarray,
    combine_threshold: float = 0.55,
    add_top_combine: bool = True,
) -> list[dict]:
    """Infer relation-labelled edges for one example's subtask embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must have shape [num_nodes, dimension]")
    if not 0.0 <= combine_threshold <= 1.0:
        raise ValueError("combine_threshold must be in [0, 1]")

    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    similarity = normalized @ normalized.T
    edges: list[dict] = []
    seen: set[tuple[int, int, str]] = set()

    def add(source: int, target: int, relation: str, score: float, rule: str) -> None:
        key = (source, target, relation)
        if key not in seen:
            seen.add(key)
            edges.append({
                "source": source,
                "target": target,
                "relation": relation,
                "score": round(float(score), 6),
                "source_rule": rule,
            })

    for index in range(embeddings.shape[0] - 1):
        add(index, index + 1, "support", 1.0, "decomposition_order")

    for source in range(embeddings.shape[0]):
        for target in range(source + 1, embeddings.shape[0]):
            score = float(similarity[source, target])
            if score >= combine_threshold:
                add(source, target, "combine", score, "embedding_cosine_threshold")

    if add_top_combine and embeddings.shape[0] > 1 and not any(
        edge["relation"] == "combine" for edge in edges
    ):
        upper = np.triu(similarity, k=1)
        source, target = np.unravel_index(np.argmax(upper), upper.shape)
        add(int(source), int(target), "combine", float(upper[source, target]), "top_pair_fallback")
    return edges


def label_split(split_dir: str | Path, combine_threshold: float = 0.55) -> int:
    """Write ``graph_labels.jsonl`` beside one encoded split."""
    split_dir = Path(split_dir)
    arrays = np.load(split_dir / "embeddings.npz")
    offsets = arrays["subtask_offsets"].astype(np.int64)
    records = [json.loads(line) for line in (split_dir / "records.jsonl").read_text().splitlines()]
    if len(records) != len(offsets) - 1:
        raise ValueError("records and subtask_offsets are misaligned")
    output = split_dir / "graph_labels.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            start, end = offsets[index], offsets[index + 1]
            edges = infer_relation_edges(
                arrays["subtask_embeddings"][start:end], combine_threshold
            )
            payload = {
                "id": record["data"].get("id", index),
                "num_nodes": int(end - start),
                "edges": edges,
                "label_status": "weak_heuristic",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/qa_embeddings"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--combine-threshold", type=float, default=0.55)
    args = parser.parse_args()
    for split in args.splits:
        count = label_split(args.data / split, args.combine_threshold)
        print(f"{split}: wrote {count} weakly labelled graphs")


if __name__ == "__main__":
    main()
