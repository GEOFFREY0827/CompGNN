"""Encode QA tasks without mixing splits or encoding labels as features."""
import argparse
import ast
import json
from pathlib import Path


def parse_subtasks(value):
    """Accept HF list columns and CSV JSON/Python list representations."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Subtasks must be a JSON/Python list of strings") from exc
    if not isinstance(value, list) or not all(
        isinstance(text, str) and text.strip() for text in value
    ):
        raise ValueError("Subtasks must be a list of nonempty strings")
    return value


def resolve_column(columns, explicit, candidates, role):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Missing {role} column {explicit!r}; available: {columns}")
        return explicit
    matches = [name for name in candidates if name in columns]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely resolve {role} column; available: {columns}. "
            f"Set --{role}-column explicitly."
        )
    return matches[0]


def prepare_records(dataset, task_column, original_column, subtasks_column, limit=None):
    records, originals, subtasks, offsets = [], [], [], [0]
    for index, row in enumerate(dataset):
        if row[task_column] != "QA":
            continue
        original = row[original_column]
        if not isinstance(original, str) or not original.strip():
            raise ValueError(f"Row {index}: original task must be a nonempty string")
        try:
            parts = parse_subtasks(row[subtasks_column])
        except ValueError as exc:
            raise ValueError(f"Row {index}: {exc}") from exc
        records.append({"source_row": index, "data": dict(row)})
        originals.append(original)
        subtasks.extend(parts)
        offsets.append(len(subtasks))
        if limit is not None and len(records) >= limit:
            break
    return records, originals, subtasks, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="YuehHanChen/DecomposedHarm")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qa_embeddings"))
    parser.add_argument("--splits", nargs="+")
    # Support both the original dataset schema and the current schema.  An
    # explicit CLI value still takes precedence and is validated below.
    parser.add_argument("--task-column")
    parser.add_argument("--original-column")
    parser.add_argument("--subtasks-column")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Maximum QA examples per split for smoke runs")
    parser.add_argument("--device", help="cpu, cuda, mps; default is automatic")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    # Imports are lazy so --help and schema tests work without ML dependencies.
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    dataset = load_dataset(args.dataset, revision=args.dataset_revision)
    splits = args.splits or list(dataset)
    if set(splits) - set(dataset):
        parser.error(f"Unknown splits; available: {list(dataset)}")
    prepared = {}
    resolved_columns = {}
    for split in splits:
        columns = dataset[split].column_names
        task = resolve_column(columns, args.task_column,
                              ["Task", "task_type", "task"], "task")
        original = resolve_column(columns, args.original_column,
                                  ["original_task", "original_prompt", "Original Task"], "original")
        subtasks = resolve_column(columns, args.subtasks_column,
                                  ["decomposed_subtasks", "Decomposed Subtasks",
                                   "subtasks", "decomposition"], "subtasks")
        prepared[split] = prepare_records(dataset[split], task, original, subtasks, args.limit)
        resolved_columns[split] = {"task": task, "original": original, "subtasks": subtasks}
    if not any(item[0] for item in prepared.values()):
        raise ValueError("No rows with the task value exactly equal to 'QA' in selected splits")
    # Refuse overwrites before downloading the encoder or starting expensive inference.
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {args.output}")
    model = SentenceTransformer(args.model, revision=args.model_revision, device=args.device)
    dimension = model.get_sentence_embedding_dimension()
    def encode(texts):
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)
        return model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=not args.no_normalize
                            ).astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, (records, originals, subtasks, offsets) in prepared.items():
        # Use safe directory names, including for custom dataset split names.
        if Path(split).name != split or split in (".", ".."):
            raise ValueError(f"Unsafe split name: {split!r}")
        target = args.output / split
        target.mkdir()
        np.savez_compressed(target / "embeddings.npz",
                            original_embeddings=encode(originals),
                            subtask_embeddings=encode(subtasks),
                            subtask_offsets=np.asarray(offsets, dtype=np.int64))
        with (target / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = {"examples": len(records), "subtasks": len(subtasks)}
        print(f"{split}: {len(records)} QA examples, {len(subtasks)} subtasks -> {target}")
    manifest = {
        "dataset": args.dataset, "dataset_revision": args.dataset_revision,
        "dataset_fingerprints": {s: dataset[s]._fingerprint for s in splits},
        "model": args.model, "model_revision": args.model_revision,
        "dimension": dimension, "normalized": not args.no_normalize,
        "max_seq_length": model.max_seq_length,
        "filter": {"column": next(iter(resolved_columns.values()))["task"], "value": "QA"},
        "columns": resolved_columns,
        "splits": counts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
