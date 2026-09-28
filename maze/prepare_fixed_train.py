#!/usr/bin/env python3
"""Sample one reusable Maze training set, retaining the original validation set."""

import argparse
import json
from pathlib import Path
import random
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def prompt_text(prompt):
    # Same raw-text construction as RLHFDataset with apply_chat_template=False.
    return "".join(message.get("content", "") for message in prompt)


def read_selected_rows(parquet, indices):
    """Read selected rows in source order without loading the full dataset."""
    parts = []
    offset = cursor = 0
    for group in range(parquet.num_row_groups):
        end = offset + parquet.metadata.row_group(group).num_rows
        start = cursor
        while cursor < len(indices) and indices[cursor] < end:
            cursor += 1
        if cursor > start:
            table = parquet.read_row_group(group)
            parts.append(table.take(pa.array([i - offset for i in indices[start:cursor]])))
        offset = end
    return pa.concat_tables(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=SCRIPT_DIR / "data/train.parquet")
    parser.add_argument("--validation", type=Path, default=SCRIPT_DIR / "data/test.parquet")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-train", type=int, default=7424)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=320)
    parser.add_argument("--tokenizer-json", type=Path, default=SCRIPT_DIR / "ckpt-1500/tokenizer.json")
    args = parser.parse_args()
    train = args.train.resolve()
    validation = args.validation.resolve()
    output = (args.output_dir or train.parent / f"fixed_train{args.num_train}_seed{args.seed}").resolve()
    if output.exists():
        parser.error(f"Output already exists; reuse its train.parquet and manifest.json: {output}")
    if args.num_train <= 0 or args.batch_size <= 0 or args.max_prompt_length <= 0:
        parser.error("Training size, batch size, and prompt limit must be positive")
    if args.num_train % args.batch_size:
        parser.error("--num-train must be divisible by --batch-size to avoid drop_last cold starts")
    source = pq.ParquetFile(train)
    source_rows = source.metadata.num_rows
    if args.num_train > source_rows:
        parser.error(f"Requested {args.num_train} rows from only {source_rows}")
    indices = sorted(random.Random(args.seed).sample(range(source_rows), args.num_train))
    selected = read_selected_rows(source, indices)
    texts = [prompt_text(prompt) for prompt in selected["prompt"].to_pylist()]
    if len(set(texts)) != len(texts):
        parser.error("Sample contains duplicate prompts; choose another seed or deduplicate the source")
    val_prompts = pq.read_table(validation, columns=["prompt"])
    val_texts = {prompt_text(prompt) for prompt in val_prompts["prompt"].to_pylist()}
    if set(texts) & val_texts:
        parser.error("Selected training prompts overlap validation; fix the source split before training")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths = [len(item.ids) for item in tokenizer.encode_batch(texts, add_special_tokens=False)]
    if max(lengths) > args.max_prompt_length:
        parser.error("Selected prompts exceed the length limit and would be dropped by RLHFDataset")

    # Key estimates by original source row, even when old exports lack an index
    # or assign the same index to every prompt. Preserve any old ID for auditing.
    old_info = selected["extra_info"].to_pylist() if "extra_info" in selected.column_names else [{}] * len(indices)
    extra_info = []
    for row_index, previous in zip(indices, old_info):
        info = dict(previous or {})
        if "index" in info:
            info["original_index"] = info["index"]
        info["index"] = row_index
        extra_info.append(info)
    info_column = pa.array(extra_info)
    if "extra_info" in selected.column_names:
        selected = selected.set_column(selected.column_names.index("extra_info"), "extra_info", info_column)
    else:
        selected = selected.append_column("extra_info", info_column)

    manifest = {
        "source_train": {"path": str(train), "rows": source_rows},
        "validation": {"path": str(validation), "rows": len(val_prompts)},
        "selection": "random.Random(seed).sample without replacement; sorted by source row",
        "seed": args.seed,
        "num_train": len(selected),
        "batch_size": args.batch_size,
        "steps_per_epoch": len(selected) // args.batch_size,
        "source_row_indices": indices,
        "prompt_tokens": {"min": min(lengths), "max": max(lengths), "limit": args.max_prompt_length},
        "train_validation_prompt_overlap": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".maze-subset-", dir=output.parent) as temporary:
        staging = Path(temporary)
        pq.write_table(selected, staging / "train.parquet")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    print(f"Train: {output / 'train.parquet'} ({len(selected)} prompts)")
    print(f"Validation: {validation} ({len(val_prompts)} prompts; unchanged)")
    print(f"Epoch: {manifest['steps_per_epoch']} steps at B={args.batch_size}")
    print(f"Manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
