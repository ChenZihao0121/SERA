# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
# See LICENSE or https://www.apache.org/licenses/LICENSE-2.0
"""Prepare the pinned POLARIS training split using the original MaxRL fields."""

import argparse
from pathlib import Path

REVISION = "296f8e34132e63f4a1d70e0dcc8bddebb43f03e4"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def process_row(example, index):
    return dict(data_source="polaris", id=index,
                prompt=[dict(role="user", content=example["problem"] + INSTRUCTION)],
                ability="math", reward_model=dict(style="rule", ground_truth=str(example["answer"])),
                extra_info=dict(split="train", index=index))


def main():
    import datasets
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/polaris")
    args = parser.parse_args()
    output = Path(args.output)
    path = output / "train.parquet"
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    raw = datasets.load_dataset("POLARIS-Project/Polaris-Dataset-53K", "default", revision=REVISION)["train"]
    # Original map retains source columns; preserve that behavior and row order.
    prepared = raw.map(process_row, with_indices=True)
    output.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(path)
    from evaluation.prepare_data import file_sha
    import json
    (output / "manifest.json").write_text(json.dumps(dict(
        repo="POLARIS-Project/Polaris-Dataset-53K", revision=REVISION,
        rows=len(prepared), sha256=file_sha(path), online_validation="AIME 2025 and MATH-500"), indent=2) + "\n")


if __name__ == "__main__":
    main()
