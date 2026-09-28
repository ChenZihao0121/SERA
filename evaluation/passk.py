"""Aggregate generated, graded responses into per-dataset Pass@K."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .protocol import (atomic_json, digest, file_digest, output_lock, read_records,
                       response_files, validate_record, write_records)


def pass_at_k(n, c, k):
    """Unbiased finite-pool estimator: 1 - C(n-c, k) / C(n, k)."""
    if not all(type(x) is int for x in (n, c, k)) or not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("Require integer 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if k == 1:
        return c / n
    if n - c < k:
        return 1.0
    a, b = (c, k) if c < k else (k, c)
    return -math.expm1(math.fsum(math.log1p(-b / i) for i in range(n - a + 1, n + 1)))


def default_ks(n):
    return sorted({2**i for i in range(n.bit_length())} | {n})


def maze_pass_at_k(correct, k, repeats=1000, seed=42):
    """Legacy trainer bootstrap: reset MT19937 for every question and K.

    Preserve sample order. The finite bootstrap is deliberately not replaced
    by either the combination estimator or its analytic with-replacement mean.
    """
    import numpy as np
    if not correct or any(type(value) is not bool for value in correct):
        raise ValueError("Maze correctness must be a nonempty ordered boolean pool")
    if type(k) is not int or not 1 <= k <= len(correct) or type(repeats) is not int or repeats < 1:
        raise ValueError("Invalid Maze bootstrap K or repetition count")
    if k == 1:
        return sum(correct) / len(correct)
    rng = np.random.RandomState(seed)
    values = np.asarray(correct, dtype=np.bool_)
    indices = rng.choice(len(values), size=(repeats, k), replace=True)
    return float(values[indices].any(axis=1).mean())


def summarize(output_dir, ks=None, already_locked=False):
    if already_locked:
        return _summarize(output_dir, ks)
    with output_lock(output_dir):
        return _summarize(output_dir, ks)


def _summarize(output_dir, ks=None):
    output_dir = Path(output_dir)
    config = json.loads((output_dir / "run.json").read_text())
    n = config["samples"]
    if type(n) is not int or n < 1:
        raise ValueError("Response pool size must be a positive integer")
    if config.get("identity") and digest({key: value for key, value in config.items() if key != "identity"}) != config["identity"]:
        raise ValueError("Saved protocol identity is invalid")
    ks = sorted(set(ks or default_ks(n)))
    for k in ks:
        pass_at_k(n, 0, k)
    prompts = list(read_records(output_dir / "prompts.jsonl"))
    if config.get("prompts_digest") and digest(prompts) != config["prompts_digest"]:
        raise ValueError("Saved prompts differ from the frozen protocol")
    prompt_map = {p["prompt_id"]: p for p in prompts}
    counts = {p["prompt_id"]: dict(n=0, c=0, capped=0, grading_errors=0, context_limited=0,
                                  seen=bytearray(n), correct=[False] * n) for p in prompts}
    if not prompts or len(counts) != len(prompts):
        raise ValueError("Prompts must be nonempty and have unique IDs")
    for path in response_files(output_dir):
        if config.get("schema_version", 0) >= 2:
            receipt_path = output_dir / "receipts" / (path.name.removesuffix(".jsonl.gz") + ".json")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("identity") != config["identity"] or receipt.get("sha256") != file_digest(path):
                raise ValueError("Committed response checksum or protocol identity mismatch")
        for record in read_records(path):
            if record.get("prompt_id") not in counts:
                raise ValueError("Response has an unknown prompt ID")
            validate_record(record, prompt_map[record["prompt_id"]], config)
            group = counts[record["prompt_id"]]
            index = record["sample_index"]
            if group["seen"][index]:
                raise ValueError("Duplicate response index")
            group["seen"][index] = 1
            group["correct"][index] = record["correct"]
            group["n"] += 1
            group["c"] += int(record["correct"])
            group["capped"] += int(record["hit_response_cap"])
            group["context_limited"] += int(record.get("hit_context_limit", False))
            group["grading_errors"] += int(record["grading_status"] in ("timeout", "verifier_error"))
    datasets = defaultdict(list)
    per_prompt = []
    for prompt in prompts:
        group = counts[prompt["prompt_id"]]
        if group["n"] != n:
            raise ValueError(f"Incomplete response pool for {prompt['prompt_id']}: {group['n']}/{n}")
        group.pop("seen")
        correct = group.pop("correct")
        row = dict(prompt_id=prompt["prompt_id"], dataset=prompt["dataset"], **group)
        if config["task"] == "maze":
            bootstrap = config.get("bootstrap", dict(seed=42, repeats=1000))
            row.update({f"pass@{k}": maze_pass_at_k(correct, k, **bootstrap) for k in ks})
        else:
            row.update({f"pass@{k}": pass_at_k(n, group["c"], k) for k in ks})
        per_prompt.append(row)
        datasets[row["dataset"]].append(row)
    result = {}
    for dataset, rows in datasets.items():
        result[dataset] = dict(prompts=len(rows), samples_per_prompt=n,
                              capped_responses=sum(r["capped"] for r in rows),
                              context_limited_responses=sum(r["context_limited"] for r in rows),
                              grading_errors=sum(r["grading_errors"] for r in rows),
                              estimator="maze_bootstrap" if config["task"] == "maze" else "combination")
        result[dataset].update({f"pass@{k}": math.fsum(r[f"pass@{k}"] for r in rows) / len(rows) for k in ks})
    write_records(output_dir / "per_prompt.jsonl", per_prompt)
    atomic_json(output_dir / "passk.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory written by evaluation.sequence")
    parser.add_argument("--ks", nargs="+", type=int)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output_dir, args.ks), indent=2))
