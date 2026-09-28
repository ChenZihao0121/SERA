"""Dependency-light identities and response contracts for formal evaluation."""

import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

RAW_FIELDS = ("identity", "prompt_id", "dataset", "sample_index", "seed", "ground_truth", "response",
              "response_token_ids", "response_length", "finish_reason", "stop_reason", "hit_response_cap",
              "hit_context_limit")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def raw_digest(record):
    return digest({key: record[key] for key in RAW_FIELDS if key in record})


@contextmanager
def output_lock(output):
    """Use the same advisory lock for generation and standalone aggregation."""
    import fcntl
    with (Path(output) / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_records(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def sample_seed(task, master_seed, model_id, row, sample_index, n=None, batch_size=None):
    if task == "smollm":
        # Historical SmolLM streams are distinct by the evaluation model ID.
        return int(digest([master_seed, model_id, row["prompt_id"], sample_index])[:15], 16)
    if task in ("qwen25", "qwen3"):
        # Qwen uses the same question/attempt stream across methods.
        return int(digest([master_seed, row["dataset"], row["prompt_id"], sample_index])[:16], 16) % (2**63)
    if task == "maze":
        # A fresh standalone batch stream; training-time per-rank RNG is unavailable.
        position = row["evaluation_index"] * n + sample_index
        return master_seed + (position // batch_size) * batch_size
    raise ValueError(f"Unknown task: {task}")


def validate_record(record, prompt, config, scored=True):
    """Reject changed identities, malformed outputs and inconsistent grades."""
    n = config["samples"]
    index = record.get("sample_index")
    if type(index) is not int or not 0 <= index < n:
        raise ValueError("Invalid response sample index")
    if record.get("prompt_id") != prompt["prompt_id"] or record.get("ground_truth") != prompt["ground_truth"]:
        raise ValueError("Response prompt or ground truth mismatch")
    if record.get("dataset") != prompt["dataset"]:
        raise ValueError("Response dataset mismatch")
    if config.get("identity") and record.get("identity") != config["identity"]:
        raise ValueError("Response belongs to a different evaluation protocol")
    if config.get("schema_version", 0) >= 2:
        if record.get("raw_digest") != raw_digest(record):
            raise ValueError("Raw response payload changed after generation")
        expected_seed = sample_seed(config["task"], config["seed"], config["model_id"], prompt, index,
                                    n, config["batch_size"])
        if type(record.get("seed")) is not int or record["seed"] != expected_seed:
            raise ValueError("Response seed differs from the frozen protocol")
    tokens = record.get("response_token_ids")
    cap = config["profile"]["max_tokens"]
    if (not isinstance(tokens, list) or not tokens or len(tokens) > cap or
            any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError("Invalid response token IDs or response length")
    if type(record.get("response_length")) is not int or record["response_length"] != len(tokens):
        raise ValueError("Response length disagrees with token IDs")
    if record.get("finish_reason") not in ("stop", "length") or not isinstance(record.get("response"), str):
        raise ValueError("Incomplete or invalid generation")
    if type(record.get("hit_response_cap")) is not bool or record["hit_response_cap"] != (len(tokens) >= cap):
        raise ValueError("Response cap flag disagrees with token IDs")
    context = config["profile"].get("context")
    if context is not None:
        total_tokens = len(prompt["prompt_token_ids"]) + len(tokens)
        # vLLM 0.8.4 checks total_len > max_model_len after emitting a token.
        # The archived long Qwen2.5 Olympiad prompts therefore stop at 4097.
        stop_length = context + config["profile"].get("context_stop_extra_tokens", 0)
        if total_tokens > stop_length:
            raise ValueError("Response exceeds the model context capacity")
        if type(record.get("hit_context_limit")) is not bool or record["hit_context_limit"] != (total_tokens >= stop_length):
            raise ValueError("Context-limit flag disagrees with token lengths")
        if record["finish_reason"] == "length" and len(tokens) < cap and total_tokens < stop_length:
            raise ValueError("Length stop did not reach the response cap or context capacity")
    if not scored:
        return
    status = record.get("grading_status")
    if status not in ("ok", "length_cap", "timeout", "verifier_error"):
        raise ValueError("Unresolved grading status")
    if type(record.get("correct")) is not bool or type(record.get("verifier_correct")) is not bool:
        raise ValueError("Each response needs boolean correctness fields")
    strict_cap = config["task"] != "maze" and record["hit_response_cap"]
    if record["correct"] != (record["verifier_correct"] and not strict_cap):
        raise ValueError("Correctness disagrees with verifier and strict response cap")
    if status in ("timeout", "verifier_error", "length_cap") and record["verifier_correct"]:
        raise ValueError("Failed or timed-out verification cannot receive credit")
    if status == "length_cap" and not strict_cap:
        raise ValueError("length_cap status on an uncapped or Maze response")


def validate_batch(records, batch, config, scored=True):
    expected = [(row["prompt_id"], index) for row, index, _ in batch]
    if [(r.get("prompt_id"), r.get("sample_index")) for r in records] != expected:
        raise ValueError("Missing, duplicate, reordered, or unexpected responses in batch")
    for record, (row, _, _) in zip(records, batch):
        validate_record(record, row, config, scored=scored)


def response_files(output):
    output = Path(output)
    legacy = output / "responses.jsonl.gz"
    chunks = sorted((output / "responses").glob("batch_*.jsonl.gz"))
    if legacy.exists() and chunks:
        raise ValueError("Mixed legacy and batch response layouts")
    return [legacy] if legacy.exists() else chunks
