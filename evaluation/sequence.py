"""Generate the paper's fixed response pools with auditable, resumable batches.

Math uses the frozen prompt IDs and historical per-attempt seed functions.
Maze generates a fresh seeded HF pool; the original distributed training-time
RNG state is not available. Its reported estimator is reproduced in passk.py.
"""

import argparse
import importlib.metadata
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from .grading import MATH_TIMEOUT_SECONDS, grade, initialize
from .passk import summarize
from .protocol import (atomic_json, digest, file_digest, output_lock, raw_digest, read_records,
                       sample_seed, validate_batch, write_records)


PROFILES = {
    "maze": dict(samples=2048, temperature=1.0, top_p=1.0, max_tokens=180, max_prompt=320,
                 context=500, batch_size=8192),
    "smollm": dict(samples=256, temperature=0.6, top_p=0.95, max_tokens=2048, max_prompt=512,
                   context=2560, context_stop_extra_tokens=1, batch_size=128,
                   max_num_seqs=128, max_num_batched_tokens=4096),
    "qwen25": dict(samples=512, temperature=0.6, top_p=0.95, max_tokens=3000, max_prompt=4095,
                   context=4096, context_stop_extra_tokens=1, batch_size=1024,
                   max_num_seqs=256, max_num_batched_tokens=32000),
    "qwen3": dict(samples=512, temperature=0.6, top_p=0.95, max_tokens=4096, max_prompt=27904,
                  context=32000, context_stop_extra_tokens=1, batch_size=1024,
                  max_num_seqs=256, max_num_batched_tokens=32000),
}


def load_prompts(paths, tokenizer, task):
    rows = []
    for dataset_index, path in enumerate(paths):
        path = Path(path)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            source_rows = pq.read_table(path).to_pylist()
        elif path.suffix == ".jsonl" or path.suffixes[-2:] == [".jsonl", ".gz"]:
            source_rows = list(read_records(path))
        else:
            raise ValueError(f"Expected Parquet or JSONL evaluation data: {path}")
        if not source_rows:
            raise ValueError(f"Empty evaluation dataset: {path}")
        for index, row in enumerate(source_rows):
            messages = row.get("messages", row.get("prompt"))
            if not isinstance(messages, list):
                raise ValueError(f"Missing prompt messages at {path}, row {index}")
            if task == "maze":
                rendered = "".join(message.get("content", "") for message in messages)
            else:
                # No thinking-mode override; use the frozen experiment template.
                rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            if not tokens or len(tokens) > PROFILES[task]["max_prompt"]:
                raise ValueError(f"Invalid prompt length at {path}, row {index}: {len(tokens)}")
            if "rendered_prompt" in row and row["rendered_prompt"] != rendered:
                raise ValueError(f"Frozen prompt text differs from tokenizer/template at {path}, row {index}")
            if "prompt_token_ids" in row and row["prompt_token_ids"] != tokens:
                raise ValueError(f"Frozen prompt token IDs differ at {path}, row {index}")
            gold = row.get("ground_truth")
            if gold is None:
                gold = row["reward_model"]["ground_truth"]
            if not isinstance(gold, str) or not gold.strip():
                raise ValueError(f"Ground truth must be a nonempty string at {path}, row {index}")
            if task in ("qwen25", "qwen3") and not all(key in row for key in ("prompt_id", "dataset")):
                raise ValueError("Qwen needs frozen prompt_id and dataset fields; prepare reproduction data first")
            prompt_id = row.get("prompt_id")
            if task == "smollm" and "prompt_id" not in row:
                extra = row.get("extra_info", {})
                if extra.get("split") != "test" or "index" not in extra:
                    raise ValueError("SmolLM needs a frozen prompt_id or test split/source index")
                prompt_id = f"gsm8k_platinum:test:{extra['index']}"
            if task == "maze" and "prompt_id" not in row:
                prompt_id = f"{dataset_index}:{index}"
            dataset = row.get("dataset", "gsm8k_platinum" if task == "smollm" else str(path))
            if not isinstance(prompt_id, str) or not prompt_id or not isinstance(dataset, str) or not dataset:
                raise ValueError("Prompt IDs and dataset names must be nonempty strings")
            rows.append(dict(prompt_id=prompt_id, dataset=dataset, row_index=index,
                             evaluation_index=len(rows), messages=messages, prompt=rendered,
                             prompt_token_ids=tokens, ground_truth=gold))
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation prompt IDs must be unique across input files")
    return rows


def requests(rows, n, seed, task, model_id="checkpoint", batch_size=128):
    def emit(group, start, end):
        for row in group:
            for index in range(start, end):
                yield row, index, sample_seed(task, seed, model_id, row, index, n, batch_size)

    if task == "maze":
        yield from emit(rows, 0, n)
    elif task == "smollm":
        for position in range(0, len(rows), 4):
            for start in range(0, n, 32):
                yield from emit(rows[position:position + 4], start, min(start + 32, n))
    else:
        by_dataset = defaultdict(list)
        for row in rows:
            by_dataset[row["dataset"]].append(row)
        # Historical Qwen interleaves datasets, in 8-prompt / 32-attempt shards.
        for start in range(0, n, 32):
            for position in range(0, max(map(len, by_dataset.values())), 8):
                for group in by_dataset.values():
                    yield from emit(group[position:position + 8], start, min(start + 32, n))


def batches(items, size):
    items = iter(items)
    while batch := list(islice(items, size)):
        yield batch


def make_record(row, index, seed, tokens, finish_reason, tokenizer, cap, context=None,
                stop_reason=None, context_stop_extra_tokens=0):
    return dict(prompt_id=row["prompt_id"], dataset=row["dataset"], sample_index=index, seed=seed,
                ground_truth=row["ground_truth"], response=tokenizer.decode(tokens, skip_special_tokens=True),
                response_token_ids=tokens, response_length=len(tokens), finish_reason=finish_reason,
                stop_reason=stop_reason, hit_response_cap=len(tokens) >= cap,
                hit_context_limit=bool(context and len(row["prompt_token_ids"]) + len(tokens) >= context + context_stop_extra_tokens))


def vllm_generator(args, profile, tokenizer):
    from vllm import LLM, SamplingParams

    overrides = dict(eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
    if args.task in ("qwen25", "qwen3"):
        overrides.update(eos_token_id=151643, bos_token_id=151643)
    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", seed=args.seed,
              tensor_parallel_size=args.tensor_parallel_size, max_model_len=profile["context"],
              generation_config="vllm", hf_overrides=overrides, trust_remote_code=False,
              max_num_seqs=profile["max_num_seqs"], max_num_batched_tokens=profile["max_num_batched_tokens"],
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True, enable_chunked_prefill=True)

    def generate(batch):
        sampling = dict(n=1, temperature=profile["temperature"], top_p=profile["top_p"], top_k=-1,
                        min_p=0.0, max_tokens=profile["max_tokens"], min_tokens=0, presence_penalty=0.0,
                        frequency_penalty=0.0, repetition_penalty=1.0, ignore_eos=False,
                        stop=None, stop_token_ids=None)
        if args.task == "smollm":
            sampling.update(detokenize=False, logprobs=0)
        else:
            sampling.update(skip_special_tokens=True, spaces_between_special_tokens=True)
        parameters = [SamplingParams(seed=seed, **sampling) for _, _, seed in batch]
        outputs = llm.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row, _, _ in batch],
                               sampling_params=parameters, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("Generation returned an incomplete batch")
        records = []
        for (row, index, seed), output in zip(batch, outputs):
            if not output.finished or len(output.outputs) != 1 or list(output.prompt_token_ids) != row["prompt_token_ids"]:
                raise RuntimeError("Generation output does not match its request")
            response = output.outputs[0]
            records.append(make_record(row, index, seed, list(response.token_ids), response.finish_reason,
                                       tokenizer, profile["max_tokens"], profile["context"], response.stop_reason,
                                       profile["context_stop_extra_tokens"]))
        return records
    return generate


def maze_generator(args, profile, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    if args.tensor_parallel_size != 1:
        raise ValueError("Standalone Maze evaluation uses one GPU")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                local_files_only=True).to("cuda").eval()
    config = GenerationConfig(do_sample=True, num_beams=1, top_k=0, top_p=profile["top_p"],
                              temperature=profile["temperature"], num_return_sequences=1)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos is None or pad is None:
        raise ValueError("Maze tokenizer must define EOS and padding IDs")

    def generate(batch):
        # Explicit standalone streams support resume, unlike the unavailable
        # training-time RNG state used for the archived Maze validation pools.
        torch.manual_seed(batch[0][2])
        encoded = tokenizer.pad([{"input_ids": row["prompt_token_ids"]} for row, _, _ in batch],
                                padding="max_length", max_length=profile["max_prompt"],
                                return_tensors="pt").to("cuda")
        width = encoded["input_ids"].shape[1]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model.generate(**encoded, generation_config=config, max_new_tokens=profile["max_tokens"],
                                     eos_token_id=eos, pad_token_id=pad, use_cache=True)
        records = []
        for (row, index, seed), sequence in zip(batch, outputs[:, width:].tolist()):
            if eos in sequence:
                sequence = sequence[:sequence.index(eos) + 1]
            reason = "stop" if sequence and sequence[-1] == eos else "length"
            records.append(make_record(row, index, seed, sequence, reason, tokenizer,
                                       profile["max_tokens"], profile["context"]))
        if len(records) != len(batch):
            raise RuntimeError("Generation returned an incomplete batch")
        return records
    return generate


def file_inventory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Use a local, frozen checkpoint/tokenizer directory: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts))
    if not files:
        raise ValueError(f"Empty model or tokenizer directory: {directory}")
    return {str(p.relative_to(directory)): dict(bytes=p.stat().st_size, sha256=file_digest(p)) for p in files}


def validate_output_location(output, model, tokenizer):
    output = Path(output).resolve()
    for source in (model, tokenizer):
        if output.is_relative_to(Path(source).resolve()):
            raise ValueError("--output-dir must be outside the frozen model and tokenizer directories")


def make_config(args, profile, rows, tokenizer):
    versions = {}
    for name in ("torch", "transformers", "vllm", "math-verify", "sympy", "antlr4-python3-runtime", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    config = {key: value for key, value in vars(args).items() if key not in ("resume", "output_dir", "ks", "grader_workers")}
    config.update(schema_version=2, profile=profile, prompts_digest=digest(rows),
                  input_files={str(Path(p).resolve()): file_digest(p) for p in args.data},
                  model_files=file_inventory(args.model), tokenizer_files=file_inventory(args.tokenizer),
                  chat_template=getattr(tokenizer, "chat_template", None), eos_token_id=tokenizer.eos_token_id,
                  package_versions=versions,
                  evaluator_files={p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")},
                  estimator="maze_bootstrap" if args.task == "maze" else "combination",
                  bootstrap=dict(seed=42, repeats=1000),
                  grading_policy=dict(cap_is_incorrect=args.task != "maze", timeout_seconds=MATH_TIMEOUT_SECONDS,
                                      timeout_enforcement=args.grading_timeout_policy),
                  generation_note=("Fresh standalone HF batch streams; original distributed training RNG unavailable"
                                   if args.task == "maze" else "Historical attempt seeds; standalone replica scheduling"))
    config["identity"] = digest(config)
    return config


def prepare_output(output, config, rows, resume=False):
    output = Path(output)
    if resume:
        previous = json.loads((output / "run.json").read_text())
        if previous != config:
            raise ValueError("Resume protocol differs: inputs, model, tokenizer, code, packages or settings changed")
        if list(read_records(output / "prompts.jsonl")) != rows:
            raise ValueError("Saved prompts differ from the frozen protocol")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "run.json", config)
        write_records(output / "prompts.jsonl", rows)
    for name in ("responses", "pending", "receipts"):
        (output / name).mkdir(exist_ok=True)


def run_batches(output, config, batch_iterator, generate_factory, grade_batch):
    """Persist raw batches before grading; resume grades saved raw responses."""
    output, generate, completed = Path(output), None, 0
    for number, batch in enumerate(batch_iterator):
        name = f"batch_{number:07d}"
        final = output / "responses" / f"{name}.jsonl.gz"
        pending = output / "pending" / f"{name}.jsonl.gz"
        receipt = output / "receipts" / f"{name}.json"
        if receipt.exists() and not final.exists():
            raise ValueError(f"Committed response batch is missing: {name}")
        if final.exists():
            records = list(read_records(final))
            validate_batch(records, batch, config)
            if receipt.exists():
                expected = json.loads(receipt.read_text())
                if expected != dict(identity=config["identity"], sha256=file_digest(final), records=len(batch)):
                    raise ValueError(f"Committed response batch changed: {name}")
            else:
                if not pending.exists():
                    raise ValueError(f"Uncommitted scored batch lacks its raw recovery file: {name}")
                raw = list(read_records(pending))
                validate_batch(raw, batch, config, scored=False)
                if [r["raw_digest"] for r in raw] != [r["raw_digest"] for r in records]:
                    raise ValueError(f"Scored batch differs from retained raw responses: {name}")
        else:
            if pending.exists():
                records = list(read_records(pending))
                validate_batch(records, batch, config, scored=False)
            else:
                if generate is None:
                    generate = generate_factory()
                records = generate(batch)
                for record in records:
                    record["identity"] = config["identity"]
                    record["raw_digest"] = raw_digest(record)
                validate_batch(records, batch, config, scored=False)
                write_records(pending, records)
            grades = list(grade_batch([(r["response"], r["ground_truth"], r["hit_response_cap"]) for r in records]))
            if len(grades) != len(records):
                raise RuntimeError("Grading returned an incomplete batch; raw responses are retained")
            for record, result in zip(records, grades):
                record.update(result)
            validate_batch(records, batch, config)
            write_records(final, records)
        atomic_json(receipt, dict(identity=config["identity"], sha256=file_digest(final), records=len(batch)))
        pending.unlink(missing_ok=True)
        completed += len(batch)
        print(f"Scored {completed} responses", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PROFILES, required=True)
    parser.add_argument("--model", required=True, help="Frozen local HF model directory")
    parser.add_argument("--model-id", default="checkpoint", help="Historical SmolLM seed identity; published tokenmean/seqnorm runs use checkpoint")
    parser.add_argument("--tokenizer", help="Frozen local experiment tokenizer/template; defaults to --model")
    parser.add_argument("--data", nargs="+", required=True, help="Prepared Parquet or frozen JSONL files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="Validate and resume the exact saved protocol")
    parser.add_argument("--samples", type=int, help="Fixed responses per prompt; defaults to paper profile")
    parser.add_argument("--ks", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--batch-size", type=int, help="Generation requests per call; defaults to task profile")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--grader-workers", type=int, default=8)
    parser.add_argument("--grading-timeout-policy", choices=("historical", "external"), default="historical",
                        help="historical: paper's MaxRL nested alarms; external: submitted code's strict 1s deadline")
    args = parser.parse_args()
    profile = PROFILES[args.task]
    args.samples = profile["samples"] if args.samples is None else args.samples
    args.batch_size = profile["batch_size"] if args.batch_size is None else args.batch_size
    if min(args.samples, args.batch_size, args.grader_workers, args.tensor_parallel_size) < 1 or args.seed < 0:
        parser.error("Counts must be positive and seed must be nonnegative")
    if not args.model_id:
        parser.error("--model-id must be nonempty")
    if args.ks and any(k < 1 or k > args.samples for k in args.ks):
        parser.error("Every K must be between 1 and --samples")
    if len({Path(path).resolve() for path in args.data}) != len(args.data):
        parser.error("Evaluation files must be distinct")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    args.model = str(Path(args.model).resolve())
    args.tokenizer = str(Path(args.tokenizer or args.model).resolve())
    args.data = [str(Path(p).resolve()) for p in args.data]
    validate_output_location(args.output_dir, args.model, args.tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="left", local_files_only=True)
    if tokenizer.eos_token_id is None:
        parser.error("Tokenizer must define an EOS token")
    if args.task in ("qwen25", "qwen3") and tokenizer.eos_token_id != 151643:
        parser.error("Qwen requires the experiment tokenizer with EOS 151643; use --tokenizer")
    rows = load_prompts(args.data, tokenizer, args.task)
    initialize(args.task, args.grading_timeout_policy)
    print("Freezing model, tokenizer, dataset and code identities", flush=True)
    config = make_config(args, profile, rows, tokenizer)
    output = Path(args.output_dir)
    prepare_output(output, config, rows, args.resume)
    # Advisory lock excludes concurrent writers and is released on process exit.
    with output_lock(output):
        generator = maze_generator if args.task == "maze" else vllm_generator
        iterator = batches(requests(rows, args.samples, args.seed, args.task, args.model_id, args.batch_size), args.batch_size)
        # Non-daemon processes permit the optional external timeout child process.
        with ProcessPoolExecutor(max_workers=args.grader_workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize, initargs=(args.task, args.grading_timeout_policy)) as pool:
            run_batches(output, config, iterator, lambda: generator(args, profile, tokenizer),
                        lambda items: pool.map(grade, items, chunksize=1))
        print(json.dumps(summarize(output, args.ks, already_locked=True), indent=2))


if __name__ == "__main__":
    main()
