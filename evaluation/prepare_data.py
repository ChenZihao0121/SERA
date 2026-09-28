"""Prepare the paper's pinned math benchmarks and canonical tokenizer.

No model weights are downloaded. Dataset files are checked against recorded
hashes; generated JSONL preserves the IDs used by the original sampling seeds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request


QWEN_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
SMOLLM_INSTRUCTION = " Let's think step by step and output the final answer within \\boxed{}."
MANIFEST = Path(__file__).with_name("paper_assets.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def convert_rows(name, originals, spec):
    """Keep original order and answer serialization, including Olympiad math."""
    if len(originals) != spec["rows"]:
        raise ValueError(f"{name}: expected {spec['rows']} rows, got {len(originals)}")
    if name == "gsm8k_platinum" and digest([[q["question"], q["answer"]] for q in originals]) != spec["question_answer_sha256"]:
        raise ValueError("GSM8K-Platinum questions or reference answers changed")
    result = []
    for index, raw in enumerate(originals):
        question = raw[spec["question"]]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question at {name}/{index}")
        if name == "gsm8k_platinum":
            match = re.search(r"#### (\-?[0-9\.\,]+)", raw["answer"])
            if match is None:
                raise ValueError(f"Missing GSM8K reference number at row {index}")
            gold = match.group(1).replace(",", "")
            prompt_id = f"gsm8k_platinum:test:{index}"
            instruction = SMOLLM_INSTRUCTION
        else:
            if name == "olympiadbench":
                if "<img" in question or raw.get("context") or len(raw["final_answer"]) != 1:
                    raise ValueError("Expected English text-only Olympiad math with one answer")
                gold = raw["final_answer"][0].strip("$")
            else:
                gold = str(raw["answer"])
            prompt_id = f"{name}:{index:04d}"
            instruction = QWEN_INSTRUCTION
        if not gold.strip():
            raise ValueError(f"Empty reference answer at {name}/{index}")
        result.append(dict(prompt_id=prompt_id, dataset=name, index=index,
                           ground_truth=gold, messages=[dict(role="user", content=question + instruction)],
                           dataset_revision=spec["revision"], source_sha256=digest(raw),
                           question=question, source=raw))
    if len({r["question"] for r in result}) != len(result):
        raise ValueError(f"Duplicate questions in {name}")
    return result


def render_rows(rows, tokenizer, task):
    for row in rows:
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if ids != tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                                return_dict=False):
            raise ValueError("Tokenizer rendering paths disagree")
        limit = {"smollm": 512, "qwen25": 4095, "qwen3": 32000 - 4096}[task]
        if not ids or len(ids) > limit:
            raise ValueError(f"Prompt exceeds the historical context policy: {row['prompt_id']}")
        row.update(rendered_prompt=rendered, prompt_token_ids=ids)
    return rows


def training_validation_rows(rows):
    routes = {"gsm8k_platinum": "openai/gsm8k", "math500": "DigitalLearningGmbH/MATH-lighteval"}
    return [dict(data_source=routes.get(r["dataset"], r["dataset"]), ability="math",
                 prompt=r["messages"], reward_model=dict(style="rule", ground_truth=r["ground_truth"]),
                 extra_info=dict(split="test", index=r["index"], question=r["question"],
                                 answer=str(r["source"].get("answer", r["ground_truth"])))) for r in rows]


def prepare(task, output, tokenizer_path=None, local_files_only=False, raw_dir=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; existing assets are never overwritten")
    manifest = json.loads(MANIFEST.read_text())
    profile = manifest["profiles"][task]
    tok_spec = profile["tokenizer"]
    if tokenizer_path is None:
        tokenizer_path = snapshot_download(tok_spec["repo"], revision=tok_spec["revision"],
            allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.json",
                            "merges.txt", "chat_template.jinja", "config.json", "generation_config.json"],
            local_files_only=local_files_only)
    tokenizer_path = Path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    if tok_spec.get("tokenizer_sha256") and file_sha(tokenizer_path / "tokenizer.json") != tok_spec["tokenizer_sha256"]:
        raise ValueError("Tokenizer vocabulary differs from the paper snapshot")
    if tok_spec.get("template_sha256") and digest(tokenizer.chat_template) != tok_spec["template_sha256"]:
        raise ValueError("Tokenizer chat template differs from the paper snapshot")
    if tokenizer.eos_token_id != tok_spec["eos_token_id"]:
        raise ValueError("Tokenizer EOS differs from the paper snapshot")
    prepared = []
    for name in profile["datasets"]:
        spec = manifest["datasets"][name]
        if raw_dir is not None:
            source = Path(raw_dir) / (name + Path(spec["file"]).suffix)
        elif spec.get("source") == "github":
            if local_files_only:
                raise ValueError("For offline OlympiadBench use --raw-dir with pinned source files")
            # Pinned public source, checked before parsing or writing usable assets.
            import tempfile
            with urllib.request.urlopen(spec["url"], timeout=60) as response:
                data = response.read()
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp:
                temp.write(data)
                source = Path(temp.name)
        else:
            source = Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset",
                revision=spec["revision"], local_files_only=local_files_only))
        try:
            source_hash = file_sha(source)
            if spec.get("source_sha256") and source_hash != spec["source_sha256"]:
                raise ValueError(f"Source file differs from paper snapshot: {name}")
            originals = (pq.read_table(source).to_pylist() if source.suffix == ".parquet" else
                         [json.loads(line) for line in source.read_text().splitlines() if line.strip()])
            rows = render_rows(convert_rows(name, originals, spec), tokenizer, task)
            prepared.append((name, rows, source_hash))
        finally:
            if raw_dir is None and spec.get("source") == "github":
                source.unlink()
    # Complete downloads and validation before creating the output directory.
    output.mkdir(parents=True, exist_ok=True)
    token_dest = output / "canonical_tokenizer"
    token_dest.mkdir()
    for path in tokenizer_path.iterdir():
        if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model"):
            shutil.copyfile(path, token_dest / path.name)
    receipt = dict(task=task, tokenizer=tok_spec, datasets={}, model_weights_downloaded=False)
    for name, rows, source_hash in prepared:
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        pq.write_table(pa.Table.from_pylist(training_validation_rows(rows)), output / f"{name}.parquet")
        receipt["datasets"][name] = dict(rows=len(rows), source_sha256=source_hash,
            prompts_sha256=file_sha(path), parquet_sha256=file_sha(output / f"{name}.parquet"))
    receipt["tokenizer_files"] = {p.name: file_sha(p) for p in token_dest.iterdir()}
    (output / "asset_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["smollm", "qwen25", "qwen3"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", help="Use an existing canonical tokenizer, verified against the snapshot")
    parser.add_argument("--raw-dir", help="Pinned raw files named <dataset>.jsonl or <dataset>.parquet")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    prepare(args.task, args.output, args.tokenizer, args.local_files_only, args.raw_dir)
