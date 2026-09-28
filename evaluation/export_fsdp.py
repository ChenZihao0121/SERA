# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0; see LICENSE.
# Adapted from verl's model merger for the main FSDP causal-language-model runs.
"""Convert an actor FSDP checkpoint into a local Hugging Face model."""

import argparse
import re
from pathlib import Path


def merge_weights(actor, dtype):
    import torch
    from torch.distributed.tensor import DTensor

    candidates = list(actor.glob("model_world_size_*_rank_0.pt"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one rank-zero model checkpoint in --actor")
    world_size = int(re.fullmatch(r"model_world_size_(\d+)_rank_0.pt", candidates[0].name).group(1))
    first = torch.load(candidates[0], map_location="cpu", weights_only=False)
    pivot = next((value for value in first.values() if isinstance(value, DTensor)), None)
    count = world_size
    mesh_names = ("fsdp",)
    if pivot is not None:
        mesh_names = pivot.device_mesh.mesh_dim_names
        if mesh_names not in (("fsdp",), ("ddp", "fsdp")):
            raise ValueError(f"Unsupported FSDP mesh: {mesh_names}")
        # A single FSDP group is sufficient when groups are replicated across DDP.
        count = pivot.device_mesh.mesh.shape[-1]
    parts = [first]
    for rank in range(1, count):
        parts.append(torch.load(actor / f"model_world_size_{world_size}_rank_{rank}.pt",
                                map_location="cpu", weights_only=False))
    merged = {}
    for key in list(first):
        values = [part.pop(key) for part in parts]
        if isinstance(values[0], DTensor):
            placements = values[0].placements
            if any(value.placements != placements or value.shape != values[0].shape for value in values):
                raise ValueError(f"Inconsistent distributed parameter: {key}")
            if mesh_names == ("ddp", "fsdp"):
                if not placements[0].is_replicate():
                    raise ValueError("Expected a replicated DDP dimension")
                placements = placements[1:]
            if len(placements) != 1:
                raise ValueError("Only one FSDP partition dimension is supported")
            placement = placements[0]
            local = [value._local_tensor for value in values]
            if placement.is_replicate():
                tensor = local[0]
            elif placement.is_shard():
                tensor = torch.cat(local, dim=placement.dim)
            else:
                raise ValueError(f"Unsupported distributed placement: {placement}")
            if tensor.shape != values[0].shape:
                raise ValueError(f"Incomplete distributed parameter: {key}")
        elif world_size == 1:
            tensor = values[0]
        else:
            # Legacy checkpoints store partitioned tensors directly.
            tensor = torch.cat(values, dim=0) if values[0].ndim else values[0]
        merged[key] = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True, help="global_step_*/actor directory")
    parser.add_argument("--output", type=Path, required=True, help="New HF model directory")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    config_dir = args.actor if (args.actor / "config.json").exists() else args.actor / "huggingface"
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
    dtype = getattr(torch, args.dtype)
    weights = merge_weights(args.actor, dtype)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    if (config_dir / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(config_dir, local_files_only=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved model and tokenizer to {args.output}")


if __name__ == "__main__":
    main()
