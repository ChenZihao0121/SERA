# SERA

**Scale-Equalized Rollout Allocation for Maximum Likelihood Reinforcement Learning**

SERA allocates a fixed rollout budget across prompts using success-probability estimates. This repository provides the training and evaluation code for our paper.

## Installation

Run the following commands from the repository root:

```bash
conda create -n sera python=3.10.12 -y
conda activate sera
pip install setuptools==75.8.2 wheel==0.45.1 ninja packaging psutil
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Use `wandb login` for online logging or set `WANDB_MODE=offline`.

## Training data

Prepare the sequence datasets as Parquet files. Default training inputs are:

| Experiment | Dataset | Location |
|---|---|---|
| ImageNet ([ResNet-50](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet50.html)) | [benjamin-paine/imagenet-1k-256x256](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256) | Hugging Face cache |
| Maze ([Qwen2 SFT](https://github.com/tajwarfahim/maxrl/tree/main/maze/ckpt-1500)) | [SERA-Maze-17x17](https://huggingface.co/datasets/capybaraczh/SERA-Maze-17x17), 7,424 training examples | `maze/data/fixed_train7424_seed42/train.parquet` |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main`, train split | `data/gsm8k/train.parquet` |
| [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | 8,192 MATH levels 3–5 examples from [hkust-nlp/SimpleRL-Zoo-Data](https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data) | `data/math35/train.parquet` |
| [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | [POLARIS-Project/Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) | `data/polaris/train.parquet` |

Edit `TRAIN_DATA`, `VAL_DATA`, and `MODEL_PATH` at the top of each training script.

## ImageNet

Train ResNet-50 on four GPUs with an average of 128 rollouts per image:

```bash
bash imagenet/imagenet_training_script.sh
```

## Maze

```bash
bash maze/maze_17.sh
```

## SmolLM2-360M-Instruct in GSM8K

Train on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) with eight GPUs to step 2,000. Use [GSM8K-Platinum](https://huggingface.co/datasets/madrylab/gsm8k-platinum) for validation.

```bash
bash smollm/smollm.sh
```

## Qwen2.5-Math-1.5B / Qwen3-4B-Base

Qwen2.5 trains on [MATH levels 3–5](https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data) with eight GPUs to step 600:

```bash
bash qwen/qwen.sh
```

Qwen3 trains on [POLARIS](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) with four nodes of eight GPUs to step 1,000:

```bash
bash qwen/qwen3.sh
```

## Evaluation

| Model | Benchmarks |
|---|---|
| SmolLM2-360M-Instruct | [GSM8K-Platinum](https://huggingface.co/datasets/madrylab/gsm8k-platinum) |
| Qwen2.5-Math-1.5B | [BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME), [AIME 2025](https://huggingface.co/datasets/math-ai/aime25), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [OlympiadBench](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/data/olympiadbench/test.jsonl) |
| Qwen3-4B-Base | [BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME), [AIME 2025](https://huggingface.co/datasets/math-ai/aime25), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [Minerva Math](https://huggingface.co/datasets/math-ai/minervamath) |

Set `TASK` to `smollm`, `qwen25`, or `qwen3`, and `MODEL_PATH` to a local Hugging Face model directory. Pass prepared evaluation files to `--data`. The tokenizer is loaded from the model directory by default.

```bash
TASK=smollm
MODEL_PATH=models/my_checkpoint

python -m evaluation.sequence --task "$TASK" --model "$MODEL_PATH" \
  --data data/evaluation/"$TASK"/*.jsonl \
  --output-dir "outputs/$TASK"
```

## Acknowledgements

Built on [MaxRL](https://github.com/tajwarfahim/maxrl) and [verl](https://github.com/verl-project/verl). We thank the authors and contributors for their work.

## Citation

```bibtex
@misc{chen2026serascaleequalizedrolloutallocation,
  title={SERA: Scale-Equalized Rollout Allocation for Maximum Likelihood Reinforcement Learning},
  author={Zihao Chen and Fanxiang Xiong and Hongran Ren and Xuefeng Bai and Zhongxiang Dai and Kehai Chen and Zhiguo Zhang and Zhiyong Wang and Yu Cheng},
  year={2026},
  eprint={2609.36552},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.36552},
}
```
