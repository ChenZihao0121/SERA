#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ============ Configuration ============
DATA_DIR=./data/imagenet

# Training hyperparameters
LR=0.1
SEED=69
N0=128
N_MAX=$((4 * N0))

CHECKPOINT_DIR=./checkpoints/imagenet_sera_n${N0}
PROJECT_NAME=sera_imagenet
EXPERIMENT_NAME=sera_oracle_n${N0}_seed${SEED}

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=$SEED
export HF_HUB_HTTP_TIMEOUT=300

# ============ Training ============
python3 -m verl.cifar10_experiments.sampling_based_rl_objective_experiments \
    --wandb \
    --no-amp \
    --lr $LR \
    --batch-size 256 \
    --epochs 20 \
    --seed $SEED \
    --model-type resnet \
    --model-depth 50 \
    --dataset-name imagenet256 \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-steps 100000 \
    --wandb_runname "$EXPERIMENT_NAME" \
    --wandb-project "$PROJECT_NAME" \
    --eval_every_k 1000 \
    --num-workers 32 \
    --max_grad_norm 1e5 \
    --advantage-type sera \
    --max-k 2048 \
    --num_train_rollouts_per_example $N0 \
    --num_test_rollouts_per_example 1024 \
    --sera-n-min 2 \
    --sera-n-max $N_MAX \
    --sera-activation-useful-threshold 0.05 \
    --exact-logp-gradient-monitor-every-k 1000 \
    "$@"
