#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ============ Configuration ============
MODEL_PATH="$PWD/maze/ckpt-1500"
TRAIN_DATA="$PWD/maze/data/fixed_train7424_seed42/train.parquet"
VAL_DATA="$PWD/maze/data/test.parquet"
CHECKPOINT_DIR="$PWD/checkpoints"

# Training hyperparameters
LR=1e-4
N_ROLLOUTS=128
N_VAL=2048
MAX_RESPONSE_LENGTH=180
LOSS_AGG_MODE=token-mean
# LOSS_AGG_MODE=seq-mean-token-sum-norm

PROJECT_NAME=SERA_Maze
EXPERIMENT_NAME=sera_b256_n128_${LOSS_AGG_MODE}

# Runtime
RAY_DIR=/tmp/sera-maze-$$
RAY_NUM_CPUS=56
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=4
ulimit -n 65535 || true

# ============ Training ============
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir="$RAY_DIR" \
  ray_init.num_cpus="$RAY_NUM_CPUS" \
  ray_init.include_dashboard=False \
  algorithm.adv_estimator=sera \
  algorithm.SERA.enabled=True \
  algorithm.SERA.pq_estimator=historical \
  algorithm.SERA.n_min=2 \
  algorithm.SERA.n_max=512 \
  algorithm.SERA.allocation_mode=dynamic_active_equalized \
  algorithm.SERA.activation_useful_threshold=0.05 \
  algorithm.SERA.use_rollout_count_normalization=True \
  algorithm.SERA.use_rho_normalization=False \
  algorithm.SERA.historical_prior_alpha=0.5 \
  algorithm.SERA.historical_prior_beta=0.5 \
  algorithm.SERA.historical_discount=0.75 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=256 \
  +data.seed=1 \
  data.max_prompt_length=320 \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.apply_chat_template=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.loss_agg_normalizer=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.actor.dtype=float16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size=8192 \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8192 \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=$N_VAL \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  reward_model.reward_manager=prime \
  +reward_model.reward_kwargs.num_processes=16 \
  +reward_model.reward_kwargs.chunksize=64 \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  "trainer.logger=['console','wandb']" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=250 \
  trainer.test_freq=250 \
  trainer.max_actor_ckpt_to_keep=40 \
  trainer.default_local_dir="$CHECKPOINT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.resume_mode=disable \
  trainer.total_epochs=400 \
  trainer.total_training_steps="3000" \
  "$@"
