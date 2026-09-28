#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ============ Configuration ============
MODEL_PATH=HuggingFaceTB/SmolLM2-360M-Instruct
TRAIN_DATA="$PWD/data/gsm8k/train.parquet"
VAL_DATA="$PWD/data/gsm8k/test.parquet"
CHECKPOINT_DIR="$PWD/checkpoints"

# Training hyperparameters
LR=1e-5
N_ROLLOUTS=16
N_VAL=32
MAX_RESPONSE_LENGTH=2048
LOSS_AGG_MODE=token-mean
LOSS_AGG_NORMALIZER=null
# LOSS_AGG_MODE=seq-mean-token-sum-norm
# LOSS_AGG_NORMALIZER=2048

PROJECT_NAME=SERA_SmolLM2_GSM8K
EXPERIMENT_NAME=sera_b256_n16_${LOSS_AGG_MODE}

# Runtime
RAY_DIR=/tmp/sera-smollm-$$
RAY_NUM_CPUS=112
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
  algorithm.SERA.n_max=64 \
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
  data.filter_overlong_prompts=True \
  data.max_prompt_length=512 \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.loss_agg_normalizer="$LOSS_AGG_NORMALIZER" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=$N_VAL \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.reward_manager=multi_thread \
  +reward_model.reward_kwargs.num_reward_actors=64 \
  +reward_model.reward_kwargs.zero_reward_on_max_response_length=True \
  +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  "trainer.logger=['console','wandb']" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.balance_batch="False" \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.default_local_dir="$CHECKPOINT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.resume_mode=disable \
  trainer.total_epochs=200 \
  trainer.total_training_steps=2000 \
  "$@"
