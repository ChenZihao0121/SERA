#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ============ Configuration ============
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
TRAIN_DATA="$PWD/data/math35/train.parquet"
VAL_DATA="['$PWD/data/aime25/test.parquet','$PWD/data/math500/test.parquet']"
CHECKPOINT_DIR="$PWD/checkpoints"

# Training hyperparameters
LR=1e-6
N_ROLLOUTS=16
N_VAL=32
MAX_RESPONSE_LENGTH=3000
N_GPUS_PER_NODE=8
NNODES=1

PROJECT_NAME=SERA_Qwen25_Math35
EXPERIMENT_NAME=sera_qwen25-math15_b256_n16

# Runtime
RAY_DIR=/tmp/sera-qwen-$$
RAY_NUM_CPUS=112
export PYTHONPATH="$PWD:$PYTHONPATH"
export SEED=79

# ============ Training ============
python3 -W ignore -m verl.trainer.main_ppo \
  algorithm.adv_estimator=sera \
  algorithm.SERA.enabled=True \
  algorithm.SERA.n_min=2 \
  algorithm.SERA.n_max=64 \
  algorithm.SERA.allocation_mode=dynamic_active_equalized \
  algorithm.SERA.activation_useful_threshold=0.05 \
  algorithm.SERA.pq_estimator=historical \
  algorithm.SERA.historical_prior_alpha=0.5 \
  algorithm.SERA.historical_prior_beta=0.5 \
  algorithm.SERA.historical_discount=0.75 \
  algorithm.SERA.use_rollout_count_normalization=True \
  algorithm.SERA.use_rho_normalization=False \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="$TRAIN_DATA" \
  "data.val_files=$VAL_DATA" \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.filter_overlong_prompts=True \
  data.apply_chat_template=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=512 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=512 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=$N_ROLLOUTS \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=$N_VAL \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=multi_thread \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.val_on_last_step=True \
  "trainer.logger=['console','wandb']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$CHECKPOINT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq=100 \
  trainer.max_actor_ckpt_to_keep=8 \
  trainer.max_critic_ckpt_to_keep=8 \
  trainer.test_freq=50 \
  trainer.total_epochs=32 \
  trainer.total_training_steps=600 \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null \
  ray_init.ray_dir="$RAY_DIR" \
  ray_init.num_cpus="$RAY_NUM_CPUS" \
  ray_init.include_dashboard=False \
  "$@"
