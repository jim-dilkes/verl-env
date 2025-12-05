#!/usr/bin/env bash

set -euo pipefail

module load cuda/13.0.0
module load gcc/13.3.0

# --- User-tunable variables --------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN_ID="${RUN_ID:-local_test}"
PROJECT_NAME="${PROJECT_NAME:-local_multistep}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fastsnake_grpo}"
NUM_STEPS="${NUM_STEPS:-50}"
MICRO_BSZ="${MICRO_BSZ:-8}"

# --- Environment setup -------------------------------------------------------
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-50000000000}"
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0

# Optional: activate a conda/env (uncomment if needed)
# source activate verl

PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=1500 \
  data.max_response_length=128 \
  data.train_batch_size=256 \
  data.train_files=examples/data/placeholder.parquet \
  data.val_files=examples/data/placeholder.parquet \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BSZ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BSZ" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  algorithm.adv_estimator=grpo \
  algorithm.step_gamma=0.99 \
  algorithm.step_lam=0.95 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  envs.enable=True \
  envs.n_rollouts=32 \
  envs.group_initial_seed=random \
  envs.group_rollout_size=16 \
  envs.duplication_mode=none \
  envs.freeze_completed_episodes=True \
  envs.env_name=fastsnake \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.fastsnake_kwargs.width=10 \
  envs.fastsnake_kwargs.height=10 \
  envs.fastsnake_kwargs.max_rounds=8 \
  envs.fastsnake_kwargs.num_external_snakes=1 \
  envs.fastsnake_kwargs.num_random_snakes=1 \
  envs.fastsnake_kwargs.death_reward=-1 \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="${EXPERIMENT_NAME}_${RUN_ID}" \
  trainer.group="$EXPERIMENT_NAME" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.render=False \
  trainer.total_epochs="$NUM_STEPS" \
  trainer.default_local_dir="outputs/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}_${RUN_ID}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  evaluation=snake_evals_128_min \
  prompt=snake_128 "$@"

