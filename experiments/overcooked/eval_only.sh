#!/bin/bash
# EVAL-ONLY script for environment evaluation - NO actor/critic loading
#
# Key difference from training: uses load_format=auto instead of dummy_dtensor
# This loads model ONCE in vLLM, not in FSDP + vLLM
#
# Usage: bash experiments/overcooked/eval_only.sh

set -e

export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_OBJECT_STORE_MEMORY=10000000000
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export VERL_PPO_LOGGING_LEVEL=DEBUG
export VERL_MULTIENV_EVALUATOR_DEBUG=1

MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-4B-Instruct-2507"}
FOLDER_NAME="models--$(echo "$MODEL_ID" | sed 's/\//--/')"
SNAPSHOT_DIR="$HF_HOME/hub/$FOLDER_NAME/snapshots"

if [ -d "$SNAPSHOT_DIR" ]; then
    LATEST_SNAPSHOT=$(ls -d $SNAPSHOT_DIR/* | head -n 1)
    MODEL_PATH="$LATEST_SNAPSHOT"
    echo "Model path: $MODEL_PATH"
else
    echo "Error: Could not find local cache for $MODEL_ID"
    echo "Run: huggingface-cli download $MODEL_ID"
    exit 1
fi

number_of_gpus=1
ray_num_cpus=${SLURM_CPUS_PER_TASK:-8}
n_rollouts=10

echo "=========================================="
echo "EVAL-ONLY: Environment Evaluation"
echo "Model: $MODEL_ID"
echo "Load format: auto (direct HF load, NO FSDP)"
echo "GPU memory target: 0.85"
echo "=========================================="

# Use main_generation style approach but with environments
# KEY CHANGES:
# 1. load_format=auto (not dummy_dtensor)
# 2. critic.enable=False
# 3. algorithm.adv_estimator=grpo (no critic needed)
# 4. total_training_steps=0, val_before_train=True

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  +ray_kwargs.ray_init.include_dashboard=false \
  ray_kwargs.ray_init.num_cpus=$ray_num_cpus \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.train_batch_size=8 \
  data.train_files=examples/data/placeholder.parquet \
  data.val_files=examples/data/placeholder.parquet \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$number_of_gpus \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.max_model_len=640 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.calculate_log_probs=false \
  +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
  critic.enable=False \
  algorithm.adv_estimator=grpo \
  envs.enable=True \
  envs.n_rollouts=$n_rollouts \
  envs.env_name=overcooked \
  envs.format_penalty=1 \
  envs.overcooked_kwargs.layout_name=cramped_room \
  envs.overcooked_kwargs.horizon=40 \
  envs.overcooked_kwargs.partner_policy=none \
  envs.overcooked_kwargs.shaped_reward=true \
  envs.captioner.type=naive \
  envs.vec_env_multiprocessing=spawn \
  trainer.val_before_train=True \
  trainer.total_training_steps=0 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  evaluation=overcooked_diverse \
  prompt=overcooked 2>&1 | tee eval_only.log

echo ""
echo "=========================================="
echo "Eval complete! Check log for diverse/* metrics:"
echo "  grep 'diverse/' eval_only.log"
echo "=========================================="
