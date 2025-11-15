#!/bin/bash
# Minimal GRPO test script for Qwen2.5-0.5B-Instruct with LoRA
# Optimized for single GPU with limited memory
#
# Note: 
# - Uses eager attention instead of flash_attention_2 to avoid GLIBC compatibility issues
# - Disables use_remove_padding to avoid flash_attn import (GLIBC issue)
# - Loads CUDA module to enable nvcc for vllm/flashinfer compilation
#
# Usage:
#   1. If you have GSM8K dataset already:
#      bash run_qwen2_5-0.5b_test_grpo_lora.sh
#
#   2. To use a minimal test dataset instead:
#      python3 create_minimal_test_dataset.py
#      # Then edit this script to use: data.train_files=$HOME/data/test_minimal/train.parquet
#      # and: data.val_files=$HOME/data/test_minimal/test.parquet

# Load CUDA module to enable nvcc for vllm/flashinfer compilation
module load cuda/13.0.0
module load gcc/13.3.0
# export CUDA_HOME=${CUDA_HOME:-/iridisfs/ixsoftware/cuda/13.0.0/}

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=$HOME/data/test_minimal/train.parquet \
    data.val_files=$HOME/data/test_minimal/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_test' \
    trainer.experiment_name='qwen2.5_0.5b_grpo_lora_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@

