#!/bin/bash
# run_state_visitation_smoke.sh
set -euo pipefail

# Optional: adjust or remove these if not on the cluster modules
# module load conda/python3
# eval "$(conda shell.bash hook)"
# conda activate verl

project_name=verl_env
experiment_name=FS_PPO_TEST
number_of_gpus=1

module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

eval "$(conda shell.bash hook)"
conda activate verl

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export VERL_PPO_LOGGING_LEVEL=DEBUG

model="Qwen/Qwen2.5-0.5B-Instruct"
max_prompt_length=256
max_response_length=128
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 128))
rollout_max_num_batched_tokens=$((4 * max_token_len_per_gpu + 256))
rollout_max_num_seqs=256

python -m verl.trainer.main_ppo \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.train_batch_size=16 \
  data.train_files=examples/data/placeholder.parquet \
  data.val_files=examples/data/placeholder.parquet \
  actor_rollout_ref.model.path=${model} \
  critic.model.path=${model} \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.max_model_len=$max_token_len_per_gpu \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$number_of_gpus \
  actor_rollout_ref.rollout.engine_kwargs.vllm.max_seq_len_to_capture=$((max_token_len_per_gpu + critic_max_token_len_per_gpu)) \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  envs.enable=True \
  envs.n_rollouts=8 \
  envs.env_name=fastsnake \
  envs.freeze_completed_episodes=True \
  envs.duplication_mode=none \
  envs.vec_env_multiprocessing=fork \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.fastsnake_kwargs.width=10 \
  envs.fastsnake_kwargs.height=10 \
  envs.fastsnake_kwargs.max_rounds=8 \
  envs.fastsnake_kwargs.num_external_snakes=1 \
  envs.fastsnake_kwargs.num_random_snakes=0 \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_name} \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.total_training_steps=1 \
  trainer.test_freq=1 \
  trainer.save_freq=99999 \
  trainer.log_val_generations=0 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  +evaluation.environments="[{name:StateVisitationTest,n_rollouts:16,episode_length:8,generation:{temperature:1.0,top_p:1.0,top_k:-1,min_p:0.0,do_sample:False},initial_seed:1234,seed_group_size:4,freeze_completed_episodes:True,duplication_mode:none,env_name:fastsnake,task:none,format_penalty:0.0,instruction_prompt:'You control a snake. Respond with <action> tags.',captioner:{type:naive,max_text_history:0},fastsnake_kwargs:{width:10,height:10,max_rounds:8,num_external_snakes:1,num_random_snakes:0,death_reward:-1,step_reward:0,no_respawn:True}},{name:EntropyProbeTest,n_rollouts:12,episode_length:2,generation:{temperature:0.0,top_p:1.0,top_k:-1,min_p:0.0,do_sample:False},initial_seed:2025,seed_group_size:6,freeze_completed_episodes:True,duplication_mode:none,env_name:fastsnake,task:none,format_penalty:0.0,instruction_prompt:'You control a snake. Respond with <action> tags.',captioner:{type:naive,max_text_history:0},action_entropy:{enabled:True,n_samples:8,temperature:1.2,measure_at_steps:start,step_interval:1,exclusive_metric:True,max_batch_size:32},fastsnake_kwargs:{width:10,height:10,max_rounds:4,num_external_snakes:1,num_random_snakes:0,death_reward:-1,step_reward:0,no_respawn:True}}]" \
  prompt=snake 2>&1 | tee visitation_smoke.log