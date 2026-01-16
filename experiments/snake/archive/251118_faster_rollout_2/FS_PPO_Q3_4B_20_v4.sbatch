#!/bin/bash
#SBATCH --job-name=FS_PPO_Q3_4B_20
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=250GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=a100,scavenger_mathsa100
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=2

# --- Define experiment variables ---
project_name=AAMAS_msrl
experiment_name=FS_PPO_Q3_4B
run_number=20 
number_of_gpus=2
WANDB_TAGS='["snake","project:verl-env"]'

# --- Setup ---
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

eval "$(conda shell.bash hook)"
conda activate verl_latest

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES 

# # --- Clean up previous Ray instances (this is good practice) ---
# ray stop --force 2>/dev/null || true
# sleep 5
# rm -rf /tmp/ray/session_* 2>/dev/null || true
# rm -rf /tmp/ray/raylet_* 2>/dev/null || true
# rm -rf /tmp/ray/plasma_* 2>/dev/null || true
# sleep 2
# --- Configure Environment for Ray ---
export RAY_OBJECT_STORE_MEMORY=50000000000

# Additional Ray configuration for stability
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=false

export VERL_PPO_LOGGING_LEVEL=DEBUG

# --- Define model and batch size ---
model=Qwen/Qwen3-4B-Instruct-2507
# model=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
micro_batch_size=8

max_prompt_length=768
max_response_length=256
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 256))

rollout_max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 256))
rollout_max_num_seqs=$((2048))


# --- Run your main application ---
PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.train_batch_size=256 \
  data.train_files=examples/data/placeholder.parquet \
  data.val_files=examples/data/placeholder.parquet \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$number_of_gpus \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
  trainer.balance_batch=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  critic.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.update_weights_bucket_megabytes=64 \
  +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +critic.model.override_config.torch_dtype=bfloat16 \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.model.path=$model \
  critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
  critic.forward_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
  algorithm.step_gamma=0.99 \
  algorithm.step_lam=0.95 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  envs.enable=True \
  envs.n_rollouts=32 \
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
  envs.vec_env_multiprocessing=fork \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_name}_run${run_number}_slu${SLURM_JOB_ID} \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.critic_warmup=5 \
  trainer.critic_warmup_micro_batch_size_per_gpu=8 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.render=False \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=1000 \
  trainer.default_local_dir=/scratch/jsbd1n24/checkpoints/${project_name}/${experiment_name}_run${run_number} \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.tags=$WANDB_TAGS \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  evaluation=snake_evals \
  prompt=snake 2>&1 | tee verl_demo.log

# --- Cleanup Ray Cluster ---
if [ -n "$RAY_HEAD_PID" ]; then
    echo "Stopping Ray cluster (PID: $RAY_HEAD_PID)..."
    ray stop
    kill $RAY_HEAD_PID 2>/dev/null || true
    echo "Ray cluster stopped."
fi



