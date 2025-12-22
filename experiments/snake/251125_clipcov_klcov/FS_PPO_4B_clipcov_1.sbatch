#!/bin/bash
#SBATCH --job-name=FS_PPO_4B_clipcov_1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=250GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=swarm_h100,swarm_a100,a100
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=2

# --- Define experiment variables ---
project_name=verl_env
experiment_name=FS_PPO_4B_clipcov
run_number=1
number_of_gpus=2
WANDB_TAGS='["snake","project:verl-env","vllm:0.9.1"]'

job_id=${SLURM_JOB_ID:-local}
run_id=${experiment_name}_run${run_number}_slu${job_id}
user_name=${USER:-$(whoami)}
short_temp_dir=/tmp/ray/${user_name}/${job_id}
mkdir -p "$short_temp_dir"
readable_log_root=/scratch/${user_name}/ray_logs
mkdir -p "$readable_log_root"

experiment_run_name=${experiment_name}_run${run_number}_slu${SLURM_JOB_ID}
export WANDB_DIR="wandb/${experiment_run_name}"
mkdir -p "$WANDB_DIR"
profile_dump_dir=/scratch/${user_name}/profile_dumps/${experiment_run_name}

# --- Setup ---
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

eval "$(conda shell.bash hook)"
conda activate verl_latest_vllm9

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 

# Introduce random delay to avoid simultaneous Ray init on H100 nodes
RANDOM_DELAY=$(( ( RANDOM % 120 ) + 1 ))
echo "Sleeping for $RANDOM_DELAY seconds to avoid simultaneous Ray cluster startup..."
sleep $RANDOM_DELAY

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES 


# --- Configure Environment for Ray ---

export RAY_OBJECT_STORE_MEMORY=50000000000
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=false
export VERL_PPO_LOGGING_LEVEL=DEBUG

model=Qwen/Qwen3-4B-Instruct-2507
micro_batch_size=16

max_prompt_length=768
max_response_length=256
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 256))

rollout_max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 2560))
rollout_max_num_seqs=$((2048))

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
  actor_rollout_ref.rollout.engine_kwargs.vllm.max_seq_len_to_capture=$((max_token_len_per_gpu + critic_max_token_len_per_gpu)) \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
  actor_rollout_ref.rollout.max_model_len=$max_token_len_per_gpu \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
  actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 \
  actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 \
  actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
  trainer.balance_batch=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.update_weights_bucket_megabytes=256 \
  +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +critic.model.override_config.torch_dtype=bfloat16 \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.model.path=$model \
  critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
  critic.forward_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
  algorithm.step_gamma=0.99 \
  algorithm.token_gamma=1.0 \
  algorithm.step_lam=0.95 \
  algorithm.token_lam=1.0 \
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
  trainer.log_val_generations=1 \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_run_name} \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.critic_warmup=5 \
  trainer.critic_warmup_micro_batch_size_per_gpu=16 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.render=False \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=1000 \
  trainer.default_local_dir=/scratch/${user_name}/checkpoints/${project_name}/${experiment_name}_run${run_number} \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.tags=$WANDB_TAGS \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  evaluation=snake_evals \
  prompt=snake 2>&1 | tee verl_demo.log

actual_session_dir="${short_temp_dir}/session_latest"
readable_log_path="${readable_log_root}/${experiment_run_name}"
if [ -d "$actual_session_dir" ]; then
    ln -sfn "$actual_session_dir" "$readable_log_path"
    echo "Ray logs accessible at: $readable_log_path"
fi

if [ -n "$RAY_HEAD_PID" ]; then
    echo "Stopping Ray cluster (PID: $RAY_HEAD_PID)..."
    ray stop
    kill $RAY_HEAD_PID 2>/dev/null || true
    echo "Ray cluster stopped."
fi

