#!/bin/bash
#SBATCH --job-name=FS_test_multi_action_L4
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=swarm_l4
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1

set -e

mkdir -p logs

# --- Define experiment variables ---
project_name=verl_env
experiment_name=test_multi_action_L4
run_number=1
number_of_gpus=1
WANDB_TAGS='["snake","project:verl-env","captioner:multi_action","reasoning:true","epsilon:0.1"]'

job_id=${SLURM_JOB_ID:-local}
user_name=${USER:-$(whoami)}
short_temp_dir=/tmp/ray/${user_name}/${job_id}
mkdir -p "$short_temp_dir"
readable_log_root=/scratch/${user_name}/ray_logs
mkdir -p "$readable_log_root"

experiment_run_name=${experiment_name}_run${run_number}_slu${job_id}
export WANDB_DIR="wandb/${experiment_run_name}"
mkdir -p "$WANDB_DIR"

# Ensure we run from the repo root (works whether launched via sbatch from anywhere)
cd "${SLURM_SUBMIT_DIR:-/home/jsbd1n24/verl-env}"

# --- Setup ---
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

eval "$(conda shell.bash hook)"
conda activate verl

# --- Offline / cluster-friendly defaults ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Introduce random delay to avoid simultaneous Ray init on shared partitions
RANDOM_DELAY=$(( ( RANDOM % 120 ) + 1 ))
echo "Sleeping for $RANDOM_DELAY seconds to avoid simultaneous Ray cluster startup..."
sleep $RANDOM_DELAY

# --- Configure Environment for Ray ---
export RAY_OBJECT_STORE_MEMORY=10000000000
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export VERL_PPO_LOGGING_LEVEL=DEBUG

# Match the login-node test model (must already be cached)
MODEL_ID=Qwen/Qwen3-0.6B-Base
FOLDER_NAME="models--$(echo "$MODEL_ID" | sed 's/\//--/')"

# HF_HOME is typically defined on cluster; fall back to the standard cache location.
HF_HOME=${HF_HOME:-"${HOME}/.cache/huggingface"}
SNAPSHOT_DIR="$HF_HOME/hub/$FOLDER_NAME/snapshots"

if [ -d "$SNAPSHOT_DIR" ]; then
    LATEST_SNAPSHOT=$(ls -d $SNAPSHOT_DIR/* | head -n 1)
    MODEL_PATH="$LATEST_SNAPSHOT"
    echo "Model path: $MODEL_PATH"
else
    echo "Error: Could not find local cache for $MODEL_ID"
    echo "Run (on a machine with access): huggingface-cli download $MODEL_ID"
    exit 1
fi

# Tuned for 24GB L4 GPUs
micro_batch_size=2
max_prompt_length=384
max_response_length=256
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 64))
rollout_max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 512))
rollout_max_num_seqs=128

# Ray defaults num_cpus=None which makes it try to use *all* visible CPUs.
ray_num_cpus=${SLURM_CPUS_PER_TASK:-4}
if ! [[ "$ray_num_cpus" =~ ^[0-9]+$ ]]; then
    echo "Error: ray_num_cpus must be an integer, got '$ray_num_cpus' (SLURM_CPUS_PER_TASK='$SLURM_CPUS_PER_TASK')" >&2
    exit 1
fi

echo "Using ray_num_cpus=$ray_num_cpus"

auto_log="logs/test_multi_action_L4-${job_id}.log"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  +ray_kwargs.ray_init.include_dashboard=false \
  ray_kwargs.ray_init.num_cpus=$ray_num_cpus \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.train_batch_size=16 \
  data.train_files=examples/data/placeholder.parquet \
  data.val_files=examples/data/placeholder.parquet \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$number_of_gpus \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
  actor_rollout_ref.rollout.engine_kwargs.vllm.max_seq_len_to_capture=$((max_token_len_per_gpu + critic_max_token_len_per_gpu)) \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.max_model_len=$max_token_len_per_gpu \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
  trainer.balance_batch=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.update_weights_bucket_megabytes=256 \
  +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +critic.model.override_config.torch_dtype=bfloat16 \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.model.path=$MODEL_PATH \
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
  envs.n_rollouts=4 \
  envs.duplication_mode=none \
  envs.freeze_completed_episodes=True \
  envs.env_name=fastsnake \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  envs.captioner.type=multi_action \
  envs.captioner.max_text_history=0 \
  envs.captioner.max_cot_history=0 \
  envs.fastsnake_kwargs.width=10 \
  envs.fastsnake_kwargs.height=10 \
  envs.fastsnake_kwargs.max_rounds=8 \
  envs.fastsnake_kwargs.num_external_snakes=1 \
  envs.fastsnake_kwargs.num_random_snakes=1 \
  envs.fastsnake_kwargs.death_reward=-1 \
  envs.vec_env_multiprocessing=fork \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  algorithm.rollout_correction.rollout_is=sequence \
  algorithm.rollout_correction.rollout_is_threshold=3.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.rollout_rs_threshold=null \
  algorithm.rollout_correction.rollout_rs_threshold_lower=null \
  algorithm.rollout_correction.rollout_token_veto_threshold=null \
  algorithm.rollout_correction.bypass_mode=false \
  algorithm.rollout_correction.use_policy_gradient=false \
  algorithm.rollout_correction.rollout_is_batch_normalize=false \
  trainer.log_val_generations=1 \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_name} \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.critic_warmup=1 \
  trainer.critic_warmup_micro_batch_size_per_gpu=$micro_batch_size \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.render=False \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=2 \
  trainer.tags=$WANDB_TAGS \
  evaluation=snake_evals \
  prompt=snake \
  prompt.prompt.multi_action_reasoning=True \
  prompt.prompt.epsilon=0.1 2>&1 | tee "$auto_log"

echo "Multi-action L4 test complete! Log: $auto_log"
