#!/bin/bash
#SBATCH --job-name=B_TREAT_GEN_p80_r32_101
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=370GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=swarm_h100,swarm_a100
#SBATCH --time=96:00:00
#SBATCH --gpus-per-node=2

# DIME Treatment-Generic: 20% supplement dose (no_supp=0.8), split evals

# --- Define experiment variables ---
project_name=rl_sdm
experiment_name=OC_4B_DIME_TREAT_GEN_p80_r32
run_number=101
number_of_gpus=2
WANDB_TAGS='["overcooked","project:verl-env","vllm:0.9.1","singl-act","horizon:40","dime-treatment-generic","prob-mask-0.5","supp-0.2","n-rollouts-32","split-eval"]'

job_id=${SLURM_JOB_ID:-local}
run_id=${experiment_name}_run${run_number}_slu${job_id}
user_name=${USER:-$(whoami)}
short_temp_dir=/tmp/ray/${user_name}/${job_id}
mkdir -p "$short_temp_dir"
readable_log_root=/scratch/${user_name}/ray_logs
mkdir -p "$readable_log_root"

export RAY_TMPDIR="$short_temp_dir"
export TMPDIR="$short_temp_dir"



experiment_run_name=${experiment_name}_run${run_number}_slu${job_id}
export WANDB_DIR="wandb/${experiment_run_name}"
mkdir -p "$WANDB_DIR"
profile_dump_dir=/scratch/${user_name}/profile_dumps/${experiment_run_name}

# --- Setup ---
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

eval "$(conda shell.bash hook)"
conda activate verl

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

RANDOM_DELAY=$(( ( RANDOM % 120 ) + 1 ))
echo "Sleeping for $RANDOM_DELAY seconds to avoid simultaneous Ray cluster startup..."
sleep $RANDOM_DELAY

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES


# --- Configure Environment for Ray ---

export RAY_OBJECT_STORE_MEMORY=30000000000
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=false
export VERL_PPO_LOGGING_LEVEL=DEBUG

MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
FOLDER_NAME="models--$(echo "$MODEL_ID" | sed 's/\//--/')"
SNAPSHOT_DIR="$HF_HOME/hub/$FOLDER_NAME/snapshots"

echo SNAPSHOT_DIR: $SNAPSHOT_DIR

if [ -d "$SNAPSHOT_DIR" ]; then
    LATEST_SNAPSHOT=$(ls -dt $SNAPSHOT_DIR/* | head -n 1)
    MODEL_PATH="$LATEST_SNAPSHOT"
    echo "Auto-resolved model path to: $MODEL_PATH"
else
    echo "Error: Could not find local cache for $MODEL_ID"
    exit 1
fi
micro_batch_size=16

max_prompt_length=2048
max_response_length=512
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 256))

rollout_max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 2560))
rollout_max_num_seqs=$((2048))

set +e
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.38 \
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
  envs.n_rollouts=32 \
  envs.duplication_mode=none \
  envs.freeze_completed_episodes=True \
  envs.env_name=overcooked \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.captioner.max_cot_history=1 \
  envs.overcooked_kwargs.layout_name=cramped_room \
  envs.overcooked_kwargs.horizon=40 \
  envs.overcooked_kwargs.partner_policy=none \
  envs.overcooked_kwargs.shaped_reward=True \
  envs.overcooked_kwargs.pot_cook_time=3 \
  envs.overcooked_kwargs.print_visualization=false \
  envs.overcooked_kwargs.print_coordinates=true \
  envs.overcooked_kwargs.random_agent_positions=true \
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
  prompt.prompt.dime.enabled=true \
  prompt.prompt.dime.source=generic \
  prompt.prompt.dime.mask_for_training=true \
  prompt.prompt.dime.mask_probability=0.5 \
  prompt.prompt.dime.no_supplement_prob=0.8 \
  trainer.log_val_generations=1 \
  trainer.project_name=$project_name \
  trainer.experiment_name=OC_4B_DIME_TREAT_GEN_p80_r32 \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.critic_warmup=5 \
  trainer.critic_warmup_micro_batch_size_per_gpu=16 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.render=False \
  trainer.total_epochs=1300 \
  trainer.total_training_steps=1300 \
  trainer.default_local_dir=/scratch/${user_name}/checkpoints/${project_name}/${experiment_name}_run${run_number} \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.tags=$WANDB_TAGS \
  ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-32} \
  +ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  evaluation=overcooked_evals_dime_split \
  prompt=overcooked 2>&1 | tee verl_demo.log
python_status=${PIPESTATUS[0]}
set -e


actual_session_dir="${short_temp_dir}/ray/session_latest"
readable_log_path="${readable_log_root}/${experiment_run_name}"
if [ -d "$actual_session_dir" ]; then
    mkdir -p "$readable_log_path"
    cp -a "$actual_session_dir/logs" "$readable_log_path/" || true
    echo "Ray logs copied to: $readable_log_path/logs"
fi

if [ -n "$RAY_HEAD_PID" ]; then
    echo "Stopping Ray cluster (PID: $RAY_HEAD_PID)..."
    ray stop
    kill $RAY_HEAD_PID 2>/dev/null || true
    echo "Ray cluster stopped."
fi

exit $python_status
