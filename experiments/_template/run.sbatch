#!/bin/bash
#SBATCH --job-name=aaai_harness
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=300GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=swarm_a100,swarm_h100
#SBATCH --time=5-00:00:00
#SBATCH --gpus-per-node=2
#
# Canonical AAAI entropy/exploration harness. Forkable, env-var driven.
# Launch via launch.sh, or directly:
#   sbatch --export=ALL,ALGO=ppo,ENV=overcooked,INTERVENTION=baseline,SEED=1 experiments/_template/run.sbatch
#
# Required env vars (with defaults): ALGO ENV INTERVENTION SEED MODEL
#   ALGO         : ppo | grpo
#   ENV          : overcooked | snake
#   INTERVENTION : baseline | H005 | H01 | H05 | clip_cov | kl_cov | adaptive | adaptive_decay
#   SEED         : integer (controls envs.group_initial_seed + data.seed + run naming)

set -euo pipefail

# ----------------------------- parameters ------------------------------------
ALGO=${ALGO:-ppo}
ENV=${ENV:-overcooked}
INTERVENTION=${INTERVENTION:-baseline}
SEED=${SEED:-1}
MODEL_ID=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
TOTAL_STEPS=${TOTAL_STEPS:-600}
TEST_FREQ=${TEST_FREQ:-50}
SAVE_FREQ=${SAVE_FREQ:-200}
number_of_gpus=2

project_name=${PROJECT_NAME:-rl_sdm_aaai}
experiment_name=AAAI_${ALGO}_${ENV}_${INTERVENTION}
run_number=${SEED}

job_id=${SLURM_JOB_ID:-local}
user_name=${USER:-$(whoami)}
experiment_run_name=${experiment_name}_seed${SEED}_slu${job_id}

# ----------------------------- intervention switch ---------------------------
# Emits actor-side override flags. clip_cov/kl_cov REPLACE the entropy bonus (entropy_coeff=0).
INTERVENTION_OVERRIDES=()
case "$INTERVENTION" in
  baseline)       INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.001 actor_rollout_ref.actor.policy_loss.loss_mode=vanilla ) ;;
  H005)           INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.005 actor_rollout_ref.actor.policy_loss.loss_mode=vanilla ) ;;
  H01)            INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.01  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla ) ;;
  H05)            INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.05  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla ) ;;
  clip_cov)       INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
                     actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 ) ;;
  kl_cov)         INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
                     actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.0002 ) ;;
  adaptive)       INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.002 actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
                     actor_rollout_ref.actor.entropy_coeff_low=0.0005 actor_rollout_ref.actor.entropy_coeff_high=0.02 actor_rollout_ref.actor.entropy_coeff_lr=0.002 \
                     actor_rollout_ref.actor.entropy_low=0.5 actor_rollout_ref.actor.entropy_high=0.7 actor_rollout_ref.actor.entropy_top_p=0.33 ); USE_FUSED_KERNELS=False ;;
  adaptive_decay) INTERVENTION_OVERRIDES=( actor_rollout_ref.actor.entropy_coeff=0.002 actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
                     actor_rollout_ref.actor.entropy_coeff_low=0.0005 actor_rollout_ref.actor.entropy_coeff_high=0.02 actor_rollout_ref.actor.entropy_coeff_lr=0.002 \
                     actor_rollout_ref.actor.entropy_low=0.5 actor_rollout_ref.actor.entropy_high=0.7 actor_rollout_ref.actor.entropy_top_p=0.33 \
                     actor_rollout_ref.actor.entropy_low_final=0.35 actor_rollout_ref.actor.entropy_high_final=0.55 ); USE_FUSED_KERNELS=False ;;
  *) echo "Unknown INTERVENTION=$INTERVENTION" >&2; exit 1 ;;
esac
# entropy_top_p<1 (adaptive) is incompatible with fused kernels (needs logits); disable there.
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}

# ----------------------------- env switch ------------------------------------
ENV_OVERRIDES=()
case "$ENV" in
  overcooked)
    PROMPT=overcooked
    EVALUATION=overcooked_evals_harness
    WANDB_TAGS='["aaai","overcooked","project:verl-env"]'
    ENV_OVERRIDES=( envs.env_name=overcooked envs.captioner.type=naive envs.captioner.max_text_history=0 envs.captioner.max_cot_history=1 \
      envs.overcooked_kwargs.layout_name=cramped_room envs.overcooked_kwargs.horizon=20 envs.overcooked_kwargs.partner_policy=none \
      envs.overcooked_kwargs.shaped_reward=True envs.overcooked_kwargs.pot_cook_time=3 \
      envs.overcooked_kwargs.print_visualization=false envs.overcooked_kwargs.print_coordinates=true ) ;;
  snake)
    PROMPT=snake_multi_action
    EVALUATION=snake_evals_combined
    WANDB_TAGS='["aaai","snake","project:verl-env"]'
    ENV_OVERRIDES=( envs.env_name=fastsnake envs.captioner.type=multi_action envs.captioner.max_text_history=0 envs.captioner.max_cot_history=1 \
      envs.fastsnake_kwargs.width=10 envs.fastsnake_kwargs.height=10 envs.fastsnake_kwargs.max_rounds=8 \
      envs.fastsnake_kwargs.num_external_snakes=1 envs.fastsnake_kwargs.num_random_snakes=1 envs.fastsnake_kwargs.death_reward=-1 ) ;;
  *) echo "Unknown ENV=$ENV" >&2; exit 1 ;;
esac

# ----------------------------- algo switch -----------------------------------
ALGO_OVERRIDES=()
critic_warmup=5
case "$ALGO" in
  ppo)  ALGO_OVERRIDES=( algorithm.adv_estimator=gae ) ;;
  grpo) ALGO_OVERRIDES=( algorithm.adv_estimator=grpo ); critic_warmup=0 ;;
  *) echo "Unknown ALGO=$ALGO" >&2; exit 1 ;;
esac

# ----------------------------- paths / capture -------------------------------
short_temp_dir=/tmp/ray/${user_name}/${job_id}; mkdir -p "$short_temp_dir"
readable_log_root=/scratch/${user_name}/ray_logs; mkdir -p "$readable_log_root"
export WANDB_DIR="wandb/${experiment_run_name}"; mkdir -p "$WANDB_DIR"
val_dump_dir=/scratch/${user_name}/val_dumps/${experiment_run_name}; mkdir -p "$val_dump_dir"
rollout_dump_dir=/scratch/${user_name}/rollout_dumps/${experiment_run_name}; mkdir -p "$rollout_dump_dir"
ckpt_dir=/scratch/${user_name}/checkpoints/${project_name}/${experiment_name}_seed${SEED}

# ----------------------------- setup -----------------------------------------
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0
eval "$(conda shell.bash hook)"
conda activate verl

export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_OBJECT_STORE_MEMORY=50000000000 RAY_DISABLE_IMPORT_WARNING=1 RAY_DEDUP_LOGS=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=false
export VERL_PPO_LOGGING_LEVEL=INFO
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true

RANDOM_DELAY=$(( ( RANDOM % 90 ) + 1 ))
echo "Sleeping ${RANDOM_DELAY}s to desync Ray init..."; sleep $RANDOM_DELAY

# Resolve local model snapshot
FOLDER_NAME="models--$(echo "$MODEL_ID" | sed 's/\//--/')"
SNAPSHOT_DIR="$HF_HOME/hub/$FOLDER_NAME/snapshots"
if [ -d "$SNAPSHOT_DIR" ]; then
    MODEL_PATH="$(ls -dt "$SNAPSHOT_DIR"/* | head -n 1)"
    echo "Model path: $MODEL_PATH"
else
    echo "Error: no local cache for $MODEL_ID" >&2; exit 1
fi

# ----------------------------- sizing ----------------------------------------
micro_batch_size=16
max_prompt_length=2048
max_response_length=512
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 256))
rollout_max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 2560))
rollout_max_num_seqs=2048

echo "=== RUN: algo=$ALGO env=$ENV intervention=$INTERVENTION seed=$SEED ==="
echo "=== overrides: ${INTERVENTION_OVERRIDES[*]} ==="

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.train_batch_size=256 \
  data.train_files=examples/data/placeholder.parquet \
  data.val_files=examples/data/placeholder.parquet \
  data.shuffle=True \
  data.seed=$SEED \
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
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  "${INTERVENTION_OVERRIDES[@]}" \
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
  actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +critic.model.override_config.torch_dtype=bfloat16 \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.model.path=$MODEL_PATH \
  critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
  critic.forward_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
  "${ALGO_OVERRIDES[@]}" \
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
  envs.group_initial_seed=$SEED \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  "${ENV_OVERRIDES[@]}" \
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
  trainer.log_val_generations=20 \
  trainer.validation_data_dir=$val_dump_dir \
  trainer.rollout_data_dir=$rollout_dump_dir \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_run_name} \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.critic_warmup=$critic_warmup \
  trainer.critic_warmup_micro_batch_size_per_gpu=16 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.render=False \
  trainer.total_epochs=10000 \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.default_local_dir=$ckpt_dir \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.tags="$WANDB_TAGS" \
  ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-6} \
  ray_kwargs.ray_init.num_gpus=$number_of_gpus \
  evaluation=$EVALUATION \
  prompt=$PROMPT 2>&1 | tee verl_demo.log

actual_session_dir="${short_temp_dir}/session_latest"
readable_log_path="${readable_log_root}/${experiment_run_name}"
[ -d "$actual_session_dir" ] && ln -sfn "$actual_session_dir" "$readable_log_path" && echo "Ray logs: $readable_log_path"
