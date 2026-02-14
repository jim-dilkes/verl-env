#!/bin/bash
# DIME login node smoke test — validates DIME feature before full cluster runs
# Usage: bash experiments/overcooked/test_dime_login_node.sh
# Runs: 1 critic warmup step, 3 training steps with DIME enabled + masking

set -e

project_name=verl_env
experiment_name=test_dime_login_node
run_number=1
number_of_gpus=1

export WANDB_MODE=offline
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_OBJECT_STORE_MEMORY=10000000000
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export VERL_PPO_LOGGING_LEVEL=DEBUG

MODEL_ID=Qwen/Qwen3-0.6B-Base
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

# --- Setup ---
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

eval "$(conda shell.bash hook)"
conda activate verl

# Tuned for 24GB L4 GPUs (login nodes)
micro_batch_size=2
max_prompt_length=384
max_response_length=96
max_token_len_per_gpu=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$((max_token_len_per_gpu + 64))
rollout_max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 512))
rollout_max_num_seqs=128

ray_num_cpus=${SLURM_CPUS_PER_TASK:-4}

if ! [[ "$ray_num_cpus" =~ ^[0-9]+$ ]]; then
    echo "Error: ray_num_cpus must be an integer, got '$ray_num_cpus' (SLURM_CPUS_PER_TASK='$SLURM_CPUS_PER_TASK')" >&2
    exit 1
fi
echo "Using ray_num_cpus=$ray_num_cpus"

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
  envs.env_name=overcooked \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.captioner.max_cot_history=1 \
  envs.overcooked_kwargs.layout_name=cramped_room \
  envs.overcooked_kwargs.horizon=20 \
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
  prompt.prompt.dime.mask_for_training=true \
  prompt.prompt.dime.no_supplement_prob=0.125 \
  trainer.log_val_generations=1 \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_name} \
  trainer.val_before_train=False \
  trainer.critic_warmup=1 \
  trainer.critic_warmup_micro_batch_size_per_gpu=$micro_batch_size \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=2 \
  trainer.render=False \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=2 \
  evaluation=overcooked_evals_minimal \
  prompt=overcooked 2>&1 | tee test_dime_login_node.log

echo "DIME test run complete!"
