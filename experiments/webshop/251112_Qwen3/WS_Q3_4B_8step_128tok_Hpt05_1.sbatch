#!/bin/bash
#SBATCH --job-name=WS_Q3_4B_8step_Hpt05_1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=460GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=swarm_h100,swarm_a100,a100
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=2

# --- Setup ---
module load conda/python3
module load cuda/12.4.0
module load gcc/13.3.0

# jdk required for WebShop environment (pyserini)
module load jdk/22.0.1 

cd Verlog
eval "$(conda shell.bash hook)"
conda activate verlog_new

run_number=1
number_of_steps=1000
number_of_gpus=2

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

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

export VERL_PPO_LOGGING_LEVEL=DEBUG

# --- Define model and batch size ---
model=Qwen/Qwen3-4B-Instruct-2507
# model=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
micro_batch_size=8

# --- Define experiment variables ---
project_name=webshop
experiment_name=8step_100k

# --- Run your main application ---
PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=1500 \
  data.max_response_length=128 \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$number_of_gpus \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.actor.entropy_coeff=0.05 \
  critic.model.path=$model \
  critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  critic.optim.lr=5e-5 \
  critic.highlight_first=True \
  critic.highlight_ratio=3.0 \
  algorithm.step_gamma=0.99 \
  algorithm.step_lam=0.95 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  envs.env_name=webshop \
  envs.n_rollouts=32 \
  envs.webshop_kwargs.max_steps=8 \
  envs.webshop_kwargs.num_products=10_000 \
  envs.webshop_kwargs.goals_seed=${run_number} \
  envs.duplication_mode=none \
  envs.freeze_completed_episodes=True \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.vec_env_multiprocessing=spawn \
  trainer.project_name=$project_name \
  trainer.experiment_name=${experiment_name}_run${run_number}_slu${SLURM_JOB_ID} \
  trainer.group=${experiment_name} \
  trainer.val_before_train=True \
  trainer.critic_warmup=20 \
  trainer.critic_warmup_step=5 \
  critic.critic_warmup_micro_batch_size_per_gpu=64 \
  trainer.n_gpus_per_node=$number_of_gpus \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.render=False \
  trainer.total_epochs=$number_of_steps \
  trainer.default_local_dir=/scratch/jsbd1n24/checkpoints/${project_name}/${experiment_name}_run${run_number} \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  evaluation=webshop_evals 2>&1 | tee verl_demo.log

# --- Cleanup Ray Cluster ---
if [ -n "$RAY_HEAD_PID" ]; then
    echo "Stopping Ray cluster (PID: $RAY_HEAD_PID)..."
    ray stop
    kill $RAY_HEAD_PID 2>/dev/null || true
    echo "Ray cluster stopped."
fi
