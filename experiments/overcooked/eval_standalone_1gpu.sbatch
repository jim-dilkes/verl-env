#!/bin/bash
#SBATCH --job-name=eval_4b_1gpu
#SBATCH --output=logs/eval-4b-1gpu-%j.out
#SBATCH --error=logs/eval-4b-1gpu-%j.err
#SBATCH --partition=swarm_l4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Standalone 4B Evaluation - Single GPU
# Bypasses PPO trainer, uses vLLM directly

cd /home/jsbd1n24/verl-env.worktrees/origin-feat-parallel-diverse-eval
mkdir -p logs

echo "===== Standalone Eval (1 GPU) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
echo "==================================="

# Environment setup
module load cuda/12.8.0
module load gcc/13.3.0
source ~/.bashrc
conda activate verl

# Disable FlashInfer JIT issues
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run standalone eval
bash scripts/eval_overcooked_4b.sh ${N_ROLLOUTS:-10}

echo ""
echo "===== Job Complete ====="
date
