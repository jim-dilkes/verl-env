#!/bin/bash
#SBATCH --job-name=eval_4b_2gpu
#SBATCH --output=logs/eval-4b-2gpu-%j.out
#SBATCH --error=logs/eval-4b-2gpu-%j.err
#SBATCH --partition=swarm_l4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# Standalone 4B Evaluation - 2 GPU Tensor Parallel
# Model split across 2 GPUs = ~4GB per GPU for weights
# Plenty of room for KV cache!

cd /home/jsbd1n24/verl-env.worktrees/origin-feat-parallel-diverse-eval
mkdir -p logs

echo "===== Standalone Eval (2 GPU Tensor Parallel) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
echo "==================================================="

# Environment setup
module load cuda/12.8.0
module load gcc/13.3.0
source ~/.bashrc
conda activate verl

# Disable FlashInfer JIT issues
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Check GPUs
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Run standalone eval with 2 GPU tensor parallel
bash scripts/eval_overcooked_4b_2gpu.sh ${N_ROLLOUTS:-50}

echo ""
echo "===== Job Complete ====="
date
