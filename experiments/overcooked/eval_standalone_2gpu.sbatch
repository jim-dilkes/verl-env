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

cd /home/jsbd1n24/verl-env.worktrees/origin-feat-parallel-diverse-eval
mkdir -p logs

echo "===== Standalone Eval (2 GPU) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
echo "===================================="

module load cuda/12.8.0
module load gcc/13.3.0

nvidia-smi || { echo "FATAL: No GPU!"; exit 1; }

source ~/.bashrc
conda activate verl

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

bash scripts/eval_overcooked_4b_2gpu.sh ${N_ROLLOUTS:-50}

echo "===== Done ====="
date
