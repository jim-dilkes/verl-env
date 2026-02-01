#!/bin/bash
#SBATCH --job-name=eval_4b_1gpu
#SBATCH --output=logs/eval-4b-1gpu-%j.out
#SBATCH --error=logs/eval-4b-1gpu-%j.err
#SBATCH --partition=scavenger_mathsa100,swarm_a100,a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

cd /home/jsbd1n24/verl-env.worktrees/origin-feat-parallel-diverse-eval
mkdir -p logs

# Add worktree to Python path for verl imports
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "===== Standalone Eval (1 GPU) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
echo "===================================="

# Module loading - match training scripts
module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

nvidia-smi || { echo "FATAL: No GPU!"; exit 1; }

# Conda activation - match training scripts
eval "$(conda shell.bash hook)"
conda activate verl

# Environment setup - match training scripts
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Force single-process mode to avoid subprocess CUDA issues
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY

# CRITICAL: Remove CUDA stubs from LD_LIBRARY_PATH - they break CUDA detection
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/stubs' | tr '\n' ':' | sed 's/:$//')

# Debug: print CUDA paths
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

bash scripts/eval_overcooked_4b.sh ${N_ROLLOUTS:-10}

echo "===== Done ====="
date
