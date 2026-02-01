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

# Environment setup - ORDER MATTERS!
module load cuda/12.8.0
module load gcc/13.3.0

# Set CUDA env BEFORE conda/python
export CUDA_HOME=/iridisfs/ixsoftware/cuda/12.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify GPU is available BEFORE Python
echo "Checking GPU..."
nvidia-smi || { echo "ERROR: No GPU available!"; exit 1; }

# Now activate conda
source ~/.bashrc
conda activate verl

# Disable problematic JIT compilation
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCH_CUDA_ARCH_LIST="8.9"  # L4 GPU
export TORCHAO_DISABLE=1  # Disable torchao if possible

# Check GPU again from Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run standalone eval
bash scripts/eval_overcooked_4b.sh ${N_ROLLOUTS:-10}

echo ""
echo "===== Job Complete ====="
date
