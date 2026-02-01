#!/bin/bash
#SBATCH --job-name=diverse_eval_4b
#SBATCH --output=logs/diverse-eval-4b-%j.out
#SBATCH --error=logs/diverse-eval-4b-%j.err
#SBATCH --partition=swarm_l4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00

# Parallel Diverse Evaluation Test - 4B Model on L4
# Tight memory config for 24GB GPU

cd /home/jsbd1n24/verl-env

mkdir -p logs

echo "===== Diverse Eval Test (4B) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
date
echo "=================================="

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.free --format=csv

# Run the test script
bash experiments/overcooked/test_diverse_eval_4b.sh

echo ""
echo "===== Job Complete ====="
date
