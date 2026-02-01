#!/bin/bash
#SBATCH --job-name=diverse_eval_test
#SBATCH --output=logs/diverse-eval-%j.out
#SBATCH --error=logs/diverse-eval-%j.err
#SBATCH --partition=swarm_l4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00

# Parallel Diverse Evaluation Test - L4 Swarm
# Tests the Dipper-style ensemble evaluation on Overcooked

cd /home/jsbd1n24/verl-env

# Create logs directory if needed
mkdir -p logs

echo "===== Diverse Eval Test ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
date
echo "=============================="

# Run the test script
bash experiments/overcooked/test_diverse_eval.sh

echo ""
echo "===== Job Complete ====="
date
