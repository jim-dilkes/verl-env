#!/bin/bash
#SBATCH --job-name=fa
#SBATCH --output=fa_%j.out
#SBATCH --error=fa_%j.err
#SBATCH --time=04:00:00          # 2 hours should be enough
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Use 8 CPUs for compilation
#SBATCH --mem=128G               # Request 128GB memory (safe for compiling 2 GPU architectures)
#                                 # Flash-attn compilation is memory-intensive, especially with
#                                 # multiple architectures (sm_80 + sm_90). 128GB provides buffer.
#SBATCH --gres=gpu:0             # No GPU needed - won't count against GPU quotas
#                                 # Note: On GPU-only clusters, this still runs on GPU nodes
#                                 # but doesn't allocate/use GPU resources
#SBATCH --partition=a100,swarm_a100,swarm_h100         # Prefer A100 nodes (more available, sufficient for CPU work)
#                                 # Can also use swarm_a100 or swarm_h100 if needed

# Print job info
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Run the compilation script
bash /home/jsbd1n24/verlog_2/compile_flash_attn.sh

# Print completion info
echo ""
echo "=== Job Completed ==="
echo "End time: $(date)"
echo "Check output files: compile_flash_attn_${SLURM_JOB_ID}.out and compile_flash_attn_${SLURM_JOB_ID}.err"

