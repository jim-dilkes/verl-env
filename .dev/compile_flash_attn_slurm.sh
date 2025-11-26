#!/bin/bash
#SBATCH --job-name=fa
#SBATCH --output=fa_%j.out
#SBATCH --error=fa_%j.err
#SBATCH --time=05:00:00          # builds can take several hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # more CPU parallelism for compile
#SBATCH --mem=256G               # high memory usage when building multiple archs
#                                 # Flash-attn compilation is memory-intensive, especially with
#                                 # multiple architectures (sm_80 + sm_90). 128GB provides buffer.
#SBATCH --gres=gpu:0             # No GPU needed
#SBATCH --partition=a100         # compile on CPU-only resources

# Print job info
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Run the compilation script
bash /home/jsbd1n24/verl-env/.dev/compile_flash_attn.sh

# Print completion info
echo ""
echo "=== Job Completed ==="
echo "End time: $(date)"
echo "Check output files: compile_flash_attn_${SLURM_JOB_ID}.out and compile_flash_attn_${SLURM_JOB_ID}.err"

