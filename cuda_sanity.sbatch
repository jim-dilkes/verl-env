#!/bin/bash
#SBATCH --job-name=cuda_check
#SBATCH --output=logs/cuda-sanity-%j.out
#SBATCH --error=logs/cuda-sanity-%j.err
#SBATCH --partition=swarm_a100
#SBATCH --nodes=1
#SBATCH --nodelist=swarma1004
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

module load conda/python3
module load cuda/12.8.0
module load gcc/13.3.0

cd /home/jsbd1n24/verl-env
eval "$(conda shell.bash hook)"
conda activate verlog_new

echo "===== CUDA Sanity Check ====="
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi && nvidia-smi || echo "nvidia-smi not available"
echo "======================================="

python - <<'PY'
import os
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"[GPU {idx}] {props.name}, total_mem={props.total_memory/1e9:.2f} GB")
else:
    print("ERROR: CUDA unavailable.")
    raise SystemExit(1)
PY

