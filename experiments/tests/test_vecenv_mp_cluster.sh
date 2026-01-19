#!/bin/bash
# Test VecEnv multiprocessing methods on cluster
# Run on login node (no SLURM needed for this test)
#
# Usage:
#   bash experiments/tests/test_vecenv_mp_cluster.sh
#   bash experiments/tests/test_vecenv_mp_cluster.sh --n-workers 50

set -e

# Default values
N_WORKERS=20

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        *)
            N_WORKERS="$1"
            shift
            ;;
    esac
done

echo "=========================================="
echo "VecEnv Multiprocessing Cluster Test"
echo "=========================================="
echo "Workers: $N_WORKERS"
echo ""

# Setup environment
module load conda/python3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate verl

# Run the comparison test
echo "Running multiprocessing comparison..."
echo ""

PYTHONUNBUFFERED=1 python scripts/test_vecenv_multiprocessing.py \
    --n-workers "$N_WORKERS" \
    --envs fastsnake \
    --methods spawn fork forkserver \
    --n-trials 2

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "Key findings to look for:"
echo "  - If fork is much faster: NFS contention is the bottleneck"
echo "  - If spawn/forkserver are similar: bottleneck is elsewhere"
echo "  - Expected: fork >> forkserver > spawn on NFS"
echo ""
echo "To enable fork for all training runs, add to your config:"
echo "  envs.vec_env_multiprocessing=fork"
