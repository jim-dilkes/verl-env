#!/bin/bash
# Launcher for the AAAI entropy/exploration harness.
#
# Usage:
#   ./launch.sh <ALGO> <ENV> <INTERVENTION> <SEED_START> <SEED_END>
# Examples:
#   ./launch.sh ppo overcooked baseline 1 1      # single n=1 run
#   ./launch.sh ppo overcooked baseline 1 5      # seeds 1..5
#
# INTERVENTION in: baseline H005 H01 H05 clip_cov kl_cov adaptive adaptive_decay
# ENV in: overcooked snake ; ALGO in: ppo grpo
set -euo pipefail

ALGO=${1:?algo}; ENV=${2:?env}; INTERVENTION=${3:?intervention}
SEED_START=${4:-1}; SEED_END=${5:-$SEED_START}
HERE="$(cd "$(dirname "$0")" && pwd)"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
  echo "Submitting ALGO=$ALGO ENV=$ENV INTERVENTION=$INTERVENTION SEED=$SEED"
  sbatch --job-name="AAAI_${ALGO}_${ENV}_${INTERVENTION}_s${SEED}" \
    --export=ALL,ALGO="$ALGO",ENV="$ENV",INTERVENTION="$INTERVENTION",SEED="$SEED" \
    "$HERE/run.sbatch"
done
