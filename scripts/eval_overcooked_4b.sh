#!/bin/bash
# One-liner eval for Overcooked with 4B model
# Usage: bash scripts/eval_overcooked_4b.sh [n_rollouts]

MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-4B-Instruct-2507"}
N_ROLLOUTS=${1:-10}

# Find cached model
FOLDER_NAME="models--$(echo "$MODEL_ID" | sed 's/\//--/')"
MODEL_PATH=$(ls -d $HF_HOME/hub/$FOLDER_NAME/snapshots/* 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "Model not found. Downloading..."
    huggingface-cli download $MODEL_ID
    MODEL_PATH=$(ls -d $HF_HOME/hub/$FOLDER_NAME/snapshots/* | head -1)
fi

python scripts/env_eval_standalone.py \
    --model "$MODEL_PATH" \
    --env overcooked \
    --layout cramped_room \
    --n_rollouts $N_ROLLOUTS \
    --gpu_memory 0.80
