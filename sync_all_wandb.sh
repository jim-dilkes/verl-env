#!/bin/bash
source activate verl

# Maximum number of parallel jobs
MAX_PARALLEL=8

# Check for command-line argument for the search directory
if [ -z "$1" ]; then
  SEARCH_DIR="wandb"
  echo "No search directory provided, using default: $SEARCH_DIR"
else
  SEARCH_DIR="$1"
  echo "Using search directory: $SEARCH_DIR"
fi

# Function to sync a single run
sync_run() {
    local run_dir=$1
    echo "Syncing: $run_dir"
    wandb sync "$run_dir"
}

# Find all offline run directories within the specified or default directory
RUNS=$(find "$SEARCH_DIR" -type d -name "offline-run-*")

# Simple loop as fallback
for run in $RUNS; do
    sync_run "$run" &
    
    # Control number of background processes
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
done

# Wait for all background jobs to complete
wait

echo "All syncs completed!"
