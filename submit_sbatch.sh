#!/bin/bash

# Script to submit all .sbatch files in a given directory
# Usage: ./submit_sbatch.sh <directory_path> [pause_seconds]

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path> [pause_seconds]"
    echo "Example: $0 /path/to/sbatch/files"
    echo "Example: $0 /path/to/sbatch/files 5"
    echo ""
    echo "Arguments:"
    echo "  directory_path  - Path to directory containing .sbatch files"
    echo "  pause_seconds   - Optional pause in seconds between submissions (default: 0)"
    exit 1
fi

# Get the directory path from the first argument
SBATCH_DIR="$1"

# Get the pause duration from the second argument (default to 0 if not provided)
PAUSE_SECONDS="${2:-0}"

# Validate pause_seconds is a non-negative integer
if ! [[ "$PAUSE_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Error: pause_seconds must be a non-negative integer, got: '$PAUSE_SECONDS'"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$SBATCH_DIR" ]; then
    echo "Error: Directory '$SBATCH_DIR' does not exist."
    exit 1
fi

# Check if the directory contains any .sbatch files
SBATCH_COUNT=$(find "$SBATCH_DIR" -maxdepth 1 -name "*.sbatch" | wc -l)

if [ $SBATCH_COUNT -eq 0 ]; then
    echo "No .sbatch files found in directory '$SBATCH_DIR'"
    exit 1
fi

echo "Found $SBATCH_COUNT .sbatch file(s) in '$SBATCH_DIR'"
if [ "$PAUSE_SECONDS" -gt 0 ]; then
    echo "Pause between submissions: $PAUSE_SECONDS seconds"
fi
echo "Submitting jobs..."

# Collect all .sbatch files into an array
mapfile -t sbatch_files < <(find "$SBATCH_DIR" -maxdepth 1 -name "*.sbatch" -type f | sort)

# Submit each .sbatch file
file_count=${#sbatch_files[@]}
for ((i=0; i<file_count; i++)); do
    sbatch_file="${sbatch_files[$i]}"
    if [ -f "$sbatch_file" ]; then
        echo "Submitting: $(basename "$sbatch_file")"
        sbatch "$sbatch_file"
        
        # Check if sbatch command was successful
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully submitted $(basename "$sbatch_file")"
        else
            echo "  ✗ Failed to submit $(basename "$sbatch_file")"
        fi
        
        # Add pause between submissions (except after the last file)
        if [ "$PAUSE_SECONDS" -gt 0 ] && [ $i -lt $((file_count - 1)) ]; then
            echo "  Pausing for $PAUSE_SECONDS seconds..."
            sleep "$PAUSE_SECONDS"
        fi
        echo ""
    fi
done

echo "Job submission complete!"
