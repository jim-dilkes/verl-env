#!/bin/bash
# Script to download flash-attention source code on login node
# Run this BEFORE submitting the SLURM job

set -e

echo "=== Downloading flash-attention source code ==="
echo "This should be run on the login node (which has internet access)"
echo ""

# Create directory for flash-attention source
SOURCE_DIR="$HOME/flash_attn_source"
mkdir -p "$SOURCE_DIR"

cd "$SOURCE_DIR"

# Check if we already have the repository
if [ -d "flash-attention" ]; then
    echo "Repository already exists, updating..."
    cd flash-attention
    git fetch --all
    git checkout v2.7.4.post1 2>/dev/null || git checkout v2.7.4.post1
    echo "Initializing/updating submodules..."
    git submodule update --init --recursive
else
    echo "Cloning flash-attention repository with submodules..."
    git clone --recursive https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    echo "Checking out version 2.7.4.post1..."
    git checkout v2.7.4.post1
    echo "Ensuring submodules are initialized..."
    git submodule update --init --recursive
fi

echo ""
echo "=== Source code ready ==="
echo "Location: $SOURCE_DIR/flash-attention"
echo "Version: $(git describe --tags)"
echo ""
echo "You can now submit the SLURM job: sbatch compile_flash_attn_slurm.sh"

