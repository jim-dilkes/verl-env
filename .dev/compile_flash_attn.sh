#!/bin/bash
# Script to compile flash-attn from source for GLIBC 2.28 compatibility
# This will build flash-attn with your system's GLIBC version

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl

# Load CUDA module (must match PyTorch's CUDA version)
module load cuda/12.8.0
module load gcc/13.3.0

# Check versions
echo "=== System Information ==="
echo "GLIBC version: $(ldd --version | head -1)"
echo "Python version: $(python3 --version)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "nvcc: $(which nvcc)"
echo "Available memory: $(free -h | grep Mem | awk '{print $7}')"
echo ""

# Uninstall existing flash-attn
echo "=== Uninstalling existing flash-attn ==="
pip uninstall -y flash-attn flash_attn 2>/dev/null || true

# Use pre-downloaded flash-attention repository
echo "=== Using pre-downloaded flash-attention repository ==="
SOURCE_DIR="$HOME/flash_attn_source/flash-attention"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Flash-attention source not found at $SOURCE_DIR"
    echo "Please run download_flash_attn_source.sh on the login node first!"
    exit 1
fi

# Copy to a temporary directory for building (to avoid modifying the source)
FLASH_ATTN_DIR="/tmp/flash-attention-$(date +%s)"
echo "Copying source to: $FLASH_ATTN_DIR"
# Use cp -a to preserve git metadata and submodules
cp -a "$SOURCE_DIR" "$FLASH_ATTN_DIR"
cd "$FLASH_ATTN_DIR"

echo "Using flash-attention version: $(git describe --tags 2>/dev/null || echo 'unknown')"

# Verify submodules are present (they should be from the copy)
if [ ! -d "csrc/composable_kernel" ] || [ -z "$(ls -A csrc/composable_kernel 2>/dev/null)" ]; then
    echo "ERROR: Submodule csrc/composable_kernel not found or empty!"
    echo "Please run download_flash_attn_source.sh on the login node to initialize submodules."
    exit 1
fi

# Patch setup.py to skip submodule updates (since we don't have internet on compute nodes)
echo "Patching setup.py to skip submodule updates..."
if grep -q "git submodule update" setup.py 2>/dev/null; then
    # Comment out git submodule update calls
    sed -i 's/subprocess.run(\["git", "submodule", "update"/# subprocess.run(["git", "submodule", "update"/g' setup.py
    sed -i 's/subprocess.run(\["git", "submodule"/# subprocess.run(["git", "submodule"/g' setup.py
    echo "Patched setup.py to skip submodule updates"
else
    echo "setup.py doesn't seem to have submodule update calls (might be in pyproject.toml)"
fi

# Install from source
echo "=== Compiling flash-attn from source ==="
echo "This may take 20-40 minutes depending on your system..."
echo "Using reduced parallelism to avoid memory issues..."

# Reduce MAX_JOBS to avoid OOM kills (flash-attn compilation is very memory-intensive)
# Compile sequentially to avoid memory pressure
export MAX_JOBS=${MAX_JOBS:-4}

# Set environment variables to reduce memory usage during compilation
# Compile for multiple GPU architectures:
# - sm_80 (A100, L4 sm_89 is backward compatible)
# - sm_90 (H100, H200)
# Note: Compiling for multiple architectures increases memory usage, so we keep MAX_JOBS low
export TORCH_CUDA_ARCH_LIST="8.0;9.0"  # A100/L4 and H100/H200
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0

# Additional memory-saving flags
export CMAKE_BUILD_PARALLEL_LEVEL=2

echo "Compiling for GPU architectures: sm_80 (A100/L4) and sm_90 (H100/H200)"
echo "This will use more memory but ensures compatibility across all your GPUs"

# Build wheel first (so we can reuse it later)
echo "Building wheel..."
WHEEL_DIR="$HOME/flash_attn_wheels"
mkdir -p "$WHEEL_DIR"

# Disable pip's index lookup (compute nodes have no internet)
# Use --no-deps because we already have torch installed
# --no-build-isolation uses the current environment's build tools
pip wheel . --no-build-isolation --no-deps --no-index --find-links /dev/null --wheel-dir="$WHEEL_DIR" -v 2>&1 | tee /tmp/flash_attn_build.log

# Find the built wheel
WHEEL_FILE=$(ls -t "$WHEEL_DIR"/flash_attn*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: Wheel file not found!"
    exit 1
fi

echo ""
echo "=== Wheel built successfully ==="
echo "Wheel location: $WHEEL_FILE"
echo "You can reuse this wheel with: pip install $WHEEL_FILE"
echo ""

# Install from the wheel
echo "Installing from wheel..."
pip install "$WHEEL_FILE" --no-deps --force-reinstall

echo ""
echo "=== Installation complete! ==="
echo "Testing import..."
python3 -c "import flash_attn; print(f'Successfully imported flash_attn version: {flash_attn.__version__}')" || {
    echo "ERROR: Import failed. Check the error messages above."
    exit 1
}

echo ""
echo "=== Cleanup ==="
cd ~
# Keep the source directory for now in case we need to debug
# rm -rf "$FLASH_ATTN_DIR"
echo "Source directory kept at: $FLASH_ATTN_DIR"
echo "Wheel saved at: $WHEEL_FILE"
echo "To clean up source later: rm -rf $FLASH_ATTN_DIR"

echo ""
echo "=== SUCCESS ==="
echo "flash-attn has been compiled and installed with GLIBC 2.28 compatibility!"
echo "You can now use use_remove_padding=True and flash_attention_2 in your verl scripts."

