#!/bin/bash
# Script to try installing an older flash-attn version that might work with GLIBC 2.28
# This is faster than compiling from source

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl_latest

# Load CUDA module
module load cuda/13.0.0
module load gcc/13.3.0


echo "=== System Information ==="
echo "GLIBC version: $(ldd --version | head -1)"
echo "Python version: $(python3 --version)"
PYTORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)')
echo "PyTorch version: $PYTORCH_VERSION"
echo "CUDA version: $CUDA_VERSION"
echo ""

# Determine Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
echo "Python version for wheel: cp${PYTHON_VERSION}"

# Uninstall existing flash-attn
echo "=== Uninstalling existing flash-attn ==="
pip uninstall -y flash-attn flash_attn 2>/dev/null || true

# Try version 2.7.4.post1 first (reported to work with older GLIBC)
echo "=== Attempting to install flash-attn 2.7.4.post1 ==="
echo "Trying pre-built wheels compatible with PyTorch 2.8 and CUDA 12.8..."

# Try different wheel variants
WHEEL_VARIANTS=(
    "flash_attn-2.7.4.post1+cu12torch2.8cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
    "flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
)

FLASH_ATTN_REPO="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1"

SUCCESS=false
for wheel in "${WHEEL_VARIANTS[@]}"; do
    echo ""
    echo "Trying: $wheel"
    if wget -q --spider "${FLASH_ATTN_REPO}/${wheel}"; then
        echo "Found wheel, downloading..."
        wget -nv "${FLASH_ATTN_REPO}/${wheel}" -O /tmp/${wheel}
        echo "Installing..."
        pip install /tmp/${wheel} --no-cache-dir
        rm /tmp/${wheel}
        
        # Test import
        echo "Testing import..."
        if python3 -c "import flash_attn; print(f'Success! flash_attn version: {flash_attn.__version__}')" 2>/dev/null; then
            SUCCESS=true
            break
        else
            echo "Import failed, trying next variant..."
            pip uninstall -y flash-attn flash_attn 2>/dev/null || true
        fi
    else
        echo "Wheel not available, trying next variant..."
    fi
done

if [ "$SUCCESS" = false ]; then
    echo ""
    echo "=== Pre-built wheels failed ==="
    echo "No compatible pre-built wheel found for your system."
    echo "You have two options:"
    echo "1. Compile from source: bash compile_flash_attn.sh"
    echo "2. Continue without flash-attn optimizations (current setup)"
    exit 1
fi

echo ""
echo "=== SUCCESS ==="
echo "flash-attn 2.7.4.post1 installed successfully!"
echo "You can now try enabling use_remove_padding=True in your verl scripts."
echo ""
echo "Note: If you still get GLIBC errors, you'll need to compile from source:"
echo "  bash compile_flash_attn.sh"

