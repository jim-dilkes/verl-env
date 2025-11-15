# verl Installation Guide for SLURM Cluster

This guide adapts the verl installation instructions for a SLURM cluster environment.

## Key Differences from Standard Installation

1. **CUDA**: Use `module load cuda/13.0.0` instead of installing from .deb packages
2. **cuDNN**: May come with CUDA module or install via conda/pip (nvidia-cudnn-cu12)
3. **No apt-get**: Cannot use apt-get commands on SLURM cluster
4. **Module system**: Use module load for system libraries

## Installation Steps

### 1. Prerequisites

#### CUDA (via module load)
```bash
module load cuda/13.0.0
```

#### cuDNN
cuDNN will be installed via pip/conda as part of dependencies. If needed for Megatron:
```bash
pip install nvidia-cudnn-cu12
```

### 2. Setup Conda Environment

```bash
conda activate verl_latest
# Python 3.12 already installed
```

### 3. Load CUDA Module

```bash
module load cuda/13.0.0
export CUDA_HOME=$CUDA_HOME  # Set by module
```

### 4. Install vLLM and Dependencies

**Option A: Use verl's install script (recommended)**
```bash
git clone https://github.com/volcengine/verl.git
cd verl
# For FSDP backend (no Megatron)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

**Option B: Manual installation**
If script fails, install manually:
```bash
# Install PyTorch with CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install vLLM 11.0
pip install vllm==11.0

# Install SGLang (if needed)
pip install sglang[all]
```

### 5. Install verl from Source

```bash
cd verl
pip install --no-deps -e .
```

### 6. Install Missing Dependencies

```bash
# Install verl dependencies (without overriding PyTorch/vLLM)
pip install -r requirements.txt  # If verl has one
# Or install common dependencies:
pip install pyarrow tensordict transformers accelerate
```

### 7. Optional: NVIDIA Apex (for Megatron backend)

Only needed if using Megatron-LM backend:
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
MAX_JOBS=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./
```

### 8. Verify Installation

```bash
python -c "import verl; print('verl installed successfully')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## Post-Installation Checklist

Verify these packages are correct versions:
- ✅ torch and torch series
- ✅ vLLM (should be 11.0)
- ✅ SGLang (if installed)
- ✅ pyarrow
- ✅ tensordict
- ✅ nvidia-cudnn-cu12 (for Megatron backend)

## SLURM Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=verl_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load cuda/13.0.0
conda activate verl_latest

# Your verl commands here
python your_verl_script.py
```

