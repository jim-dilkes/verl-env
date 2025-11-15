# verl Installation Summary

## Installation Date
November 14, 2025

## Environment Setup
- **Conda Environment**: `verl_latest`
- **Python Version**: 3.12.12
- **CUDA Module**: `cuda/13.0.0` (via `module load`)
- **Installation Directory**: `~/verlog_2/verl`

## Installed Packages

### Core Frameworks
- **verl**: 0.7.0.dev (installed from source in editable mode)
- **vLLM**: 0.11.0
- **SGLang**: 0.5.2
- **PyTorch**: 2.8.0+cu128
  - CUDA available: True
  - CUDA version: 12.8

### Key Dependencies
- **numpy**: 1.26.4 (< 2.0.0 as required by verl)
- **pyarrow**: 22.0.0 (>= 15.0.0 as required)
- **tensordict**: 0.10.0 (>= 0.8.0, <= 0.10.0, != 0.9.0 as required)

## Installation Method
1. Created conda environment with Python 3.12
2. Loaded CUDA 13.0.0 module
3. Ran verl's installation script: `USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh`
4. Fixed numpy version conflict (downgraded to 1.26.4)
5. Installed verl from source: `pip install --no-deps -e .`

## Notes
- **Editable Installation**: verl is installed in editable mode, so code changes in `~/verlog_2/verl` will be immediately available
- **CUDA**: Using system CUDA 13.0.0 via module load (no Docker)
- **cuDNN**: Installed via pip package `nvidia-cudnn-cu12` (not available as module)
- **Warnings**: Some harmless warnings about distutils and pynvml deprecation (non-critical)

## Usage
To use verl in SLURM jobs:
```bash
#!/bin/bash
#SBATCH --job-name=verl_job
#SBATCH --gres=gpu:1

module load cuda/13.0.0
conda activate verl_latest
cd ~/verlog_2/verl

# Your verl commands here
```

## Verification Commands
```bash
conda activate verl_latest
module load cuda/13.0.0
python -c "import verl; print('verl:', verl.__version__)"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

