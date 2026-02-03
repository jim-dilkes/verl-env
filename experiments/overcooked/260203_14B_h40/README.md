# Overcooked 14B Horizon=40 Experiments

## Overview

Fresh training runs with 14B model using horizon=40 from the start. These experiments use the doubled horizon approach to test whether longer episodes improve learning dynamics with larger models.

## Changes from Original 14B Experiments

### Environment Changes
- **horizon**: 20 → 40 (episodes now twice as long)
- **n_rollouts**: 32 → 16 (halved to maintain consistent compute/memory per iteration)
- **Compute per iteration**: ~640 env steps (same as original: 32×20 = 640, 16×40 = 640)

### Training Configuration
- **total_training_steps**: 1500 (fresh training from step 0)
- **total_epochs**: 1500
- **time limit**: 48h → 96h (4 days) to account for longer episodes

### Experiment Names

1. `OC_PPO_14B_NT_h40_run1`
   - Config: Single-action baseline (naive captioner)
   - Model: Qwen/Qwen3-14B
   - Saves to: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_14B_NT_h40_run1/`

2. `OC_PPO_14B_MA_NT_eps02_h40_run1`
   - Config: Multi-action with epsilon=0.2
   - Model: Qwen/Qwen3-14B
   - Saves to: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_14B_MA_NT_eps02_h40_run1/`

3. `OC_PPO_14B_MA_NT_eps00_h40_run1`
   - Config: Multi-action with epsilon=0.0 (deterministic action selection)
   - Model: Qwen/Qwen3-14B
   - Saves to: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_14B_MA_NT_eps00_h40_run1/`

## Files

- `OC_PPO_14B_NT_h40_1.sbatch` - Single-action baseline
- `OC_PPO_14B_MA_NT_eps02_h40_1.sbatch` - Multi-action with epsilon=0.2
- `OC_PPO_14B_MA_NT_eps00_h40_1.sbatch` - Multi-action with epsilon=0.0

## Expected Behavior

1. **On startup**: Initializes fresh model weights (no checkpoint loading)
2. **Training range**: Trains from step 0 to step 1500 (full training run)
3. **Checkpointing**: Saves checkpoints every 100 steps to `*_h40` directories
4. **Evaluation**: Runs every 100 steps at horizon=40 setting from the start
5. **Time limit**: 96 hours (4 days) to accommodate longer episodes

## Key Config Parameters

```bash
# Environment
envs.n_rollouts=16              # Halved from 32
envs.overcooked_kwargs.horizon=40  # Doubled from 20

# Training
trainer.total_training_steps=1500
trainer.total_epochs=1500
trainer.default_local_dir=.../${experiment_name}_run${run_number}  # New save location

# Checkpointing
trainer.save_freq=100
trainer.test_freq=100
trainer.max_actor_ckpt_to_keep=1
trainer.max_critic_ckpt_to_keep=1

# Compute
--time=96:00:00  # 4 days
--partition=dual_h200,quad_h200
--gpus-per-node=2
```

## Differences from Original 14B Experiments

| Aspect | Original (260115) | This Run (260203) |
|--------|-------------------|-------------------|
| **Horizon** | 20 | 40 |
| **n_rollouts** | 32 | 16 |
| **Training steps** | 1000 | 1500 |
| **Time limit** | 48-60h | 96h (4 days) |
| **Experiment names** | `*_NT_*` | `*_NT_h40_*` |
| **WANDB tags** | No horizon tag | Includes `"horizon:40"` |

## Notes

- **WANDB tags** updated to include `"horizon:40"`
- All other hyperparameters unchanged from originals
- Same model: Qwen/Qwen3-14B
- Same hardware: 2 GPUs (H200), 300GB RAM
- Multi-action configs fixed: removed duplicate captioner.type lines from originals
