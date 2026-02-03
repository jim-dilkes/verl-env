# Overcooked Horizon=40 Experiments

## Overview

Two sets of experiments with horizon=40 (double the original horizon=20):
1. **Continuation runs** (`cont_h40`): Resume from existing h=20 checkpoints
2. **Fresh runs** (`h40`): Train from scratch with h=40 from start

## Changes from Original Experiments

### Environment Changes
- **horizon**: 20 → 40 (episodes now twice as long)
- **n_rollouts**: 32 → 16 (halved to maintain consistent compute/memory per iteration)
- **Compute per iteration**: ~640 env steps (same as original: 32×20 = 640, 16×40 = 640)

### Training Configuration
- **total_training_steps**: 2500 (continues from ~1300 → 2500)
- **total_epochs**: 2500
- **Checkpoint behavior**:
  - Loads from: Original run checkpoints (BL_run2, MA_pos_eps00_run1)
  - Saves to: New directories with `cont_h40` naming

### New Experiment Names
1. `OC_PPO_4B_BL_cont_h40_run1`
   - Loads from: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_4B_BL_run2/`
   - Saves to: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_4B_BL_cont_h40_run1/`
   - Config: Single-action baseline

2. `OC_PPO_4B_MA_pos_eps00_cont_h40_run1`
   - Loads from: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_4B_MA_pos_eps00_run1/`
   - Saves to: `/scratch/${USER}/checkpoints/rl_sdm/OC_PPO_4B_MA_pos_eps00_cont_h40_run1/`
   - Config: Multi-action with epsilon=0.0

## Files

### Original (horizon=20)
- `OC_PPO_4B_BL_2.sbatch` - Baseline, run 2
- `OC_PPO_4B_MA_pos_eps00_1.sbatch` - Multi-action, run 1

### Continuation runs (horizon=40, resume from h=20 checkpoints)
- `OC_PPO_4B_BL_cont_h40_1.sbatch` - Baseline continuation
- `OC_PPO_4B_MA_pos_eps00_cont_h40_1.sbatch` - Multi-action continuation

### Fresh runs (horizon=40, train from scratch)
- `OC_PPO_4B_BL_h40_1.sbatch` - Baseline fresh
- `OC_PPO_4B_MA_pos_eps00_h40_1.sbatch` - Multi-action fresh

## Expected Behavior - Continuation Runs (`cont_h40`)

1. **On startup**: Will automatically find and load latest checkpoint from original run directory
2. **Training range**: Resumes from ~step 1300, trains to step 2500 (~1200 additional steps)
3. **Checkpointing**: New checkpoints saved every 100 steps to new `cont_h40` directories
4. **Evaluation**: Runs every 100 steps at the new horizon=40 setting

## Expected Behavior - Fresh Runs (`h40`)

1. **On startup**: Initializes fresh model weights (no checkpoint loading)
2. **Training range**: Trains from step 0 to step 2500 (full training run)
3. **Checkpointing**: Saves checkpoints every 100 steps to `h40` directories
4. **Evaluation**: Runs every 100 steps at horizon=40 setting from the start

## Comparison: Continuation vs Fresh

| Aspect | Continuation (`cont_h40`) | Fresh (`h40`) |
|--------|---------------------------|---------------|
| **Starting weights** | Load from h=20 checkpoint ~step 1300 | Initialize fresh weights |
| **Training steps** | 1300 → 2500 (~1200 steps) | 0 → 2500 (full 2500 steps) |
| **Horizon experience** | Trained 1300 steps at h=20, then h=40 | All 2500 steps at h=40 |
| **Load checkpoint** | Yes, from original runs | No |
| **WANDB tags** | Includes "continuation" | No "continuation" tag |
| **Experiment names** | `*_cont_h40_run1` | `*_h40_run1` |
| **Purpose** | Test effect of switching horizon mid-training | Establish baseline for h=40 from scratch |

## Key Config Parameters

```bash
# Environment
envs.n_rollouts=16              # Halved from 32
envs.overcooked_kwargs.horizon=40  # Doubled from 20

# Training
trainer.total_training_steps=2500
trainer.total_epochs=2500
trainer.load_checkpoint=$LOAD_CHECKPOINT_DIR  # Points to original run
trainer.default_local_dir=.../${experiment_name}_run${run_number}  # New save location

# Checkpointing
trainer.save_freq=100
trainer.test_freq=100
trainer.max_actor_ckpt_to_keep=1
trainer.max_critic_ckpt_to_keep=1
```

## Notes

- **WANDB tags** updated to include `"horizon:40"` and `"continuation"`
- **RAY_LOG_TO_STDERR** enabled for BL, disabled for MA (as in originals)
- All other hyperparameters unchanged from originals
- Same model: Qwen3-4B-Instruct-2507
- Same hardware: 2 GPUs, 300GB RAM, 96 hour time limit
