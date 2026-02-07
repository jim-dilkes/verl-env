# MARA + MaxRL Test Runs (2026-02-07)

## Purpose
Quick validation runs on `scavenger_mathsa100` to verify new features work
end-to-end on Iridis before queueing full CAIS experiments.

## Experiments

### TEST_MARA_OC_4B — MARA reward augmentation
- **Branch:** `feat/mara-tre-entropy` on `verl-contrib`
- **What's different:** `algorithm.use_mara=True`, `algorithm.mara_tau=0.5`, `algorithm.mara_beta=0.001`
- **Uses:** GRPO advantage estimator + MARA reward augmentation on top
- **Verify:** Check wandb for `mara/num_groups_augmented > 0` after a few steps
- **KL coef:** 0.001 (must match `mara_beta` for correct anchoring)

### TEST_MAXRL_OC_4B — MaxRL advantage estimator
- **Branch:** `feat/maxrl` on `verl-contrib`
- **What's different:** `algorithm.adv_estimator=maxrl` (instead of `grpo`)
- **Uses:** MaxRL normalizes by K (successes) not N (total), approximating ML objective
- **Verify:** Training should proceed normally; check advantage magnitudes in wandb
- **KL coef:** 0.0 (pure MaxRL, no KL penalty)

## Common settings
- **Model:** Qwen3-4B-Instruct-2507
- **Env:** Overcooked cramped_room, horizon=20, shaped_reward
- **Steps:** 100 (short validation run)
- **Partition:** scavenger_mathsa100 only (spare capacity, won't impact main queue)
- **Time limit:** 12h (generous for 100 steps)

## Before submitting
1. Ensure correct branch is checked out in `verl-contrib` conda env on Iridis
2. For MARA: `cd verl-contrib && git checkout feat/mara-tre-entropy && pip install -e .`
3. For MaxRL: `cd verl-contrib && git checkout feat/maxrl && pip install -e .`
4. Note: Can't run both simultaneously from same conda env (different branches)
   - Option A: Run sequentially (MARA first, then switch branch, then MaxRL)
   - Option B: Create separate conda envs

## Submit
```bash
cd /path/to/verl-env
sbatch experiments/tests/260207_mara_maxrl_test/TEST_MARA_OC_4B_1.sbatch
sbatch experiments/tests/260207_mara_maxrl_test/TEST_MAXRL_OC_4B_1.sbatch
```
