# (fix) - Token metrics include post-terminal generations

**Type:** fix
**Created:** 2026-01-15

## Problem

In `MultiEnvEvaluator`, token metrics are computed from generations that continue after rollouts have terminated:

1. **`total_tokens_generated`**: Sums tokens for ALL rollouts every step, even those that have ended. After auto-reset, we're counting tokens from a new episode.

2. **`toks_out_mean/std`**: Uses `response_n_tokens` from the last step that was executed. If some rollouts ended at step 5 but others continued to step 10, the step-5 rollouts will have their token counts from step 10 (which is a different episode post auto-reset).

**Result**: Token metrics don't accurately reflect the evaluated trajectories. They include generation overhead from post-terminal steps and potentially from restarted episodes.

## Expected Behavior

Token metrics should only count tokens generated during the actual evaluated trajectory:
- `total_tokens_generated`: Only count tokens for active (non-terminated) rollouts
- `toks_out_mean/std`: Use token counts from each rollout's final step before termination

## Suggested Fix

Track per-rollout token counts and freeze them on termination:

```python
# Initialize
rollout_token_counts = np.zeros(n_rollouts, dtype=np.int64)
rollout_final_step_tokens = np.zeros(n_rollouts, dtype=np.int64)

# In step loop, before updating end_of_traj:
active_mask = ~end_of_traj if end_of_traj is not None else np.ones(n_rollouts, dtype=bool)
rollout_token_counts += response_n_tokens.numpy() * active_mask
# Capture final step tokens for rollouts that just ended
newly_ended = done & ~end_of_traj
rollout_final_step_tokens[newly_ended] = response_n_tokens[newly_ended]
```

## Context

- File: `verl/trainer/ppo/multi_env_evaluator.py`
- Discovered during batched-evals review but is pre-existing behavior
- Related to evaluator continuing to generate for ended rollouts
