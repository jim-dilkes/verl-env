# (fix) - Entropy probing tracks restarted episodes after auto-reset

**Type:** fix
**Created:** 2026-01-15

## Problem

In `MultiEnvEvaluator._evaluate_single_env_body()`, entropy probing uses `active_rollouts` to determine which rollouts to probe:

```python
if entropy_enabled and active_rollouts is not None:
    done_mask = np.logical_or(terminated_vec, truncated_vec)
    active_rollouts = np.logical_not(done_mask)
```

However, `VecEnv` auto-resets on termination: when an episode ends, the worker immediately resets and returns the new observation while still marking `terminated=True` for that step. On the **next** step, `terminated` will be `False` (new episode started), so `active_rollouts` becomes `True` again.

**Result**: If a rollout ends at step k and entropy is measured at step k+1, we probe step 1 of a **new episode**, not "step k+1 of the original trajectory."

## Expected Behavior

Entropy probing should only measure states from the original trajectory. Once a rollout terminates, it should be excluded from all subsequent entropy measurements.

## Suggested Fix

Track termination persistently across the episode:

```python
# Initialize once at start
ever_terminated = np.zeros(n_rollouts, dtype=bool)

# In step loop, after env.step():
done_mask = np.logical_or(terminated_vec, truncated_vec)
ever_terminated = np.logical_or(ever_terminated, done_mask)
active_rollouts = np.logical_not(ever_terminated)  # Never becomes True again
```

## Context

- File: `verl/trainer/ppo/multi_env_evaluator.py`
- Discovered during batched-evals review but is pre-existing behavior
- Affects entropy metrics accuracy when rollouts have varying lengths
