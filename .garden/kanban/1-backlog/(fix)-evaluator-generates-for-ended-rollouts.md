# (fix) - Evaluator generates for ended rollouts

**Type:** fix
**Created:** 2026-01-15

## Problem

`MultiEnvEvaluator` continues calling `actor_rollout_wg.generate_sequences()` for ALL rollouts every step, even after some rollouts have terminated. The VecEnv `__SKIP__` mechanism exists but is not used.

**Consequences**:
1. Wasted compute: generating responses for rollouts that have already ended
2. Memory overhead: maintaining generation context for inactive rollouts
3. Metric pollution: token counts and other metrics include post-terminal generations (see related card)
4. After auto-reset, we're generating for new episodes that aren't being evaluated

## Current Behavior

```python
# Step loop generates for all rollouts
val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
# Uses end_of_traj to skip accumulating rewards, but generation still happens
```

## Expected Behavior

Once a rollout terminates, skip generation for that rollout:
- Either use `__SKIP__` action to prevent env stepping
- Or filter the generation batch to only include active rollouts
- Or mask outputs from terminated rollouts

## Options

1. **Use `__SKIP__` mechanism**: Send `__SKIP__` action for ended rollouts so VecEnv doesn't step them
2. **Filter generation batch**: Only generate for active rollouts, requires dynamic batch sizing
3. **Early exit**: Break loop when `end_of_traj.all()` (already done) but also skip generation for individual ended rollouts

Option 1 is simplest but still generates (just doesn't step). Option 2 is most efficient but more complex.

## Context

- File: `verl/trainer/ppo/multi_env_evaluator.py`
- VecEnv supports `__SKIP__` action in `worker()` function
- Discovered during batched-evals review but is pre-existing behavior
- Fixing this would also address token metrics pollution
