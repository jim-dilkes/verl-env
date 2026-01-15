# (feat) - Implement batched evals in MultiEnvEvaluator to prevent OOM

**Type:** feat
**Branch:** batched-evals
**Created:** 2026-01-15 10:24
**Started:** 2026-01-15 10:24
**Completed:** —

## Goal
Prevent OOM during evaluation by batching environment rollouts instead of creating all at once.

## Scope
- [x] Add `batch_size` config option per evaluation environment
- [ ] Modify `MultiEnvEvaluator.evaluate()` to process rollouts in batches
- [ ] Accumulate metrics across batches, aggregate once at end
- [ ] Default behavior unchanged (`batch_size=None` = run all at once)

## Out of Scope
- Streaming/partial metrics logging
- Global batch_size (each eval env specifies its own)
- Changes to training rollout batching

## Design

### Config location
Each evaluation environment in `config.evaluation.environments[]` can specify:
```yaml
environments:
  - name: "my_eval"
    n_rollouts: 256
    batch_size: 64  # NEW: optional, defaults to None (all at once)
    ...
```

### Implementation approach
In `_evaluate_single_env_body()`:
1. If `batch_size` is None or >= `n_rollouts`, run as before (no batching)
2. Otherwise, split `n_rollouts` into chunks of `batch_size`
3. For each chunk:
   - Create temp config with `n_rollouts = batch_size`
   - Create VecEnv, run episode loop, close VecEnv
   - Accumulate raw trajectory data (rewards, lengths, scores, etc.)
4. After all chunks: compute aggregated metrics once

### Key considerations
- `seed_group_size` must align with batching (groups shouldn't be split across batches)
- Episode tracking for logging: pick from first batch only
- Memory cleanup: `gc.collect()` between batches (already done between environments)
- Entropy probing: each batch independently measures (measurements accumulate)

## Acceptance Criteria
- Setting `batch_size: 64` with `n_rollouts: 256` runs 4 sequential batches
- Metrics match non-batched run (same seeds, same results)
- OOM resolved for large n_rollouts evaluations
- Existing configs without `batch_size` work unchanged

## Test Cases
- [ ] `batch_size=None` → same behavior as before
- [ ] `batch_size >= n_rollouts` → same behavior as before
- [ ] `batch_size < n_rollouts` → runs multiple batches, metrics aggregate correctly
- [ ] `batch_size` not divisible into `n_rollouts` → last batch smaller, still works
- [ ] `seed_group_size` validation: error if `batch_size < seed_group_size`

## Constraints
- Must not break existing eval configs
- Metrics must be identical to non-batched (given same seeds)

## Context
- Key file: `verl/trainer/ppo/multi_env_evaluator.py`
- VecEnv creation: `make_vec_env()` uses `config.envs.n_rollouts` to determine parallelism
- Current loop: `evaluate()` iterates environments, `_evaluate_single_env_body()` runs full episode

## Interview Notes
- OOM caused by too many parallel envs (not model batch size)
- Want optional fixed batch size per eval environment
- Aggregation at end (no streaming metrics)
- `eval.batch_size` per-env, default=None means all at once (current behavior)
