# (feat) - Implement batched evals in MultiEnvEvaluator to prevent OOM

**Type:** feat
**Branch:** batched-evals
**Created:** 2026-01-15 10:24
**Started:** 2026-01-15 21:30
**Completed:** 2026-01-15 21:39

## Goal
Prevent OOM during evaluation by batching environment rollouts instead of creating all at once.

## Scope
- [x] Add `batch_size` config option per evaluation environment
- [x] Modify `MultiEnvEvaluator.evaluate()` to process rollouts in batches
- [x] Accumulate raw per-rollout data across batches, aggregate once at end
- [x] Default behavior unchanged (`batch_size=None` = run all at once)

## Implementation Status

### Completed Changes

**1. VecEnv.reset() interface** (`verl/envs/vec_env.py:176-220`)
- Added `seeds: Optional[List[int]]` parameter
- If seeds list provided, uses them directly (one per worker)
- Otherwise falls back to existing seed computation logic

**2. MultiEnvEvaluator changes** (`verl/trainer/ppo/multi_env_evaluator.py`)
- Added `_compute_seed_sequence()` static method (lines 1132-1169)
  - Computes full seed list upfront for determinism
  - Handles both incremental seeds and seed_group_size grouping
- Modified `_create_env_config()` to accept `n_rollouts_override` parameter (line 260)
- Refactored `_evaluate_single_env_body()` (lines 509-1006):
  - Batch loop with VecEnv creation per batch
  - Global indexing for group assignment across batches
  - Accumulated arrays for metrics computation
- Simplified `evaluate()` - VecEnv creation moved inside `_evaluate_single_env_body()`
- Validation: `n_rollouts > 0`, `batch_size > 0` if provided

### Review Fixes Applied

**1. toks_out_mean/std batch invariance**
- Problem: Used only last batch's `response_n_tokens` tensor, which varies when last batch is smaller
- Fix: Collect per-rollout token counts across all batches via `response_n_tokens_last_step[batch_start:batch_end]`

**2. seed_group_size=None crash**
- Problem: Config could explicitly set `seed_group_size: null`, causing `n_rollouts % None` error
- Fix: Normalize `seed_group_size` to `n_rollouts` if None, validate > 0 before use

**3. Entropy-only runs crash on group metrics**
- Problem: When `exclusive_metric=True`, `all_len_of_traj` stays empty but per-frame/coverage metrics tried to index it
- Fix: Gate per-frame/coverage metrics on `if all_len_of_traj:`

### Behavior Notes

- **Seed semantics**: `_compute_seed_sequence()` mirrors existing `VecEnv.reset(use_incremental_seeds=True)` behavior, including `seed_group_size == n_rollouts` meaning "no grouping" (incremental seeds)
- **Existing constraint preserved**: `n_rollouts % seed_group_size == 0` still required (card only promised no batch_size/seed_group_size alignment)
- **Batch recreation overhead**: VecEnv recreated per batch (simplicity over efficiency trade-off)
- **Print noise**: `VecEnvContextManager` prints "Successfully closed..." per batch - could gate behind debug flag later

## Out of Scope
- Streaming/partial metrics logging
- Global batch_size (each eval env specifies its own)
- Changes to training rollout batching
- VecEnv reuse across batches (recreate each batch for simplicity)

## Final Design

### Config location
Each evaluation environment in `config.evaluation.environments[]` can specify:
```yaml
environments:
  - name: "my_eval"
    n_rollouts: 256
    batch_size: 64  # NEW: optional, defaults to None (all at once)
    ...
```

### Validation
- `n_rollouts > 0` required (error otherwise)
- `batch_size > 0` if provided (error otherwise)
- No alignment constraints between batch_size and seed_group_size

### Seeding strategy (determinism)
1. Compute full seed sequence upfront (same logic as current `use_incremental_seeds=True`)
2. Slice contiguous seed ranges per batch
3. Pass explicit seeds to each batch's `reset()`
4. Guarantees identical rollout-to-seed mapping regardless of batch_size

### Batch loop structure
```python
def _evaluate_single_env(self, env_config, eval_name):
    n_rollouts = env_config['n_rollouts']
    batch_size = env_config.get('batch_size') or n_rollouts

    # 1. Compute full seed list upfront
    seeds = self._compute_seed_sequence(initial_seed, n_rollouts)

    # 2. Accumulator for raw per-rollout data
    all_rewards = []
    all_lengths = []
    all_scores = []
    all_state_action_texts = [[] for _ in range(n_rollouts)]  # global indexing
    # ... other accumulators

    # 3. Process in batches
    for batch_start in range(0, n_rollouts, batch_size):
        batch_end = min(batch_start + batch_size, n_rollouts)
        batch_seeds = seeds[batch_start:batch_end]
        batch_n = batch_end - batch_start

        # Create temp config with batch_n rollouts
        temp_config = self._create_env_config(env_config, n_rollouts=batch_n)

        with VecEnvContextManager(...) as vec_env:
            # Run episode loop, collect raw data
            batch_data = self._run_batch_rollouts(vec_env, batch_seeds, ...)

        # Accumulate with global indexing
        for local_idx, data in enumerate(batch_data):
            global_idx = batch_start + local_idx
            all_rewards.append(data.reward)
            all_lengths.append(data.length)
            # Group assignment uses global_idx
            group_idx = global_idx // seed_group_size
            all_state_action_texts[group_idx].extend(data.state_actions)

        gc.collect()  # Memory cleanup between batches

    # 4. Aggregate metrics once from accumulated data
    return self._compute_metrics(all_rewards, all_lengths, all_scores, ...)
```

### Key implementation details

**Global index tracking**: Groups can span batch boundaries. Group assignment happens during accumulation using `global_idx = batch_start + local_idx`.

**Episode tracking for logging**: Capture from first batch only (rollout 0).

**Entropy probing**: Runs per-batch, measurements accumulate across batches (existing pattern).

**Policy/model**: Already shared via `actor_rollout_wg`, no changes needed.

**VecEnv lifecycle**: Recreate per batch (same as current per-environment pattern). Simplicity over efficiency.

### Refactoring approach
Current `_evaluate_single_env_body` does:
1. Episode loop (step through env)
2. Raw data collection
3. Metric computation

Split into:
1. `_run_batch_rollouts()` - episode loop + raw data collection for one batch
2. `_compute_metrics()` - aggregation from accumulated data
3. `_evaluate_single_env_body()` - batch loop orchestration

## Acceptance Criteria
- Setting `batch_size: 64` with `n_rollouts: 256` runs 4 sequential batches
- Metrics identical to non-batched run (same seeds → same results)
- OOM resolved for large n_rollouts evaluations
- Existing configs without `batch_size` work unchanged

## Test Cases
- [ ] `batch_size=None` → same behavior as before
- [ ] `batch_size >= n_rollouts` → same behavior as before
- [ ] `batch_size < n_rollouts` → runs multiple batches, metrics match non-batched
- [ ] `batch_size` not divisible into `n_rollouts` → last batch smaller, still works
- [ ] `batch_size` smaller than `seed_group_size` → works (groups span batches)
- [ ] `n_rollouts=0` → error
- [ ] `batch_size=0` → error

## Constraints
- Must not break existing eval configs
- Metrics must be identical to non-batched (given same seeds)

## Context
- Key file: `verl/trainer/ppo/multi_env_evaluator.py`
- VecEnv creation: `make_vec_env()` uses `config.envs.n_rollouts`
- Current: `evaluate()` iterates environments, `_evaluate_single_env_body()` runs full episode

## Key Decisions
1. **Flexible batching**: No alignment constraints between batch_size and seed_group_size
2. **Global indexing**: Groups can span batch boundaries, assigned during accumulation
3. **Deterministic seeding**: Full seed list computed upfront, sliced per batch
4. **Simple lifecycle**: Recreate VecEnv per batch (no complex reuse)
5. **Validation**: Only `n_rollouts > 0` and `batch_size > 0` required

## Known Limitations

### Introduced by this card
1. **Model-side determinism not guaranteed**: Env seeding is controlled, but if model uses stochastic sampling (`do_sample=True`) and its RNG depends on batch ordering, results may differ across batch configurations. Eval typically uses greedy decoding (`do_sample=False`), so this is usually not an issue.

2. **Addresses env memory, not model memory**: Batching reduces peak env count but not model memory (KV cache, prompt embeddings). If OOM is from model side, this won't help.

### Pre-existing issues (not introduced by this card)
See backlog cards:
- `(fix)-entropy-probing-tracks-restarted-episodes.md`
- `(fix)-token-metrics-include-post-terminal-generations.md`
- `(fix)-evaluator-generates-for-ended-rollouts.md`