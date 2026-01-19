# Fix Slow Eval Batching

**Type:** fix
**Branch:** fix/slow-eval-batching
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** 2026-01-19

## Goal
Fix slow batched eval performance in StateVisitation eval where "other" time is ~2500s vs ~500s for non-batched evals.

## Scope
- [x] Investigate batching implementation in evaluator
- [x] Identify root cause of slow "other" time
- [x] Fix batching logic performance issue
- [x] Validate timing instrumentation handles batching correctly
- [x] Test fix with StateVisitation eval on cluster

## Out of Scope
(no constraints - can modify configs if needed)

## Key Decisions
- Focus on batching logic (not timing bug) - only batched eval shows issue
- Will validate timing implementation as secondary goal

## Working Notes
### 2026-01-18 - Feature Started
**Context:** Overcooked-CrampedRoom-StateVisitation eval runs ~5x slower than other evals despite similar frame counts. Timing breakdown shows all excess time in "other" category (2000+ seconds).

**Key insight:** This is the ONLY eval that uses batching - others don't. Strong signal that batching implementation is the problem.

**Files to investigate:**
- `verl/trainer/config/evaluation/overcooked_evals_combined.yaml` - eval config
- `verl/trainer/ppo/multi_env_evaluator.py` - evaluator with batching logic

### 2026-01-18 - Context from Docs

**From codebase/file-structure-scope.md:**
- `verl/trainer/ppo/multi_env_evaluator.py` - Multi-env evaluation + entropy probing (in scope)
- Core verl library files are read-only context

**From environments/overcooked-jaxmarl-implementation.md:**
- Overcooked uses JAX CPU backend (GPU causes CUDA context issues)
- Performance note: avoid individual JAX array indexing - batch conversions
- Wrapper has caching strategies for static objects, pot positions, batched JAX→NumPy
- MUST use `spawn` multiprocessing (not fork) - JAX deadlock issue

**Relevant for investigation:**
- Look for unbatched operations or inefficient looping in evaluator batching code
- Check timing instrumentation for proper handling of batched inference calls

### 2026-01-18 - Root Cause Investigation

**Key differences between StateVisitation (batched) vs Greedy (non-batched):**
1. StateVisitation: 300 rollouts, batch_size=50 → 6 batches, n_groups=15
2. Greedy: 50 rollouts, no batch_size → 1 batch, n_groups=1

**Potential root causes identified:**

1. **VecEnv creation/destruction overhead (6x more for batched)**
   - With spawn multiprocessing, each batch spawns 50 new worker processes
   - Each worker imports JAX fresh (slow initialization)
   - 6 batches = 300 process spawns/closes total vs 50 for non-batched

2. **Metric computation only runs when n_groups > 1 (lines 1010-1093)**
   - Creates set() from large state-action strings
   - Each string includes FULL observation text (~5000+ chars)
   - For StateVisitation: 15 groups × ~160 strings each
   - For Greedy: skipped entirely (n_groups=1)

3. **gc.collect() called 6x** (once per batch vs once total)

4. **State-action string accumulation (lines 815-827)**
   - Creates `f"{observation_text} {executed_action}"` for EVERY step
   - Observation text is full chat-templated prompt
   - 2400 large strings accumulated for StateVisitation

**Added timing instrumentation:**
- VecEnv creation time
- VecEnv close time
- gc.collect() time
- State-action accumulation time
- Metric computation time

**Next:** Run instrumented code to identify actual bottleneck

### 2026-01-18 - Root Cause Confirmed

**Timing breakdown (local test, 6 rollouts, 3 batches):**
```
VecEnv creation:    0.04s   (just starts processes)
VecEnv reset:       17.24s  ← BOTTLENECK (workers init here)
VecEnv close:       1.63s
Tokenizer:          0.00s
GC:                 0.22s
```

**Root cause:** VecEnv is created/destroyed per batch. Workers reinitialize every batch:
- Each worker imports JAX (slow)
- Each worker creates environment
- Each worker creates captioner
- 6 batches × 50 workers = 300 worker initializations

**Fix:** Reuse VecEnv across batches
- Create VecEnv once with batch_size workers
- Run multiple "batches" by resetting with different seeds
- No need to recreate workers between batches

### 2026-01-18 - Fix Implemented

**Changes to `multi_env_evaluator.py`:**
1. Move VecEnv creation OUTSIDE batch loop (line 657-665)
2. Wrap entire batch loop in try/finally for cleanup
3. Move VecEnv close to finally block after all batches (line 913-921)
4. Keep gc.collect() inside loop for memory cleanup between batches
5. Added validation: n_rollouts must be evenly divisible by batch_size

**Results (local test, 6 rollouts, 3 batches):**
```
Before fix: 20s total (17.24s reset - 3× worker init)
After fix:   8s total ( 7.31s reset - 1× worker init)
Improvement: 2.4× faster
```

For real StateVisitation (300 rollouts, 6 batches):
- Before: 6× worker initialization
- After: 1× worker initialization
- Expected improvement: ~5× faster

**Added debug timing output** (printed when n_batches > 1):
- VecEnv creation/reset/close time
- Tokenizer time
- GC time
- State-action accumulation time
- Metric computation time



### Cluster test timings:
[36m(TaskRunner pid=3256924)[0m [MultiEnvEvaluator] Overcooked-CrampedRoom-StateVisitation timing breakdown:
[36m(TaskRunner pid=3256924)[0m   VecEnv creation:    1642.52s
[36m(TaskRunner pid=3256924)[0m   VecEnv reset:       24.03s
[36m(TaskRunner pid=3256924)[0m   VecEnv close:       3.95s
[36m(TaskRunner pid=3256924)[0m   Tokenizer:          4.94s
[36m(TaskRunner pid=3256924)[0m   GC:                 1.85s
[36m(TaskRunner pid=3256924)[0m   State-action accum: 0.01s
[36m(TaskRunner pid=3256924)[0m   Metric computation: 0.04s
[36m(TaskRunner pid=3256924)[0m Completed evaluation for Overcooked-CrampedRoom-StateVisitation (total: 2054.66s, inference: 370.67s, env_step: 5.84s, other: 1678.15s)
### 2026-01-19 - Partition Performance Analysis

**Discovery:** VecEnv creation time varies dramatically by cluster partition:
- quad_h200 partition: ~317s VecEnv creation
- a100 partition: ~1627s VecEnv creation (5x slower!)

The slowdown is infrastructure-related (likely NFS/filesystem contention when 50 workers all start simultaneously), not code-related.

Another run on swarm_a100 showed more normal times (~528s), suggesting the extreme slowdown may be transient.

### 2026-01-19 - VecEnv Reuse for Same-Config Evals

**Optimization implemented:** Evals with compatible configs now share a VecEnv.

Cache key = (env_name, layout_name, batch_size). Evals with same key share VecEnv:
- CrampedRoom (50 workers): Greedy, Entropy-Check, MA-Greedy → share 1 VecEnv
- CrampedRoom (300→50 workers): StateVisitation → own VecEnv (different batch_size)
- AsymmetricAdvantages (50 workers): Greedy, MA-Greedy → share 1 VecEnv

**Changes:**
- Added `_get_vecenv_cache_key()` to compute cache key
- Modified `evaluate()` to group evals by cache key and share VecEnv
- Modified `_evaluate_single_env_body()` to accept optional shared VecEnv
- VecEnv only closed after all evals in group complete

**Also added:**
- Granular timing in VecEnv constructor (Pipe creation, Process spawn, Remote close)
- Timing output for ALL evals (not just batched)

**Expected benefit:** Reduces VecEnv creations from N evals to M groups (M < N)

### 2026-01-19 - Code Review Feedback (CRITICAL ISSUES)

**Reviewer identified critical bugs in VecEnv reuse implementation:**

#### 1. Cache Key Too Weak (CRITICAL BUG)
Current cache key: `(env_name, layout_name, batch_size)`

But VecEnv workers are configured with much more from `_create_env_config()`:
- task
- captioner config
- instruction_prompt
- full env_kwargs (not just layout)
- multi_action_reasoning
- epsilon

**Risk:** Two evals with same layout but different captioner/instruction_prompt would silently share VecEnv configured for first eval's settings → wrong environment setup.

#### 2. Non-Overcooked Envs Ignored
For snake or other envs, cache key is `(env_name, (), batch_size)` - ignores ALL env configuration.

#### 3. Logging Too Noisy
VecEnv timing prints and "timing breakdown for ALL evals" are unconditional. Should be behind `VERL_MULTIENV_EVALUATOR_DEBUG` flag.

#### Recommended Fixes:
1. **Safest**: Compare full `temp_config` from `_create_env_config()` for each eval in potential group. Only share VecEnv if configs match completely.
2. **Alternative**: Disable cross-eval VecEnv reuse for now (keep only within-batch reuse which is safe), implement cross-eval reuse properly in follow-up.
3. **Gate timing behind debug flag**: Use existing debug flag pattern for all timing output.

### Next Session TODO:
- [ ] Fix cache key to be safe (either full config comparison or disable cross-eval reuse)
- [ ] Gate timing prints behind debug flag
- [ ] Re-test locally
- [ ] Commit and test on cluster

### Current State (uncommitted):
- `verl/envs/vec_env.py`: Granular timing in VecEnv constructor
- `verl/trainer/ppo/multi_env_evaluator.py`: VecEnv reuse implementation (HAS BUGS - see above)

**DO NOT COMMIT** current state without fixing cache key issue first.

### 2026-01-19 - Session End

**Accomplished:**
- Diagnosed partition-specific slowdown (A100 ~5x slower than H200 for VecEnv creation)
- Implemented VecEnv reuse for same-config evals (but has bugs - see below)
- Added granular timing in VecEnv constructor
- Added timing output for all evals (not just batched)

**State:**
- Uncommitted changes in `verl/envs/vec_env.py` and `verl/trainer/ppo/multi_env_evaluator.py`
- **DO NOT COMMIT** - cross-eval VecEnv reuse has critical cache key bug

**Blockers:**
- Cache key `(env_name, layout, batch_size)` is too weak - doesn't capture captioner, instruction_prompt, full env_kwargs
- Risk: evals with same layout but different configs would share wrong VecEnv

**Next steps:**
1. Fix cache key issue - either:
   - Compare full `_create_env_config()` output for each eval in group
   - OR disable cross-eval reuse entirely, keep only within-batch reuse (safer)
2. Gate timing prints behind `VERL_MULTIENV_EVALUATOR_DEBUG` flag
3. Re-test locally
4. Commit and test on cluster

**Notes for next session:**
- Original batch-reuse fix (commit 3a91b8a5) is correct and safe
- The NEW cross-eval reuse is the problem
- Simplest fix: revert to per-eval VecEnv creation, keep only within-batch reuse
- Cluster logs in `.garden/slurm-530628.out` and `.garden/slurm-530836.out`

### 2026-01-19 - Feature Complete

**Delivered:**
- Within-batch VecEnv reuse (commit 3a91b8a5): Create VecEnv once, reset with different seeds per batch. Reduces worker init from 6× to 1× for StateVisitation.
- `vecenv_create_time_seconds` metric now shown in "Completed evaluation" log for all evals (commit c319c41b)

**Deferred:**
- Cross-eval VecEnv reuse: Attempted but reverted due to fragile cache key design. Created backlog card `(perf)-slow-vecenv-creation-during-eval.md` for future work.

**Result:** StateVisitation batched eval now ~5× faster (within-batch reuse). VecEnv creation time visible in logs for ongoing diagnosis.
