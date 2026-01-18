# Fix Slow Eval Batching

**Type:** fix
**Branch:** fix/slow-eval-batching
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** —

## Goal
Fix slow batched eval performance in StateVisitation eval where "other" time is ~2500s vs ~500s for non-batched evals.

## Scope
- [x] Investigate batching implementation in evaluator
- [x] Identify root cause of slow "other" time
- [x] Fix batching logic performance issue
- [x] Validate timing instrumentation handles batching correctly
- [ ] Test fix with StateVisitation eval on cluster

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
