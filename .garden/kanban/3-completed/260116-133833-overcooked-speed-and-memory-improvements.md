# Overcooked speed and memory improvements

**Type:** feat
**Branch:** oc-speed
**Created:** 2026-01-16 13:38
**Started:** 2026-01-16
**Completed:** 2026-01-16

## Goal
Profile Overcooked env pipeline and implement quick wins to improve speed. Memory is secondary (batched evals already mitigate OOM).

## Scope
- [x] Profile Overcooked-specific code to identify bottlenecks
- [x] Identify where time is spent (JAX env stepping, captioner, state conversion)
- [x] Implement obvious/easy optimizations found during profiling
- [x] Document findings for future optimization work

## Out of Scope
- Major architectural changes to env framework
- Multiprocessing/pipe optimizations (separate card exists)
- Deep optimizations requiring significant refactoring
- Memory-focused work (already handled via batched evals)

## Key Decisions
- Focus on Overcooked-specific code: jax_overcooked.py, captioner, env wrapper
- Quick wins only - profile then fix easy bottlenecks
- Speed is priority over memory

## Working Notes
### 2026-01-16 - Feature Started
Interview summary:
- Problem: Both speed and memory issues, but speed is priority
- Memory: Managed via batched evals (50 parallel envs ok, more causes OOM during eval)
- Speed bottleneck: Unknown - profiling needed to pinpoint
- Scope: Profile + quick wins only (not deep optimization)
- Focus: Overcooked-specific code (JAX env, captioner)

Next steps:
1. Add profiling instrumentation to Overcooked env code
2. Run test to capture timing data
3. Identify hotspots
4. Implement easy fixes

### 2026-01-16 - Context from Docs

**From overcooked-jaxmarl-implementation.md:**
- Key files: `verl/envs/environments/overcooked/`
  - `jaxmarl_wrapper.py` - wraps JaxMARL env, forces JAX CPU backend
  - `base.py` - LLMAgentsWrapper (action extraction, restructure_obs)
- JAX CPU backend required (GPU causes CUDA context issues with async workers)
- Must use `spawn` multiprocessing (not `fork`) - JAX deadlocks otherwise
- Grid decoding has bit manipulation for dynamic items
- Actions: right/down/left/up/stay/interact

**From wrapper-interface-api.md:**
- `extract_action()` - parses LLM output → action string
- `restructure_obs()` - converts state to text (long_term_context + short_term_context)
- `step()` / `reset()` - standard gym interface
- Architecture: `EnvWrapper(LLMAgentsWrapper(BaseEnv)) → VecEnv worker → Captioner`

**From file-structure-scope.md:**
- Overcooked at `verl/envs/environments/overcooked/`
- Captioners at `verl/envs/captioners/` (naive.py, cot.py)
- `env_wrapper.py` is base wrapper interface

**Potential hotspots to profile:**
1. JAX env stepping (jaxmarl_wrapper.py)
2. State → text conversion (restructure_obs in base.py)
3. Grid visualization/printing
4. Bit manipulation for item decoding
5. Captioner history building

### 2026-01-16 - Profiling Results & Optimizations

**Root cause identified:** `get_state_info()` was 55% of total time!

The bottleneck was JAX array indexing: each `int(agents.pos.x[agent_idx])` triggered full JAX dispatch through `_rewriting_take_via_slice`. With 4 array accesses × 2 agents = 8 JAX dispatch operations per step.

**Optimizations implemented in `jaxmarl_wrapper.py`:**
1. **Batched JAX→NumPy conversions**: Convert whole arrays once (`np.array(agents.pos.x)`), then index numpy (fast)
2. **Cached static objects**: `_get_static_objects()` now returns cached result (positions don't change)
3. **Cached pot positions**: `_get_pot_info()` caches positions, only reads contents at known locations

**Results (500 steps):**
| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| `get_state_info()` | 0.172ms | 0.005ms | **34x** |
| `base_env.render()` | 0.240ms | 0.041ms | **5.9x** |
| `LLMWrapper.step()` | 0.373ms | 0.157ms | **2.4x** |
| Total time | 0.339s | 0.133s | **2.5x** |

**Profiling script added:** `verl/envs/environments/overcooked/profile_overcooked.py`

### 2026-01-16 - Feature Complete
Achieved 2.5x speedup for Overcooked env step execution. Root cause was JAX array indexing triggering full dispatch per access. Fixed with batched conversions and caching.

## Original Notes
Card created via /feat interview. Original was a blank template.
