# Better Eval Timers

**Type:** feat
**Branch:** feat/better-eval-timers
**Created:** 2026-01-17
**Started:** 2026-01-17
**Completed:** —

## Goal
Add granular timing metrics to MultiEnvEvaluator: end-to-end eval time plus component breakdown (text generation, environment steps, other slow ops). Output to both W&B/logger and console.

## Scope
- [x] Review existing eval timer implementation for correctness
- [x] Add timing for text generation phase (already existed, verified correct)
- [x] Add timing for environment steps phase
- [x] Add end-to-end eval timing
- [x] Identify and time any other slow components (captured as "other")
- [x] Log all timings to W&B/existing logger
- [x] Print timing summary to console

## Out of Scope
- Training loop timing (ray_multistep_trainer)
- Changes to other evaluators

## Key Decisions
- Component breakdown: generation, env steps, + discovered slow ops
- Dual output: W&B + console
- Target: multi_env_evaluator.py only

## Working Notes
### 2026-01-17 - Feature Started
Interview summary:
- User wants better visibility into eval performance
- Focus on MultiEnvEvaluator component timing
- Need to audit existing timer first for correctness
- Output both to metrics system and console for immediate visibility

### 2026-01-17 - Context from Docs

**From codebase/file-structure-scope.md:**
- Target file: `verl/trainer/ppo/multi_env_evaluator.py`
- This handles multi-env evaluation + entropy probing
- Called from ray_multistep_trainer.py (which is out of scope)

**From training/exploration-metrics-definitions.md:**
- Evaluator computes state visitation metrics, entropy probing
- These are likely time-intensive operations to measure
- Entropy probing involves multiple LLM completions per step

**Gaps identified:**
- No existing docs on timing/logging patterns
- Need to read multi_env_evaluator.py directly for current timing impl

### 2026-01-17 - Existing Timer Audit

**Current timing in multi_env_evaluator.py:**

1. **`total_inference_time`** (line 593, 730-733):
   - Accumulates `actor_rollout_wg.generate_sequences()` time (text gen)
   - Logged to W&B: `inference_time_seconds`, `inference_time_per_rollout`, `inference_time_per_step`, `inference_time_per_step_cap`
   - ✅ Correctly measures text generation

2. **`start_time/eval_time`** (line 192-199, 213):
   - Times entire `_evaluate_single_env()` call
   - Printed to console but **NOT logged to W&B** ❌
   - Console: `Completed evaluation for {eval_name} in {eval_time:.2f}s (inference: {inference_time:.2f}s)`

3. **`entropy_probe_time`** (line 612, 689):
   - Times entropy probing generations
   - Logged as `action_entropy_probe_time_seconds` ✅

**Missing timings:**
- Environment steps (`vec_envs.step()`) - not measured explicitly
- Tokenization overhead (`self.tokenizer()` calls)
- Chat template application (`apply_chat_template`)
- Metric computation at end of eval

**Implicit time calculation:**
- `env_step_time ≈ eval_time - inference_time - entropy_probe_time`
- But we should measure explicitly for accuracy

**Plan:**
1. Add explicit `env_step_time` accumulator
2. Add `eval_time_seconds` to W&B metrics
3. Add detailed timing breakdown console print
4. Consider tokenization timing (may be negligible)

### 2026-01-17 - Implementation Complete

**Changes made to `multi_env_evaluator.py`:**

1. Added `total_env_step_time` accumulator (line 604)
2. Wrapped `vec_envs.step()` with timing (lines 761-765)
3. Added env_step metrics to metric_dict:
   - `env_step_time_seconds` - total env step time
   - `env_step_time_per_rollout` - per rollout
   - `env_step_time_per_step` - per executed step
4. Added `eval_time_seconds` to prefixed_metrics (logged to W&B)
5. Updated console print with detailed breakdown:
   - Format: `Completed evaluation for X (total: Xs, inference: Xs, env_step: Xs, [entropy_probe: Xs,] other: Xs)`
   - `other` captures tokenization, metric computation, etc.

**Decision:** Skipped separate tokenization timing - captured in "other" bucket. Can be added later if profiling shows it's significant.

**Committed:** `512fa472` - feat: add granular timing metrics to MultiEnvEvaluator

**Test added:** `tests/trainer/ppo/test_multi_env_evaluator_timing_on_cpu.py` (5 tests, all pass)

### 2026-01-17 - Review Fixes

Applied fixes from code review:
1. `time.time()` → `time.perf_counter()` (more robust for duration measurement)
2. Added `other_time_seconds` to W&B metrics (was only printed to console)
3. Clamped `other_time` to `max(0.0, ...)` (prevent negative values from rounding)
4. Moved `end_time` to after `_maybe_log_episode_generation()` (includes logging overhead)
5. Updated wandb-metrics.md documentation with all timing metrics
6. Added 2 more tests (7 total, all pass)

**Definition of "end-to-end":** `eval_time_seconds` = time from start of `_evaluate_single_env()` to after episode logging. Excludes post-eval GC (`gc.collect()`).

**Next:** Run on cluster to verify timing output in real eval run.
