# Better Eval Timers

**Type:** feat
**Branch:** feat/better-eval-timers
**Created:** 2026-01-17
**Started:** 2026-01-17
**Completed:** —

## Goal
Add granular timing metrics to MultiEnvEvaluator: end-to-end eval time plus component breakdown (text generation, environment steps, other slow ops). Output to both W&B/logger and console.

## Scope
- [ ] Review existing eval timer implementation for correctness
- [ ] Add timing for text generation phase
- [ ] Add timing for environment steps phase
- [ ] Add end-to-end eval timing
- [ ] Identify and time any other slow components
- [ ] Log all timings to W&B/existing logger
- [ ] Print timing summary to console

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
