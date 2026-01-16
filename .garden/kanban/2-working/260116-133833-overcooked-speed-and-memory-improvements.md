# Overcooked speed and memory improvements

**Type:** feat
**Branch:** oc-speed
**Created:** 2026-01-16 13:38
**Started:** 2026-01-16
**Completed:** —

## Goal
Profile Overcooked env pipeline and implement quick wins to improve speed. Memory is secondary (batched evals already mitigate OOM).

## Scope
- [ ] Profile Overcooked-specific code to identify bottlenecks
- [ ] Identify where time is spent (JAX env stepping, captioner, state conversion)
- [ ] Implement obvious/easy optimizations found during profiling
- [ ] Document findings for future optimization work

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

## Original Notes
Card created via /feat interview. Original was a blank template.
