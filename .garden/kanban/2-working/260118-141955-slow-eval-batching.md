# Fix Slow Eval Batching

**Type:** fix
**Branch:** fix/slow-eval-batching
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** —

## Goal
Fix slow batched eval performance in StateVisitation eval where "other" time is ~2500s vs ~500s for non-batched evals.

## Scope
- [ ] Investigate batching implementation in evaluator
- [ ] Identify root cause of slow "other" time
- [ ] Fix batching logic performance issue
- [ ] Validate timing instrumentation handles batching correctly
- [ ] Test fix with StateVisitation eval

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
