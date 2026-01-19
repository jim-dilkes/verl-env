# Action Distribution Tracking

**Type:** feat
**Branch:** feat/action_distribution_tracking
**Created:** 2026-01-19
**Started:** 2026-01-19
**Completed:** —

## Goal
Track action usage counts per training step, logging normalized percentages to wandb for both generated actions (pre-epsilon) and executed actions (post-epsilon-greedy).

## Scope
- [ ] Collect action counts during rollout (generated vs executed)
- [ ] Track invalid/failed parse counts separately
- [ ] Aggregate counts across all steps in all episodes per training step
- [ ] Log normalized percentages to wandb: `action_pct/generated/<action>`, `action_pct/executed/<action>`, `action_pct/invalid`
- [ ] Add login-node test to verify metrics appear

## Out of Scope
- Per-env-type breakdown (training uses single env type)
- Raw counts (only normalized percentages)
- Probability distributions from logprobs (separate feature in backlog)
- Console logging (wandb only)

## Key Decisions
- Two separate tracking categories: "generated" (what model output) vs "executed" (after epsilon-greedy)
- Invalid parses tracked as separate metric, not per-action
- Normalized to percentages for easier cross-run comparison
- Single aggregate across all envs in batch

## Working Notes
### 2026-01-19 - Feature Started
**Interview summary:**
- Goal: visibility into action distribution balance during training
- Track generated (pre-epsilon) separately from executed (post-epsilon) actions
- Also track invalid/failed parses
- Wandb logging only, normalized percentages
- Single env type per training run, no need for per-env breakdown
