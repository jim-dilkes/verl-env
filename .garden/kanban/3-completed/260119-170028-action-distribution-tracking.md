# Action Distribution Tracking

**Type:** feat
**Branch:** feat/action_distribution_tracking
**Created:** 2026-01-19
**Started:** 2026-01-19
**Completed:** 2026-01-19

## Goal
Track action usage counts per training step, logging normalized percentages to wandb for both generated actions (pre-epsilon) and executed actions (post-epsilon-greedy).

## Scope
- [x] Collect action counts during rollout (generated vs executed)
- [x] Track invalid/failed parse counts separately
- [x] Aggregate counts across all steps in all episodes per training step
- [x] Log normalized percentages to wandb: `action_pct/generated/<action>`, `action_pct/executed/<action>`, `action_pct/invalid`
- [x] Verify with existing login-node test

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

### 2026-01-19 - Context from Docs

**From wrapper-interface-api.md:**
- `extract_action()` returns 5-tuple: `(full_action, extracted_action, valid_action, is_valid, metrics)`
- `extracted_action` = what model generated (pre-validation)
- `valid_action` = what gets executed (after default fallback)
- VecEnv sets `info["executed_action_text"]` after step
- Key file: `verl/envs/vec_env.py` - worker() calls extract_action

**From epsilon-retokenization-onpolicy.md:**
- Epsilon exploration happens in `vec_env.py` worker()
- When epsilon triggers: `info["epsilon_explored"] = True`
- `info["executed_action_text"]` = actual action taken (may differ from generated)
- Key files:
  - `verl/envs/vec_env.py` - epsilon logic + extract_action call
  - `verl/trainer/ppo/ray_multistep_trainer.py` - rollout + metrics aggregation

**From exploration-metrics-definitions.md:**
- Existing metrics: `valid_action_ratio`, `valid_actions_total`, `attempted_actions_total`
- Similar pattern: aggregate counts over rollout, log to wandb
- File: `multi_env_evaluator.py` has metric aggregation patterns

**Implementation approach:**
- Collect counts in vec_env.py during rollout (per-worker)
- Aggregate across workers in trainer after rollout completes
- Normalize and log to wandb alongside existing metrics

### 2026-01-19 - Implementation Complete

**Changed:** `verl/envs/vec_env.py` lines 370-388

**Approach:** Use indicator variables (0.0/1.0) per action. When trainer averages these across all steps, the result is the percentage.

**Metrics added:**
- `action_pct/generated/{action}` - 1.0 when model generated this action (pre-epsilon), 0.0 otherwise
- `action_pct/executed/{action}` - 1.0 when this action was executed (post-epsilon), 0.0 otherwise
- `action_pct/invalid` - 1.0 when parse failed, 0.0 otherwise

**Guards for cross-env compatibility:**
- Check `__len__` before materializing action space (avoids performance trap with large/infinite iterables)
- Skip per-action tracking if action space > 20 or has no `__len__` (log warning once per worker)
- Sanitize action names: `/`, `[`, `]` → `_` (collision risk accepted for small action spaces)
- Epsilon + non-sequence action space: RuntimeError (fail fast on incompatible config)

**Verification:** Run `bash experiments/snake/test_login_node.sh`, then check log for `action_pct/` metrics.

### 2026-01-19 - Feature Complete

Added action distribution tracking to `vec_env.py`. Metrics logged to wandb:
- `action_pct/generated/{action}` - model output distribution
- `action_pct/executed/{action}` - actual execution distribution
- `action_pct/invalid` - parse failure rate

Guards for cross-env compatibility (TextWorld, WebShop). Fail-fast on epsilon + incompatible action space.
