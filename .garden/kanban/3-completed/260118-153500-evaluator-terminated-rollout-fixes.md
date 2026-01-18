# Evaluator Terminated Rollout Fixes

**Type:** fix
**Branch:** fix/various-bugfix
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** 2026-01-18

## Goal

Fix MultiEnvEvaluator to properly handle terminated rollouts: stop tracking/counting them after termination instead of continuing to include post-auto-reset generations in metrics.

## Scope

- [x] Add `ever_terminated` mask to track which rollouts have ended (persistent, never resets)
- [x] Fix entropy probing to use `ever_terminated` instead of `active_rollouts`
- [x] Fix token metrics to only count tokens for non-terminated rollouts
- [x] Freeze per-rollout token counts on termination for accurate mean/std

## Out of Scope

- Skipping generation entirely for ended rollouts (would need dynamic batch sizing)
- Using `__SKIP__` action mechanism
- Checkpoint selection script bug (separate card)

## Key Decisions

- Using `ever_terminated` mask approach: simple, minimal code change
- Still generate for all rollouts but don't count terminated ones in metrics
- Single PR combining all 3 related fixes since they share the same root cause

## Working Notes
### 2026-01-18 - Feature Started
Combined 3 related backlog cards:
1. `(fix)-entropy-probing-tracks-restarted-episodes.md` - active_rollouts resets after auto-reset
2. `(fix)-evaluator-generates-for-ended-rollouts.md` - wasted compute (partial fix via metrics)
3. `(fix)-token-metrics-include-post-terminal-generations.md` - metrics polluted

All stem from: VecEnv auto-resets on termination, so `terminated=True` on step k becomes `terminated=False` on step k+1 (new episode). Current code uses instant terminated status; need persistent tracking.

Key file: `verl/trainer/ppo/multi_env_evaluator.py`

### 2026-01-18 - Context from Docs

**From exploration-metrics-definitions.md:**
- Entropy probing metrics: `action_entropy`, `unique_texts_step`, `unique_executed_actions_step`
- Measured at configurable steps (`measure_at_steps`)
- Currently uses `active_rollouts` which resets after auto-reset

**From file-structure-scope.md:**
- `verl/trainer/ppo/multi_env_evaluator.py` - Multi-env evaluation + entropy probing (in scope)
- VecEnv auto-resets terminated episodes

**From wrapper-interface-api.md:**
- VecEnv calls `env.step()` and gets `terminated, truncated` flags
- Auto-reset happens internally: on next step, terminated=False for new episode
- `__SKIP__` action exists but out of scope for this fix

### 2026-01-18 - Implementation Complete

**Core change:** Added `ever_terminated` persistent mask (~30 lines changed)

| Change | Location |
|--------|----------|
| Init `ever_terminated = np.zeros(batch_n, dtype=bool)` | Line 694 |
| Replace `batch_response_n_tokens_last` with `batch_frozen_toks` | Lines 699-703 |
| Token counting uses `~ever_terminated` mask | Lines 798-808 |
| Update `ever_terminated = ever_terminated \| done_mask` after env.step | Lines 854-860 |
| Derive `active_rollouts` from `~ever_terminated` | Lines 858-860 |
| Use `batch_frozen_toks` for per-rollout toks_out | Lines 915-918 |

**Decisions made:**
- Count tokens from terminating step (generation happened before env.step)
- Freeze toks_out at termination for mean/std calculation
- Don't skip generation (out of scope) - just exclude from metrics

**Needs cluster test:** `bash experiments/snake/test_login_node.sh`

**Validation checklist:**
- [ ] Eval completes without crash
- [ ] `tokens_per_rollout` should be lower than before (excludes post-termination gens)
- [ ] `toks_out_mean` should reflect termination-step tokens, not final-global-step tokens
- [ ] Entropy metrics should only measure original trajectory states (not post-reset new episodes)
- [ ] Compare with a baseline run (same config, old code) if available

## Original Notes
See individual cards in backlog for detailed problem descriptions and suggested implementations.
