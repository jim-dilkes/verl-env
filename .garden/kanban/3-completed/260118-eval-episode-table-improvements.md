# Eval Episode Table Improvements

**Type:** feat
**Branch:** fix/overcooked-decision-extract
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** 2026-01-18

## Goal
Improve wandb episode table with truncation tracking and working score/reward columns.

## Scope
- [x] Add `max_length_steps` column - list of step IDs where input hit max_seq_len
- [x] Fix score column - use total_reward as fallback when env doesn't return score
- [x] Add `total_reward` column to episode_data

## Out of Scope
- Modifying individual env info dicts
- Decoding truncated input text (too expensive)

## Key Decisions
- Detect truncation by checking if tokenized input length == max_seq_len
- Use `rew_of_traj[0]` (first rollout's cumulative reward) for total_reward
- Score fallback: if score_of_traj is None, use rew_of_traj as score

## Working Notes
### 2026-01-18 - Feature Started
**Current state:**
- `episode_data` has: inputs, outputs, total_score
- `total_score` is always N/A for overcooked because it doesn't return "score" in info
- Only FastSnake and Crafter return `info["score"]`

**Implementation plan:**
- Line 721-727: After tokenizing, check if attention_mask sum equals max_seq_len
- Track step_idx when truncation detected
- Line 1062-1068: Add max_length_steps and total_reward to episode_data
- If score None, fallback to reward

### Implementation Complete
**Files modified:**
1. `verl/trainer/ppo/multi_env_evaluator.py`:
   - Added `episode_total_reward` and `episode_max_length_steps` tracking (line 615-616)
   - Detect truncation when `attention_mask.sum() >= max_seq_len` (line 758-760)
   - Set `episode_total_reward = rew_of_traj[0]` when episode ends (line 847)
   - Extended `episode_data` dict with `total_reward` and `max_length_steps` (line 1072-1078)
   - Extended sample tuple to 5 elements in `_maybe_log_episode_generation` (line 464-465)

2. `verl/utils/tracking.py` - Updated all logger backends:
   - `_log_generations_to_wandb`: Detect sample length, add reward/max_length_steps columns
   - `log_generations_to_swanlab`: Same pattern
   - `log_generations_to_mlflow`: Add new fields to JSON
   - `log_generations_to_clearml`: Add new fields to table dict
   - `log_generations_to_tensorboard`: Add new fields to text output

**Backwards compatible:** Old 3-tuple format still works for other callers.

**Renamed:** `truncated_steps` → `max_length_steps` to clarify it tracks "hit max_seq_len" not strict truncation.

### 2026-01-18 - Feature Complete
Added `max_length_steps` and `total_reward` columns to eval episode tables. Score falls back to reward when env doesn't provide it. Fixed type safety issue with rew_of_traj indexing. All logger backends updated with backward compatibility for 3-tuple samples.
