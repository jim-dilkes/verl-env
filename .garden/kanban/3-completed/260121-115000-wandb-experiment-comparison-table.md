# WandB Experiment Comparison Table Tool

**Type:** feat
**Branch:** analysis
**Created:** 2026-01-21
**Started:** 2026-01-21
**Completed:** 2026-01-21

## Goal
Build a CLI tool to generate comprehensive comparison tables across all WandB experiment groups, showing performance metrics and auto-detected config differences.

## Scope
- [x] Connect to WandB API and fetch all runs from project
- [x] Group runs by WandB `group` field
- [x] Auto-detect config keys that differ between groups
- [x] Extract final-step metrics (eval_*, val_*, exploration, learning)
- [x] Aggregate across seeds (mean ± std)
- [x] Handle runs that ended slightly early (via WandB summary's last-logged values)
- [x] Output markdown table to console
- [x] Save CSV file
- [x] Filter/exclude groups (optional CLI arg)

## Out of Scope
- Plots/visualizations
- Real-time monitoring / live updates
- Publication-quality formatting
- Handling completely failed/crashed runs

## Key Decisions
- Use WandB `group` field as primary grouping mechanism
- Auto-detect config differences rather than requiring manual specification
- Final metrics from `run.summary` (WandB's last-logged values), not computed from history
- Mean ± std aggregation across seeds within each group
- Both markdown + CSV output
- Primary metrics: `eval_*` (reward, score), `val_*`, exploration (entropy, coverage), learning (loss, grad_norm)
- Environment-aware comparisons delegated to group naming conventions

## Working Notes
### 2026-01-21 - Feature Started
**Interview summary:**
- Analysis tool for comparing experiment groups across WandB
- Purpose: guide research direction, identify promising approaches
- All groups included by default, with option to exclude
- Config diffs auto-detected
- Metrics: eval_reward, eval_score, val_* metrics, exploration (entropy, coverage), learning dynamics (loss, grad)
- Aggregation: mean ± std per group
- Output: markdown table (console) + CSV file
- Not for publication, just internal research guidance

### 2026-01-21 - Context from Docs

**From exploration-metrics-definitions.md:**
Key metrics to include:
- `n_distinct_state_actions_valid` - state/action coverage
- `distinct_state_actions_per_frame` - normalized coverage
- `action_entropy` - Shannon entropy over completions
- `unique_texts_step`, `unique_executed_actions_step` - diversity measures
- `valid_action_ratio` - validity tracking

**From experiment-configs-params.md:**
Key config params that likely vary between groups:
- `envs.env_name` (fastsnake, overcooked, babyai)
- `envs.captioner.type` (naive, cot, multi_action)
- `algorithm.adv_estimator` (GAE, GRPO, RLOO)
- `prompt.prompt.multi_action_reasoning`
- `prompt.prompt.epsilon` (epsilon-greedy)
- entropy_coefficient

**From project-research-focus.md:**
Research context - comparing:
- Token entropy vs action entropy
- Loss-based vs covariance-based entropy regularization
- Model scaling effects
- Memory-free multi-step RL

**From Experiment Consolidation (Jan 16):**
Experiments span phases:
1. Nov 2025 - baseline setup, performance tuning
2. Nov 25 - entropy HP sweep (Hpt001, Hpt005, 150tok, clipcov, klcov)
3. Jan 2026 - multi-action experiments

Key experiment groups:
- Snake 4B: eps0, eps0.2
- Overcooked 4B: eps0, eps0.2
- Overcooked 14B NoThink: eps0, eps0.2

Eval prefixes: `-Greedy`, `MA-*-Greedy`, `-Entropy-Check`, `-StateVisitation`

**From "Where we've been" (Jan 20):**
Results context:
- PPO outperforms GRPO on same base model
- Entropy regularization tradeoff: train perf vs generalization
- 1-step memory improves generalization
- Comparing across: snake, overcooked, babyai (planned)

**CRITICAL: Existing `analysis/` toolkit discovered**
- Repo already has `analysis/` CLI with:
  - `list-runs`, `diff-configs`, `compare-metrics`, `curves`, `report`, `history`
  - WandB API wrapper + caching (`analysis/fetch.py`)
  - Config diff logic (`analysis/compare.py`)
  - Markdown/CSV export (`analysis/export.py`)
- Default metric patterns in `analysis/config.py`:
  - `eval_*/rewards_mean`, `eval_*/score_mean`, `eval_*/traj_length_mean`
  - `generation/success_rate`, `val/*/rewards_mean`
- **Scope adjustment**: Likely extend existing toolkit rather than build from scratch

**From wandb-metrics.md / wandb-metrics-audit.md:**
Comprehensive metric reference:
- Actor: `pg_loss`, `pg_clipfrac`, `ppo_kl`, `entropy`, `grad_norm`, `lr`
- Critic: `vf_loss`, `vf_clipfrac`, `vpred_mean`, `grad_norm`
- Batch stats: `critic/score/*`, `critic/rewards/*`, `critic/advantages/*`
- Response: `response_length/*`, `prompt_length/*`
- Timing: `timing_s/*`, `perf/*`
- Eval: `eval_{name}/rewards_mean`, `eval_{name}/score_mean`, `eval_{name}/traj_length_mean`
- Exploration: `eval_{name}/action_entropy_mean`, `eval_{name}/n_distinct_state_actions_valid_mean`
- Validity: `eval_{name}/valid_action_ratio`

**Known issue from audit:**
- `score_mean/std` can miss terminal-step scores (bug in multi_env_evaluator.py)

**Output workflow:**
- Write analysis findings to PhD Log Book: `/Users/jim/Documents/PhD/Research Projects/4) Exploration in SDM for LLMs/Log Book/`
- Use date format: `YYYY-MM-DD - <descriptive title>.md`

### 2026-01-21 - Implementation Complete

**New CLI command:** `python -m analysis group-summary`
- Options: `-p/--project`, `-g/--group`, `-x/--exclude-group`, `-m/--metric`, `--pattern`, `--show-config-diff`
- Outputs: markdown table (console) or CSV file

**Code changes:**
- `analysis/compare.py`: Added `aggregate_by_group()`, `diff_configs_between_groups()`
- `analysis/export.py`: Added `format_group_summary()`
- `analysis/cli.py`: Added `group-summary` command
- `analysis/config.py`: Extended default metric patterns (exploration, learning)
- `analysis/fetch.py`: Fixed parquet serialization for mixed-type columns

**Tested on:** jimdilkes/verl_env (122 runs, 44 groups)

**Log book entries:**
- `2026-01-21 - WandB Group Analysis.md` - Initial tool output
- `2026-01-21 - WandB Group Analysis v2.md` - Focused metrics table
- `2026-01-21 - Deep Analysis.md` - Comprehensive analysis with findings

### 2026-01-21 - Review Fixes

Following code review, hardened the implementation:
- Fixed no-group selection semantics (normalize at source in fetch_runs)
- Added numeric coercion warning when >50% values coerced to NaN
- Hardened parquet serialization for unknown object types
- Cleaned up imports (normalize_group now public in fetch.py)
- Updated card to clarify "final metrics" = WandB summary values

**Commit:** `55623a44` - fix(analysis): harden group-summary with proper normalization and type safety
