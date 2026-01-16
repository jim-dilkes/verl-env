# Auto WandB Analysis

**Type:** feat
**Branch:** feat/auto-wandb-analysis
**Created:** 2026-01-16
**Started:** 2026-01-16
**Completed:** —

## Goal
Build Python tooling to automate wandb experiment analysis - compare runs, query by config/metrics, generate text-friendly summaries for both human review and Claude Code consumption.

## Scope
- [x] Core wandb API wrapper for fetching runs by project/filters
- [x] Config diff utility - show what hyperparams differ between runs
- [x] Final metrics comparison table (reward/score across runs)
- [x] Learning curve extraction (key metrics over training steps)
- [x] CLI interface for common queries
- [x] Markdown report generation for experiment summaries
- [x] "Experiment history" overview - what we've tried, outcomes, trajectory

## Out of Scope
- Visualization/charts (text-only output)
- Auto-alerting/notifications
- Live monitoring of in-progress runs
- Complex database/caching (allow simple local cache files like CSV/Parquet with a manual refresh flag)

## Key Decisions
- Location: `analysis/` dir in verl repo
- Data flow: Fresh wandb imports → local CSVs → analysis
- Output formats: CLI tables + markdown + LLM-friendly text
- Dual purpose: Human use + Claude Code tooling for future sessions

## Assumptions / Open Questions
- What is the canonical identifier set for a run? (e.g., `entity/project/run_id`, plus `group`, `job_type`, tags)
- What does “final metric” mean for comparisons: last logged value, best value, or value at a fixed step?
- What is the x-axis for learning curves: `_step`, `global_step`, env frames, or wall-clock?
- Which config keys are “signal” vs noise (wandb injects many `_` and system fields); do we maintain an allowlist?
- Do we require consistent run naming/tagging conventions to make the “experiment history” meaningful?

## Output Contract (for humans + Claude)
- Prefer deterministic output ordering (sort keys, stable column order) and minimal ANSI by default
- Support at least one machine-readable format (`json`/`csv`) in addition to markdown
- Ensure long text fields (notes/config blobs) are truncated with explicit `--max-chars` / `--no-trunc`

## Acceptance Criteria
- Can list runs by filters (project, tags, regex on name) without timing out on large projects
- Can diff configs across N runs with clear “only these keys differ” output
- Can generate a comparison table for chosen metrics with explicit definition of “final”
- Can export learning curves for selected keys with downsampling and missing-data handling
- Can generate a single markdown report that is stable enough to diff in git

## Edge Cases to Handle
- Crashed/aborted runs, missing summary keys, and metrics that appear late in training
- Mixed logging step keys (`_step` vs `global_step`) across runs
- Runs with nested configs / lists; need a consistent flattening strategy
- Rate limits / pagination: avoid fetching full history unless requested

## Working Notes
### 2026-01-16 - Feature Started
Interview summary:
- Primary use case: post-hoc analysis of completed experiments
- Need both run comparison and experiment tracking queries
- Outputs for terminal (quick checks) and markdown (notes/sharing)
- Key insight: should be usable by Claude Code to pull data and summarize experiments
- Metrics focus: config diffs, final scores, learning curves

### 2026-01-16 - Context from Docs

**From training/experiment-configs-params.md:**
Key config params to track for diffs:
- `envs.env_name`, `envs.task`, `envs.n_rollouts`
- `envs.captioner.type` (naive/cot/multi_action), `envs.captioner.max_text_history`
- `prompt.prompt.epsilon` (exploration), `prompt.prompt.multi_action_reasoning`
- `algorithm.adv_estimator` (GAE/GRPO/RLOO/REINFORCE_PLUS_PLUS)
- `algorithm.step_gamma/lam`, `algorithm.token_gamma/lam`
- `data.train_batch_size`, `trainer.critic_warmup`, `trainer.test_freq`
- Active conditions: Hpt001, Hpt005, 150tok, clipcov, klcov

**From training/exploration-metrics-definitions.md:**
Key metrics to extract:
- State visitation: `n_distinct_state_actions_valid`, `distinct_state_actions_per_frame`
- Action diversity: `action_entropy`, `unique_executed_actions_step`, `valid_action_ratio`
- Validity: `valid_actions_total`, `attempted_actions_total`

**From research/project-research-focus.md:**
Research questions that shape comparisons:
- Token entropy vs action entropy (high token diversity ≠ diverse actions)
- Entropy regularization: loss-based vs covariance-based (clip-cov, kl-cov)
- Model scaling: 0.5B → 4B → 7B → 14B
- Comparison papers: RAGEN, ARPO, Search-R1, LOOP

**From codebase/file-structure-scope.md:**
- Experiment configs in: `experiments/BAI/`, `experiments/snake/`, `experiments/webshop/`
- Training code: `verl/trainer/ppo/ray_multistep_trainer.py`
- Metrics collection: `verl/trainer/ppo/multi_env_evaluator.py`

### 2026-01-16 - Implementation Plan

**Module structure:**
- `analysis/fetch.py` - WandBFetcher class with caching, rate limiting
- `analysis/compare.py` - diff_configs, compare_metrics, extract_learning_curves
- `analysis/export.py` - markdown/csv/json formatters, report generation
- `analysis/cli.py` - click-based CLI with commands: list-runs, diff-configs, compare-metrics, curves, report, history
- `analysis/config.py` - allowlists, defaults, HIGHER_IS_BETTER dict

**Key design decisions:**
- RunInfo dataclass to avoid keeping full wandb.Run objects in memory
- Cache to parquet files in `analysis/data/` (gitignored)
- Dot notation for config flattening (matches wandb flat view)
- `--final` modes: last (default), best, step:N
- `--stable` flag strips volatile fields for git-diffable output

**Implementation order:** config → fetch → compare → export → cli

Full plan in `.brisk/scratchpad/implementation-plan.md`

### 2026-01-16 - Implementation Complete

**Files created:**
- `analysis/__init__.py` - package exports
- `analysis/__main__.py` - `python -m analysis` entry point
- `analysis/config.py` - constants, allowlists, defaults
- `analysis/fetch.py` - WandBFetcher class with caching
- `analysis/compare.py` - diff_configs, compare_metrics, extract_learning_curves
- `analysis/export.py` - markdown/csv/json formatters
- `analysis/cli.py` - click-based CLI (7 commands)
- `analysis/data/.gitignore` - ignore cache files

**CLI commands:**
```bash
python -m analysis list-runs --project entity/project --group "pattern*"
python -m analysis diff-configs --project entity/project --run r1 --run r2
python -m analysis compare-metrics --project entity/project --pattern "eval_*/rewards_mean"
python -m analysis curves --project entity/project --metric "eval_snake/rewards_mean" -o curves.csv
python -m analysis report --project entity/project --output report.md
python -m analysis history --project entity/project
python -m analysis clear-cache --project entity/project
```

**Unit tests pass.** Real wandb test blocked by API auth in non-interactive shell - needs `wandb login` or WANDB_API_KEY env var.

**Resolved open questions:**
- Run identifier: `entity/project/run_id` + group, tags, name
- Final metric: `--final` flag with modes: last (default), best, step:N
- X-axis: configurable via `--x-axis`, default `_step`
- Config allowlist: DEFAULT_CONFIG_ALLOWLIST in config.py, IGNORE_CONFIG_PREFIXES for noise
- Flattening: dot notation (e.g., `algorithm.step_gamma`)
