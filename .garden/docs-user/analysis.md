# `analysis/`: WandB experiment analysis + CLI

This repo includes a small “analysis toolkit” under `analysis/` for pulling runs from Weights & Biases (WandB), caching the results locally, and generating human-readable tables/reports.

It's designed for:
- Listing runs with simple filters (group/tag/state/name)
- Comparing configs across runs
- Comparing summary metrics across runs
- Aggregating metrics by group (mean ± std across seeds)
- Exporting learning curves from run history to CSV
- Generating a single markdown "report" across a set of runs

List of metrics and their definitions can be found in .garden/docs-user/wandb-metrics.md

## Quickstart

Prereqs:
- You must be able to access the project in WandB (typically `wandb login`, or set `WANDB_API_KEY`).
- Python env must have `numpy`, `pandas`, `pyarrow`, and `wandb` available (cache files are parquet).

Run the CLI:
```bash
python -m analysis --help
# or:
python -m analysis.cli --help
```

List runs:
```bash
python -m analysis list-runs --project entity/project
```

Generate a markdown report for a group:
```bash
python -m analysis report --project entity/project --group "my-group" --output report.md
```

## How the code is organized

### Entry points

- `analysis/__main__.py` lets you run the package directly:
  - `python -m analysis ...` calls `analysis.cli.main()`
- `analysis/cli.py` implements the Click CLI.

### `analysis/fetch.py`: WandB API wrapper + caching

The core is `WandBFetcher`, a thin wrapper around `wandb.Api()` that:
- Fetches runs (`fetch_runs`) and converts them into lightweight `RunInfo` records
- Fetches:
  - configs (`fetch_configs`)
  - summary metrics (`fetch_summaries`) — values from `run.summary`
  - history (`fetch_history`) — values from `run.history()`
- Caches results to `analysis/data/` as parquet files
- Adds gentle rate limiting between API calls (`API_CALL_DELAY`)

Caching details:
- Cache directory defaults to `analysis/data/`.
- Cache filenames include the project name and a short hash of request parameters. Examples:
  - `<project>_configs_<hash>.parquet`
  - `<project>_summaries_<hash>.parquet`
  - `<project>_history_<hash>.parquet`
- Most CLI commands accept `--refresh` to bypass cache and re-fetch from WandB.

Config handling details:
- WandB configs can be nested (dicts) or sometimes JSON-encoded strings.
- `flatten_config()` turns nested configs into dot keys (e.g. `trainer.test_freq`).
- `filter_config_keys()`:
  - drops keys with noisy prefixes like `_wandb`, `_timestamp`, etc.
  - optionally allowlists keys (defaults come from `analysis/config.py`).

Metadata enrichment:
- `fetch_configs(..., include_metadata=True)` tries to download `wandb-metadata.json` for each run and adds:
  - `hostname`
  - `gpu_type`

### `analysis/config.py`: defaults

This file holds defaults used by both fetch + CLI:
- `DEFAULT_CONFIG_ALLOWLIST`: which config keys to keep when comparing configs
- `DEFAULT_METRIC_PATTERNS`: glob patterns used by `compare-metrics` when you don’t pass explicit metrics
- `HIGHER_IS_BETTER`: heuristics for which metrics should be maximized vs minimized
- Cache + formatting constants

### `analysis/compare.py`: diffs, comparisons, curves

Main helpers:
- `diff_configs(configs_df)`: returns a DataFrame of keys that vary across runs
- `compare_metrics(summaries_df, ...)`: builds a table of metric columns from run summaries
- `aggregate_by_group(summaries_df, ...)`: aggregates metrics by group (mean ± std)
- `diff_configs_between_groups(configs_df)`: config differences between groups (using mode per group)
- `extract_learning_curves(history_df, metrics, ...)`:
  - pivots long-form history into per-metric wide tables
  - optional resampling to a fixed number of x-axis points
- `summarize_experiment_history(configs_df, summaries_df, ...)`:
  - counts runs by group and state
  - finds config keys that vary
  - optionally identifies "best run" per key metric

### `analysis/export.py`: output formatting

`export.py` provides:
- `to_markdown(df_or_dict)`
- `to_csv(df)`
- `to_json(df_or_dict)`

…and higher-level formatters used by the CLI:
- `format_config_diff(...)`
- `format_metrics_table(...)` (optionally bolds best numeric values in markdown)
- `format_experiment_summary(...)`
- `format_group_summary(...)` formats group-aggregated metrics with config diff
- `generate_report(...)` to assemble a multi-section markdown report

## CLI usage (command reference)

All commands live under the top-level group:
```bash
python -m analysis --help
```

### `list-runs`
List runs in a WandB project with filters.

```bash
python -m analysis list-runs --project entity/project \
  --group "FS_PPO_*" \
  --tag baseline \
  --state finished \
  --format markdown
```

Useful options:
- `--group` can be repeated; supports wildcards
- `--tag` can be repeated; run must contain *all* tags
- `--name` is a regex applied to run name
- `--format` one of `markdown|csv|json`
- `--output` write to a file (use `-` for stdout)

### `diff-configs`
Compare configs across runs.

```bash
# Compare all runs in a group
python -m analysis diff-configs --project entity/project --group "my-group"

# Compare specific runs (by ID or name)
python -m analysis diff-configs --project entity/project --run abc123 --run def456
```

Useful options:
- `--only-differing/--all` controls whether equal keys are hidden
- `--allowlist` is a comma-separated list of config keys to include
- `--refresh` bypass cache

### `compare-metrics`
Compare summary metrics across runs.

```bash
# Default metric patterns
python -m analysis compare-metrics --project entity/project --group "my-group"

# Explicit metrics
python -m analysis compare-metrics --project entity/project --group "my-group" \
  --metric "eval_task/rewards_mean" \
  --metric "generation/success_rate"

# Add pattern(s)
python -m analysis compare-metrics --project entity/project --group "my-group" \
  --pattern "eval_*/rewards_mean" \
  --pattern "val/*/rewards_mean"
```

Notes:
- This command reads from `run.summary` (i.e., whatever WandB reports as the run's summary), not the full time-series history.
- The CLI has a `--final` option, but today the command uses summary values directly (it does not compute "best"/"last" from history).

### `group-summary`
Aggregate metrics by WandB group (mean ± std across seeds within each group).

```bash
# Default metric patterns
python -m analysis group-summary --project entity/project

# Specific patterns
python -m analysis group-summary --project entity/project \
  --pattern "eval_*/rewards_mean" \
  --pattern "actor/entropy"

# Exclude certain groups
python -m analysis group-summary --project entity/project \
  --exclude-group "test_*" \
  --exclude-group "debug_*"

# Output to CSV
python -m analysis group-summary --project entity/project \
  --format csv --output summary.csv
```

Useful options:
- `--group` filter to specific groups (supports wildcards)
- `--exclude-group` / `-x` exclude groups matching pattern
- `--pattern` metric glob patterns (default: eval_*/rewards_mean, actor/entropy, etc.)
- `--show-config-diff/--no-config-diff` include config differences between groups
- `--format` one of `markdown|csv|json`
- `--state` filter by run state (default: `finished`)

Output structure:
- **Markdown**: Shows config differences section (keys that vary between groups) followed by metrics table with "mean ± std" columns
- **CSV**: Separate `_mean` and `_std` columns for each metric
- **JSON**: Dict with `config_diff` and `metrics` arrays

Notes:
- Assumes WandB `group` field corresponds to "same experiment, different seeds"
- Uses sample std (ddof=1)
- Runs with missing/null group are aggregated under "(no-group)"
- Config diff uses mode (most common value) per group, which hides within-group variation

### `curves`
Extract learning curves from run history to CSV.

```bash
python -m analysis curves --project entity/project --group "my-group" \
  --metric "train/loss" --metric "eval_task/rewards_mean" \
  --x-axis _step \
  --sample-rate 10 \
  --resample-to 200 \
  --output curves.csv
```

Behavior:
- Fetches history for the requested metrics + x-axis.
- Produces one CSV per metric by appending `_<metric>` to the output file stem.
  - Example: `curves_train_loss.csv`, `curves_eval_task_rewards_mean.csv`

### `report`
Generate a multi-section markdown report (summary + config diff + metrics table).

```bash
python -m analysis report --project entity/project --group "my-group" \
  --title "My Experiment Report" \
  --metric "eval_task/rewards_mean" \
  --metric "generation/success_rate" \
  --output report.md
```

Useful options:
- `--stable` removes volatile fields/timestamps to make output diff-friendly for git
- `--metric` restricts the metrics shown/used for “best runs” identification

### `history`
Generate a high-level overview across *all* runs in a project.

```bash
python -m analysis history --project entity/project \
  --metric "eval_task/rewards_mean" \
  --metric "generation/success_rate" \
  --output history.md
```

Output includes:
- run counts by group and by state
- config keys that differ
- best run IDs for provided metrics

(Implementation note: `--group-by` is present in the CLI but currently not used when building the summary.)

### `clear-cache`
Remove cached parquet files for a project.

```bash
python -m analysis clear-cache --project entity/project
```

This deletes files in `analysis/data/` that match `<project>_*`.

## Python API usage

You can use the toolkit directly from Python.

```python
from analysis import WandBFetcher, diff_configs, compare_metrics

fetcher = WandBFetcher(entity="entity", project="project", verbose=True)

runs = fetcher.fetch_runs(groups=["my-group"], tags=["baseline"], states=["finished"])
configs_df = fetcher.fetch_configs(runs)
summaries_df = fetcher.fetch_summaries(runs)

diff_df = diff_configs(configs_df)
metrics_df = compare_metrics(summaries_df, metric_patterns=["eval_*/rewards_mean"])
```

## Troubleshooting / gotchas

- Authentication: if you see permission errors, run `wandb login` or export `WANDB_API_KEY`.
- “Project not found”: use the full `entity/project` form for `--project`.
- Cache confusion: add `--refresh` to any command or run `clear-cache`.
- Performance: fetching history can be slow; use fewer runs, fewer metrics, and/or increase `--sample-rate`.
- Missing columns: if a metric isn’t present in run summaries/history, it will be skipped.
