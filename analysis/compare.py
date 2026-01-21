"""Config diff, metrics comparison, and learning curve extraction."""

import fnmatch
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .config import HIGHER_IS_BETTER, IGNORE_CONFIG_PREFIXES


def diff_configs(
    configs_df: pd.DataFrame,
    ignore_prefixes: list[str] | None = None,
    only_differing: bool = True,
) -> pd.DataFrame:
    """Show config differences across runs.

    Args:
        configs_df: DataFrame with run_id as index, config keys as columns
        ignore_prefixes: Config key prefixes to ignore
        only_differing: Only show keys that differ across runs

    Returns:
        DataFrame with columns: key, run_1, run_2, ..., differs
    """
    if ignore_prefixes is None:
        ignore_prefixes = IGNORE_CONFIG_PREFIXES

    # Get config columns (exclude metadata)
    metadata_cols = ["run_name", "group", "tags", "state", "created_at", "hostname", "gpu_type"]
    config_cols = [c for c in configs_df.columns if c not in metadata_cols]

    # Filter by prefix
    config_cols = [
        c for c in config_cols
        if not any(c.startswith(p) for p in ignore_prefixes)
    ]

    # Build comparison records
    records = []
    for col in sorted(config_cols):
        values = configs_df[col].tolist()
        unique_values = set(str(v) for v in values if pd.notna(v))
        differs = len(unique_values) > 1

        if only_differing and not differs:
            continue

        record = {"key": col, "differs": differs}
        for i, (run_id, val) in enumerate(zip(configs_df.index, values)):
            record[run_id] = val
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        # Reorder columns: key, differs, then run columns
        run_cols = [c for c in df.columns if c not in ["key", "differs"]]
        df = df[["key", "differs"] + run_cols]

    return df


def find_matching_keys(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    """Find column names matching glob patterns.

    Args:
        df: DataFrame to search
        patterns: List of glob patterns (e.g., "eval_*/rewards_mean")

    Returns:
        List of matching column names
    """
    matched = set()
    for pattern in patterns:
        for col in df.columns:
            if fnmatch.fnmatch(col, pattern):
                matched.add(col)
    return sorted(matched)


def get_final_value(
    series: pd.Series,
    mode: str = "last",
    higher_is_better: bool = True,
) -> float:
    """Extract final value from a series based on mode.

    Args:
        series: Metric values over time
        mode: "last", "best", or "step:N"
        higher_is_better: For "best" mode, whether to maximize

    Returns:
        Final value
    """
    series = series.dropna()
    if series.empty:
        return float("nan")

    if mode == "last":
        return series.iloc[-1]
    elif mode == "best":
        return series.max() if higher_is_better else series.min()
    elif mode.startswith("step:"):
        step = int(mode.split(":")[1])
        # Find closest step
        if "_step" in series.index.names:
            closest = series.index.get_loc(step, method="nearest")
            return series.iloc[closest]
        else:
            # Assume index is step
            if step < len(series):
                return series.iloc[step]
            return series.iloc[-1]
    else:
        raise ValueError(f"Unknown final mode: {mode}")


def is_higher_better(metric_name: str) -> bool:
    """Determine if higher values are better for a metric."""
    for suffix, higher in HIGHER_IS_BETTER.items():
        if metric_name.endswith(suffix):
            return higher
    # Default: assume higher is better
    return True


def compare_metrics(
    summaries_df: pd.DataFrame,
    metrics: list[str] | None = None,
    metric_patterns: list[str] | None = None,
    final_mode: str = "last",
) -> pd.DataFrame:
    """Compare final metrics across runs.

    Args:
        summaries_df: DataFrame with run summaries (run_id as index)
        metrics: Specific metric names to compare
        metric_patterns: Glob patterns for metrics (e.g., "eval_*/rewards_mean")
        final_mode: How to determine final value: "last", "best", "step:N"

    Returns:
        DataFrame with run_id/run_name and metric columns
    """
    # Find metrics to compare
    if metrics is None:
        metrics = []
    if metric_patterns:
        metrics = list(set(metrics + find_matching_keys(summaries_df, metric_patterns)))

    if not metrics:
        # Default to all numeric columns
        metrics = summaries_df.select_dtypes(include="number").columns.tolist()

    # Filter to existing columns
    metrics = [m for m in metrics if m in summaries_df.columns]

    # Build result
    result = summaries_df[["run_name", "group"]].copy() if "run_name" in summaries_df.columns else pd.DataFrame(index=summaries_df.index)

    for metric in sorted(metrics):
        result[metric] = summaries_df[metric]

    return result


def extract_learning_curves(
    history_df: pd.DataFrame,
    metrics: list[str],
    x_axis: str = "_step",
    resample_to: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract learning curves per metric.

    Args:
        history_df: DataFrame with history data (must have run_id column)
        metrics: Metric names to extract
        x_axis: Column to use as x-axis
        resample_to: If set, resample to this many uniform points

    Returns:
        Dict mapping metric name -> DataFrame with x and run columns
    """
    curves = {}

    for metric in metrics:
        if metric not in history_df.columns:
            continue

        # Pivot to get runs as columns
        metric_df = history_df[[x_axis, "run_id", metric]].copy()
        metric_df = metric_df.dropna(subset=[metric])

        if metric_df.empty:
            continue

        # Pivot: x_axis as index, run_id as columns
        pivoted = metric_df.pivot_table(
            index=x_axis,
            columns="run_id",
            values=metric,
            aggfunc="mean",
        )

        if resample_to and len(pivoted) > resample_to:
            # Resample to uniform points
            import numpy as np
            new_index = np.linspace(
                pivoted.index.min(),
                pivoted.index.max(),
                resample_to,
            )
            pivoted = pivoted.reindex(
                pivoted.index.union(new_index)
            ).interpolate(method="linear").loc[new_index]

        curves[metric] = pivoted

    return curves


@dataclass
class ExperimentSummary:
    """High-level summary of experiment runs."""

    total_runs: int
    groups: dict[str, int]  # group name -> count
    states: dict[str, int]  # state -> count
    config_variations: list[str]  # keys that vary
    best_runs: dict[str, str]  # metric -> run_id of best
    date_range: tuple[str, str]  # earliest, latest created_at


def summarize_experiment_history(
    configs_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    key_metrics: list[str] | None = None,
) -> ExperimentSummary:
    """Generate high-level experiment overview.

    Args:
        configs_df: Run configurations
        summaries_df: Run summaries
        key_metrics: Metrics to identify best runs for

    Returns:
        ExperimentSummary with aggregated info
    """
    # Count by group
    groups = configs_df["group"].value_counts().to_dict() if "group" in configs_df.columns else {}

    # Count by state
    states = configs_df["state"].value_counts().to_dict() if "state" in configs_df.columns else {}

    # Find config variations
    diff_df = diff_configs(configs_df, only_differing=True)
    config_variations = diff_df["key"].tolist() if not diff_df.empty else []

    # Find best runs per metric
    best_runs = {}
    if key_metrics:
        for metric in key_metrics:
            if metric in summaries_df.columns:
                higher = is_higher_better(metric)
                if higher:
                    best_idx = summaries_df[metric].idxmax()
                else:
                    best_idx = summaries_df[metric].idxmin()
                if pd.notna(best_idx):
                    best_runs[metric] = best_idx

    # Date range
    if "created_at" in configs_df.columns:
        dates = pd.to_datetime(configs_df["created_at"])
        date_range = (dates.min().isoformat(), dates.max().isoformat())
    else:
        date_range = ("", "")

    return ExperimentSummary(
        total_runs=len(configs_df),
        groups=groups,
        states=states,
        config_variations=config_variations,
        best_runs=best_runs,
        date_range=date_range,
    )


def aggregate_by_group(
    summaries_df: pd.DataFrame,
    metrics: list[str] | None = None,
    metric_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metrics by group (mean ± std across seeds).

    Args:
        summaries_df: DataFrame with run summaries (must have 'group' column)
        metrics: Specific metric names to aggregate
        metric_patterns: Glob patterns for metrics

    Returns:
        DataFrame with group as index, columns: n_runs, {metric}_mean, {metric}_std
    """
    if "group" not in summaries_df.columns:
        raise ValueError("summaries_df must have 'group' column")

    # Find metrics to aggregate
    if metrics is None:
        metrics = []
    if metric_patterns:
        metrics = list(set(metrics + find_matching_keys(summaries_df, metric_patterns)))

    if not metrics:
        # Default to all numeric columns
        exclude_cols = ["run_name", "group", "tags", "state", "created_at"]
        metrics = [
            c for c in summaries_df.select_dtypes(include="number").columns
            if c not in exclude_cols
        ]

    # Filter to existing columns
    metrics = [m for m in metrics if m in summaries_df.columns]

    # Group and aggregate
    grouped = summaries_df.groupby("group")

    records = []
    for group_name, group_df in grouped:
        record = {"group": group_name, "n_runs": len(group_df)}
        for metric in metrics:
            values = group_df[metric].dropna()
            if len(values) > 0:
                record[f"{metric}_mean"] = values.mean()
                record[f"{metric}_std"] = values.std() if len(values) > 1 else 0.0
            else:
                record[f"{metric}_mean"] = float("nan")
                record[f"{metric}_std"] = float("nan")
        records.append(record)

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.set_index("group")

    return result


def diff_configs_between_groups(
    configs_df: pd.DataFrame,
    ignore_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Show config differences between groups (not individual runs).

    Takes most common value within each group, then identifies keys
    where groups have different values.

    Args:
        configs_df: DataFrame with run configs (must have 'group' column)
        ignore_prefixes: Config key prefixes to ignore

    Returns:
        DataFrame with columns: key, group_1, group_2, ..., differs
    """
    if ignore_prefixes is None:
        ignore_prefixes = IGNORE_CONFIG_PREFIXES

    if "group" not in configs_df.columns:
        raise ValueError("configs_df must have 'group' column")

    # Get config columns (exclude metadata)
    metadata_cols = ["run_name", "group", "tags", "state", "created_at", "hostname", "gpu_type"]
    config_cols = [c for c in configs_df.columns if c not in metadata_cols]

    # Filter by prefix
    config_cols = [
        c for c in config_cols
        if not any(c.startswith(p) for p in ignore_prefixes)
    ]

    # Get unique groups
    groups = sorted(configs_df["group"].unique())

    # For each group, get the most common value for each config key
    group_configs = {}
    for group in groups:
        group_df = configs_df[configs_df["group"] == group]
        group_config = {}
        for col in config_cols:
            values = group_df[col].dropna()
            if len(values) > 0:
                # Most common value
                group_config[col] = values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0]
            else:
                group_config[col] = None
        group_configs[group] = group_config

    # Build comparison records (only keys that differ between groups)
    records = []
    for col in sorted(config_cols):
        group_values = {g: group_configs[g].get(col) for g in groups}
        unique_values = set(str(v) for v in group_values.values() if v is not None)
        differs = len(unique_values) > 1

        if not differs:
            continue

        record = {"key": col, "differs": differs}
        for group in groups:
            record[group] = group_values[group]
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        # Reorder columns: key, differs, then group columns
        df = df[["key", "differs"] + groups]

    return df
