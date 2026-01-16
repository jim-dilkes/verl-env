"""Output formatters: markdown, csv, json."""

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_FLOAT_PRECISION, DEFAULT_MAX_COL_WIDTH
from .compare import ExperimentSummary


def truncate_string(s: str, max_len: int) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def format_value(v: Any, precision: int = DEFAULT_FLOAT_PRECISION) -> str:
    """Format a value for display."""
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def to_markdown(
    data: pd.DataFrame | dict,
    title: str | None = None,
    max_col_width: int = DEFAULT_MAX_COL_WIDTH,
    precision: int = DEFAULT_FLOAT_PRECISION,
    stable: bool = False,
) -> str:
    """Convert data to markdown table.

    Args:
        data: DataFrame or dict to convert
        title: Optional title (as ## heading)
        max_col_width: Maximum column width before truncation
        precision: Float precision
        stable: Strip volatile fields (timestamps, IDs) for git-stable output

    Returns:
        Markdown string
    """
    lines = []

    if title:
        lines.append(f"## {title}")
        lines.append("")

    if isinstance(data, dict):
        # Convert dict to simple key-value table
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        for k, v in sorted(data.items()):
            if stable and k in ["run_id", "created_at", "_timestamp"]:
                continue
            v_str = truncate_string(format_value(v, precision), max_col_width)
            lines.append(f"| {k} | {v_str} |")
        return "\n".join(lines)

    # DataFrame
    df = data.copy()

    # Strip volatile columns in stable mode
    if stable:
        volatile_cols = ["run_id", "created_at", "_timestamp"]
        df = df.drop(columns=[c for c in volatile_cols if c in df.columns], errors="ignore")

    # Reset index if it's meaningful
    if df.index.name or (not df.index.equals(pd.RangeIndex(len(df)))):
        df = df.reset_index()

    # Sort columns for stability
    if stable:
        df = df.reindex(sorted(df.columns), axis=1)

    # Format values
    for col in df.columns:
        df[col] = df[col].apply(lambda x: truncate_string(format_value(x, precision), max_col_width))

    # Build markdown table
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(lines)


def to_csv(
    data: pd.DataFrame,
    path: Path | None = None,
    sort_columns: bool = True,
) -> str | None:
    """Export DataFrame to CSV.

    Args:
        data: DataFrame to export
        path: Output path (None = return string)
        sort_columns: Sort columns alphabetically

    Returns:
        CSV string if path is None, else None
    """
    df = data.copy()

    if sort_columns:
        # Keep run_id/run_name first, then sort rest
        first_cols = [c for c in ["run_id", "run_name", "group"] if c in df.columns]
        other_cols = sorted(c for c in df.columns if c not in first_cols)
        df = df[first_cols + other_cols]

    if path:
        df.to_csv(path)
        return None
    else:
        return df.to_csv()


def to_json(
    data: pd.DataFrame | dict,
    path: Path | None = None,
    indent: int = 2,
    stable: bool = False,
) -> str | None:
    """Export to JSON.

    Args:
        data: DataFrame or dict to export
        path: Output path (None = return string)
        indent: JSON indentation
        stable: Sort keys for stable output

    Returns:
        JSON string if path is None, else None
    """
    if isinstance(data, pd.DataFrame):
        # Convert to records
        data = data.reset_index().to_dict(orient="records")

    json_str = json.dumps(data, indent=indent, sort_keys=stable, default=str)

    if path:
        path.write_text(json_str)
        return None
    else:
        return json_str


def format_config_diff(
    diff_df: pd.DataFrame,
    format: str = "markdown",
    max_col_width: int = DEFAULT_MAX_COL_WIDTH,
) -> str:
    """Format config diff for display.

    Args:
        diff_df: Output from compare.diff_configs()
        format: "markdown", "csv", or "json"
        max_col_width: Max column width for markdown

    Returns:
        Formatted string
    """
    if format == "markdown":
        return to_markdown(diff_df, title="Config Differences", max_col_width=max_col_width)
    elif format == "csv":
        return to_csv(diff_df)
    elif format == "json":
        return to_json(diff_df)
    else:
        raise ValueError(f"Unknown format: {format}")


def format_metrics_table(
    metrics_df: pd.DataFrame,
    format: str = "markdown",
    highlight_best: bool = True,
    precision: int = DEFAULT_FLOAT_PRECISION,
) -> str:
    """Format metrics comparison table.

    Args:
        metrics_df: Output from compare.compare_metrics()
        format: "markdown", "csv", or "json"
        highlight_best: Bold the best value per column (markdown only)
        precision: Float precision

    Returns:
        Formatted string
    """
    if format == "csv":
        return to_csv(metrics_df)
    elif format == "json":
        return to_json(metrics_df)

    # Markdown with optional highlighting
    df = metrics_df.copy()

    if highlight_best:
        # Find best per numeric column
        from .compare import is_higher_better

        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            higher = is_higher_better(col)
            if higher:
                best_idx = df[col].idxmax()
            else:
                best_idx = df[col].idxmin()

            if pd.notna(best_idx):
                # Mark best value (will be bolded in markdown)
                best_val = df.loc[best_idx, col]
                df.loc[best_idx, col] = f"**{format_value(best_val, precision)}**"

    return to_markdown(df, title="Metrics Comparison", precision=precision)


def format_experiment_summary(
    summary: ExperimentSummary,
    format: str = "markdown",
) -> str:
    """Format experiment summary.

    Args:
        summary: ExperimentSummary object
        format: "markdown", "json"

    Returns:
        Formatted string
    """
    if format == "json":
        return to_json({
            "total_runs": summary.total_runs,
            "groups": summary.groups,
            "states": summary.states,
            "config_variations": summary.config_variations,
            "best_runs": summary.best_runs,
            "date_range": summary.date_range,
        })

    # Markdown
    lines = [
        "## Experiment Summary",
        "",
        f"**Total runs:** {summary.total_runs}",
        f"**Date range:** {summary.date_range[0]} to {summary.date_range[1]}",
        "",
    ]

    if summary.groups:
        lines.append("### Groups")
        for group, count in sorted(summary.groups.items()):
            lines.append(f"- {group}: {count} runs")
        lines.append("")

    if summary.states:
        lines.append("### Run States")
        for state, count in sorted(summary.states.items()):
            lines.append(f"- {state}: {count}")
        lines.append("")

    if summary.config_variations:
        lines.append("### Config Variations")
        lines.append("Keys that differ across runs:")
        for key in summary.config_variations[:20]:  # Limit to 20
            lines.append(f"- `{key}`")
        if len(summary.config_variations) > 20:
            lines.append(f"- ... and {len(summary.config_variations) - 20} more")
        lines.append("")

    if summary.best_runs:
        lines.append("### Best Runs")
        for metric, run_id in sorted(summary.best_runs.items()):
            lines.append(f"- **{metric}**: {run_id}")
        lines.append("")

    return "\n".join(lines)


def generate_report(
    configs_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    key_metrics: list[str] | None = None,
    title: str = "Experiment Report",
    stable: bool = False,
) -> str:
    """Generate comprehensive markdown report.

    Args:
        configs_df: Run configurations
        summaries_df: Run summaries
        history_df: Optional history data
        key_metrics: Metrics to highlight
        title: Report title
        stable: Strip volatile fields for git-stable output

    Returns:
        Markdown report string
    """
    from .compare import summarize_experiment_history, diff_configs, compare_metrics

    lines = [f"# {title}", ""]

    # Summary section
    summary = summarize_experiment_history(configs_df, summaries_df, key_metrics)
    lines.append(format_experiment_summary(summary))
    lines.append("")

    # Config diff section
    diff_df = diff_configs(configs_df, only_differing=True)
    if not diff_df.empty:
        lines.append(format_config_diff(diff_df))
        lines.append("")

    # Metrics comparison
    if key_metrics:
        metrics_df = compare_metrics(summaries_df, metrics=key_metrics)
    else:
        metrics_df = compare_metrics(summaries_df)

    if not metrics_df.empty:
        lines.append(format_metrics_table(metrics_df))
        lines.append("")

    # Footer
    if not stable:
        from datetime import datetime
        lines.append("---")
        lines.append(f"*Generated: {datetime.now().isoformat()}*")

    return "\n".join(lines)
