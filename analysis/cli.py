"""Command-line interface for wandb analysis."""

import sys
from pathlib import Path

import click

from .fetch import WandBFetcher, RunInfo, normalize_group
from .compare import (
    diff_configs,
    compare_metrics,
    extract_learning_curves,
    summarize_experiment_history,
    aggregate_by_group,
    diff_configs_between_groups,
)
from .export import (
    to_markdown,
    to_csv,
    to_json,
    format_config_diff,
    format_metrics_table,
    format_experiment_summary,
    format_group_summary,
    generate_report,
)
from .config import DEFAULT_METRIC_PATTERNS


def parse_project(project: str) -> tuple[str, str]:
    """Parse project string into entity and project name."""
    if "/" in project:
        entity, proj = project.split("/", 1)
        return entity, proj
    return "", project


def output_result(result: str, output: str | None, format: str) -> None:
    """Write result to output destination."""
    if output and output != "-":
        Path(output).write_text(result)
        click.echo(f"Written to {output}")
    else:
        click.echo(result)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """WandB experiment analysis tools."""
    pass


@cli.command("list-runs")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--group", "-g", multiple=True, help="Filter by group (supports wildcards)")
@click.option("--tag", "-t", multiple=True, help="Filter by tag (must have ALL)")
@click.option("--name", "-n", help="Filter by name (regex)")
@click.option("--state", "-s", multiple=True, help="Filter by state (finished, running, crashed)")
@click.option("--format", "-f", type=click.Choice(["markdown", "csv", "json"]), default="markdown")
@click.option("--output", "-o", help="Output file (- for stdout)")
@click.option("--verbose", "-v", is_flag=True)
def list_runs(project, group, tag, name, state, format, output, verbose):
    """List runs matching filters."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    runs = fetcher.fetch_runs(
        groups=list(group) if group else None,
        tags=list(tag) if tag else None,
        name_regex=name,
        states=list(state) if state else None,
    )

    if not runs:
        click.echo("No runs found matching filters.")
        return

    # Convert to DataFrame for display
    import pandas as pd
    df = pd.DataFrame([
        {
            "run_id": r.id,
            "name": r.name,
            "group": r.group,
            "state": r.state,
            "created_at": r.created_at,
            "tags": ",".join(r.tags),
        }
        for r in runs
    ])

    if format == "markdown":
        result = to_markdown(df, title=f"Runs in {project}")
    elif format == "csv":
        result = to_csv(df)
    else:
        result = to_json(df)

    output_result(result, output, format)


@cli.command("diff-configs")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--run", "-r", "runs", multiple=True, help="Run IDs to compare")
@click.option("--group", "-g", multiple=True, help="Filter by group")
@click.option("--only-differing/--all", default=True, help="Only show differing keys")
@click.option("--allowlist", help="Comma-separated config keys to include")
@click.option("--format", "-f", type=click.Choice(["markdown", "csv", "json"]), default="markdown")
@click.option("--output", "-o", help="Output file")
@click.option("--refresh", is_flag=True, help="Bypass cache")
@click.option("--verbose", "-v", is_flag=True)
def diff_configs_cmd(project, runs, group, only_differing, allowlist, format, output, refresh, verbose):
    """Compare configurations across runs."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    # Get runs
    if runs:
        all_runs = fetcher.fetch_runs()
        run_infos = [r for r in all_runs if r.id in runs or r.name in runs]
    else:
        run_infos = fetcher.fetch_runs(groups=list(group) if group else None)

    if len(run_infos) < 2:
        click.echo("Need at least 2 runs to compare configs.")
        return

    # Fetch configs
    allowlist_keys = allowlist.split(",") if allowlist else None
    configs_df = fetcher.fetch_configs(run_infos, allowlist=allowlist_keys, refresh=refresh)

    # Diff
    diff_df = diff_configs(configs_df, only_differing=only_differing)

    if diff_df.empty:
        click.echo("No config differences found.")
        return

    result = format_config_diff(diff_df, format=format)
    output_result(result, output, format)


@cli.command("compare-metrics")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--run", "-r", "runs", multiple=True, help="Run IDs to compare")
@click.option("--group", "-g", multiple=True, help="Filter by group")
@click.option("--metric", "-m", "metrics", multiple=True, help="Metric names to compare")
@click.option("--pattern", multiple=True, help="Metric patterns (glob, e.g., 'eval_*/rewards_mean')")
@click.option("--final", "final_mode", default="last", help="Final value: last, best, step:N")
@click.option("--format", "-f", type=click.Choice(["markdown", "csv", "json"]), default="markdown")
@click.option("--output", "-o", help="Output file")
@click.option("--refresh", is_flag=True, help="Bypass cache")
@click.option("--verbose", "-v", is_flag=True)
def compare_metrics_cmd(project, runs, group, metrics, pattern, final_mode, format, output, refresh, verbose):
    """Compare final metrics across runs."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    # Get runs
    if runs:
        all_runs = fetcher.fetch_runs()
        run_infos = [r for r in all_runs if r.id in runs or r.name in runs]
    else:
        run_infos = fetcher.fetch_runs(groups=list(group) if group else None)

    if not run_infos:
        click.echo("No runs found.")
        return

    # Fetch summaries
    summaries_df = fetcher.fetch_summaries(run_infos, refresh=refresh)

    # Compare metrics
    metric_list = list(metrics) if metrics else None
    pattern_list = list(pattern) if pattern else DEFAULT_METRIC_PATTERNS

    metrics_df = compare_metrics(
        summaries_df,
        metrics=metric_list,
        metric_patterns=pattern_list,
    )

    if metrics_df.empty:
        click.echo("No metrics found.")
        return

    result = format_metrics_table(metrics_df, format=format)
    output_result(result, output, format)


@cli.command("group-summary")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--group", "-g", multiple=True, help="Filter by group (supports wildcards, default: all)")
@click.option("--exclude-group", "-x", multiple=True, help="Exclude groups matching pattern")
@click.option("--state", "-s", multiple=True, default=["finished"], help="Filter by state (default: finished)")
@click.option("--metric", "-m", "metrics", multiple=True, help="Specific metric names")
@click.option("--pattern", multiple=True, help="Metric patterns (glob, e.g., 'eval_*/rewards_mean')")
@click.option("--show-config-diff/--no-config-diff", default=True, help="Show config differences between groups")
@click.option("--format", "-f", type=click.Choice(["markdown", "csv", "json"]), default="markdown")
@click.option("--output", "-o", help="Output file")
@click.option("--refresh", is_flag=True, help="Bypass cache")
@click.option("--verbose", "-v", is_flag=True)
def group_summary_cmd(project, group, exclude_group, state, metrics, pattern, show_config_diff, format, output, refresh, verbose):
    """Compare metrics aggregated by group (mean ± std across seeds)."""
    import fnmatch

    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    # Get runs (filter by state)
    run_infos = fetcher.fetch_runs(
        groups=list(group) if group else None,
        states=list(state) if state else None,
    )

    if not run_infos:
        click.echo("No runs found.")
        return

    # Apply exclude-group filter (use normalize_group for consistency)
    if exclude_group:
        run_infos = [
            r for r in run_infos
            if not any(fnmatch.fnmatch(normalize_group(r.group), x) for x in exclude_group)
        ]

    if not run_infos:
        click.echo("No runs remaining after exclusion filter.")
        return

    # Count groups (use same normalization as aggregation layer)
    unique_groups = set(normalize_group(r.group) for r in run_infos)
    click.echo(f"Found {len(run_infos)} runs across {len(unique_groups)} groups", err=True)

    # Fetch summaries
    summaries_df = fetcher.fetch_summaries(run_infos, refresh=refresh)

    # Determine metric patterns
    metric_list = list(metrics) if metrics else None
    pattern_list = list(pattern) if pattern else DEFAULT_METRIC_PATTERNS

    # Aggregate by group
    agg_df = aggregate_by_group(
        summaries_df,
        metrics=metric_list,
        metric_patterns=pattern_list,
    )

    if agg_df.empty:
        click.echo("No metrics found to aggregate.")
        return

    # Config diff between groups (optional)
    config_diff_df = None
    if show_config_diff:
        configs_df = fetcher.fetch_configs(run_infos, refresh=refresh)
        config_diff_df = diff_configs_between_groups(configs_df)

    # Format output
    result = format_group_summary(
        agg_df,
        config_diff_df=config_diff_df if show_config_diff else None,
        format=format,
    )

    output_result(result, output, format)


@cli.command("curves")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--run", "-r", "runs", multiple=True, help="Run IDs")
@click.option("--group", "-g", multiple=True, help="Filter by group")
@click.option("--metric", "-m", "metrics", multiple=True, required=True, help="Metric names")
@click.option("--x-axis", default="_step", help="X-axis column")
@click.option("--sample-rate", default=1, type=int, help="Downsample factor")
@click.option("--resample-to", type=int, help="Resample to N uniform points")
@click.option("--output", "-o", required=True, help="Output file (csv)")
@click.option("--refresh", is_flag=True, help="Bypass cache")
@click.option("--verbose", "-v", is_flag=True)
def curves_cmd(project, runs, group, metrics, x_axis, sample_rate, resample_to, output, refresh, verbose):
    """Extract learning curves to CSV."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    # Get runs
    if runs:
        all_runs = fetcher.fetch_runs()
        run_infos = [r for r in all_runs if r.id in runs or r.name in runs]
    else:
        run_infos = fetcher.fetch_runs(groups=list(group) if group else None)

    if not run_infos:
        click.echo("No runs found.")
        return

    # Fetch history
    metric_list = list(metrics)
    history_df = fetcher.fetch_history(
        run_infos,
        keys=[x_axis] + metric_list,
        sample_rate=sample_rate,
        refresh=refresh,
    )

    if history_df.empty:
        click.echo("No history data found.")
        return

    # Extract curves
    curves = extract_learning_curves(
        history_df,
        metrics=metric_list,
        x_axis=x_axis,
        resample_to=resample_to,
    )

    # Save each curve
    output_path = Path(output)
    for metric, curve_df in curves.items():
        metric_safe = metric.replace("/", "_")
        curve_path = output_path.with_stem(f"{output_path.stem}_{metric_safe}")
        curve_df.to_csv(curve_path)
        click.echo(f"Saved {metric} to {curve_path}")


@cli.command("report")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--group", "-g", multiple=True, help="Filter by group")
@click.option("--metric", "-m", "metrics", multiple=True, help="Key metrics to highlight")
@click.option("--title", default="Experiment Report", help="Report title")
@click.option("--stable", is_flag=True, help="Git-stable output (no timestamps/IDs)")
@click.option("--output", "-o", help="Output file")
@click.option("--refresh", is_flag=True, help="Bypass cache")
@click.option("--verbose", "-v", is_flag=True)
def report_cmd(project, group, metrics, title, stable, output, refresh, verbose):
    """Generate comprehensive experiment report."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    # Get runs
    run_infos = fetcher.fetch_runs(groups=list(group) if group else None)

    if not run_infos:
        click.echo("No runs found.")
        return

    # Fetch data
    configs_df = fetcher.fetch_configs(run_infos, refresh=refresh)
    summaries_df = fetcher.fetch_summaries(run_infos, refresh=refresh)

    # Generate report
    metric_list = list(metrics) if metrics else None
    result = generate_report(
        configs_df,
        summaries_df,
        key_metrics=metric_list,
        title=title,
        stable=stable,
    )

    output_result(result, output, "markdown")


@cli.command("history")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--group-by", default="group", help="Group runs by (group, tags, config key)")
@click.option("--metric", "-m", "metrics", multiple=True, help="Key metrics for best-run identification")
@click.option("--format", "-f", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--output", "-o", help="Output file")
@click.option("--refresh", is_flag=True, help="Bypass cache")
@click.option("--verbose", "-v", is_flag=True)
def history_cmd(project, group_by, metrics, format, output, refresh, verbose):
    """Generate experiment history overview."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=verbose)

    # Get all runs
    run_infos = fetcher.fetch_runs()

    if not run_infos:
        click.echo("No runs found.")
        return

    # Fetch data
    configs_df = fetcher.fetch_configs(run_infos, refresh=refresh)
    summaries_df = fetcher.fetch_summaries(run_infos, refresh=refresh)

    # Summarize
    metric_list = list(metrics) if metrics else None
    summary = summarize_experiment_history(configs_df, summaries_df, key_metrics=metric_list)

    result = format_experiment_summary(summary, format=format)
    output_result(result, output, format)


@cli.command("clear-cache")
@click.option("--project", "-p", required=True, help="WandB project (entity/project)")
@click.option("--verbose", "-v", is_flag=True)
def clear_cache_cmd(project, verbose):
    """Clear cached data for a project."""
    entity, proj = parse_project(project)
    fetcher = WandBFetcher(entity, proj, verbose=True)
    fetcher.clear_cache()
    click.echo("Cache cleared.")


def main():
    cli()


if __name__ == "__main__":
    main()
