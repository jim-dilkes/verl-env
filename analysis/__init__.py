"""WandB experiment analysis tools.

Usage:
    # CLI
    python -m analysis.cli list-runs --project entity/project

    # Python API
    from analysis import WandBFetcher, diff_configs, compare_metrics

    fetcher = WandBFetcher("entity", "project")
    runs = fetcher.fetch_runs(groups=["FS_PPO_*"])
    configs = fetcher.fetch_configs(runs)
    diff = diff_configs(configs)
"""

from .fetch import WandBFetcher, RunInfo, flatten_config
from .compare import (
    diff_configs,
    compare_metrics,
    extract_learning_curves,
    summarize_experiment_history,
    ExperimentSummary,
)
from .export import (
    to_markdown,
    to_csv,
    to_json,
    generate_report,
)
from .config import (
    DEFAULT_CONFIG_ALLOWLIST,
    DEFAULT_METRIC_PATTERNS,
    HIGHER_IS_BETTER,
)

__all__ = [
    # Fetch
    "WandBFetcher",
    "RunInfo",
    "flatten_config",
    # Compare
    "diff_configs",
    "compare_metrics",
    "extract_learning_curves",
    "summarize_experiment_history",
    "ExperimentSummary",
    # Export
    "to_markdown",
    "to_csv",
    "to_json",
    "generate_report",
    # Config
    "DEFAULT_CONFIG_ALLOWLIST",
    "DEFAULT_METRIC_PATTERNS",
    "HIGHER_IS_BETTER",
]
