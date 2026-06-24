"""Fetch run summaries + history for the PhD progression-report analysis.

Caches to analysis/data/ via WandBFetcher.

Usage:
    python -m analysis.progression_report.fetch_data [--refresh]
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

# Make `analysis` package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.fetch import WandBFetcher  # noqa: E402

PROJECT = "jimdilkes/verl_env"

HISTORY_KEYS = [
    "_step",
    "actor/entropy",
    "actor/entropy_coeff_used",
    "critic/rewards/mean",
    "val/rewards_mean",
    # Held-out eval splits (Snake instruction-generalisation suite).
    "eval_FastSnake-Default-Greedy/rewards_mean",
    "eval_FastSnake-Default-Greedy/score_mean",
    "eval_Snake-20Step-Greedy/rewards_mean",
    "eval_Snake-20Step-Greedy/score_mean",
    "eval_Snake-20Steps-PoisonAppleAndBanana-Greedy/rewards_mean",
    "eval_FastSnake-Default-StateVisitation/n_distinct_state_actions_valid_mean",
    "eval_FastSnake-Default-StateVisitation/distinct_state_actions_per_frame_valid_mean",
]


@click.command()
@click.option("--refresh", is_flag=True, help="Bypass all caches.")
@click.option("--with-history", is_flag=True, default=True, help="Also fetch history curves.")
@click.option(
    "--sample-rate",
    default=2,
    help="Downsample history by this factor (1 = every step).",
)
def main(refresh: bool, with_history: bool, sample_rate: int) -> None:
    entity, project = PROJECT.split("/", 1)
    fetcher = WandBFetcher(entity=entity, project=project, verbose=True)

    print(">>> fetch_runs (finished only)")
    runs = fetcher.fetch_runs(states=["finished"])
    print(f"    got {len(runs)} runs")

    print(">>> fetch_configs")
    configs = fetcher.fetch_configs(runs, include_metadata=False, refresh=refresh)
    print(f"    configs: {len(configs)} rows × {len(configs.columns)} cols")

    print(">>> fetch_summaries")
    summaries = fetcher.fetch_summaries(runs, refresh=refresh)
    print(f"    summaries: {len(summaries)} rows × {len(summaries.columns)} cols")

    if with_history:
        print(">>> fetch_history (large; sample_rate=%d)" % sample_rate)
        history = fetcher.fetch_history(runs, keys=HISTORY_KEYS, sample_rate=sample_rate, refresh=refresh)
        print(f"    history: {len(history)} rows × {len(history.columns)} cols")


if __name__ == "__main__":
    main()
