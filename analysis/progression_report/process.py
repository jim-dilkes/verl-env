"""Aggregate summaries/history into Snake-only per-method tables.

Read the latest cached parquets in ``analysis/data/`` and join configs with
summaries / history. Returns DataFrames keyed by (bucket, H, scale, run_id).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .categorize import bucket_for, _parse_h, _parse_scale, is_snake_run

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

EVAL_SPLITS = {
    "train": "val/rewards_mean",  # in-distribution proxy (val matches train env)
    "default": "eval_FastSnake-Default-Greedy/rewards_mean",
    "20step": "eval_Snake-20Step-Greedy/rewards_mean",
    "poison": "eval_Snake-20Steps-PoisonAppleAndBanana-Greedy/rewards_mean",
}

EVAL_SCORES = {
    "default": "eval_FastSnake-Default-Greedy/score_mean",
    "20step": "eval_Snake-20Step-Greedy/score_mean",
    "poison": "eval_Snake-20Steps-PoisonAppleAndBanana-Greedy/score_mean",
}

ENTROPY_KEY = "actor/entropy"
TRAIN_REWARD_KEY = "critic/rewards/mean"


def _latest(pattern: str) -> Path:
    candidates = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no cache files matching {pattern} in {DATA_DIR}")
    return candidates[-1]


def load_configs() -> pd.DataFrame:
    """Load the freshest configs parquet, attach bucket / H / scale columns."""
    df = pd.read_parquet(_latest("verl_env_configs_*.parquet"))
    df = df.reset_index().rename(columns={"index": "run_id"}) if df.index.name == "run_id" else df
    if "run_id" not in df.columns:
        df["run_id"] = df.index
    df["bucket"] = df["group"].apply(bucket_for)
    df["H"] = df["group"].apply(_parse_h)
    df["scale"] = df["group"].apply(_parse_scale)
    df["is_snake"] = df.apply(is_snake_run, axis=1)
    return df


def load_summaries() -> pd.DataFrame:
    df = pd.read_parquet(_latest("verl_env_summaries_*.parquet"))
    if df.index.name == "run_id":
        df = df.reset_index()
    if "run_id" not in df.columns:
        df["run_id"] = df.index
    return df


def load_history() -> pd.DataFrame:
    """Prefer the per-key cache (curves merge on _step); fall back to inner-join.

    The per-key fetch can produce mixed-dtype columns (a few rows where wandb
    serialises ``actor/entropy`` as a string), which breaks parquet. We always
    persist a CSV alongside the parquet; coerce numeric columns to float on load.
    """
    per_key_pq = DATA_DIR / "verl_env_history_per_key.parquet"
    per_key_csv = DATA_DIR / "verl_env_history_per_key.csv"
    if per_key_pq.exists():
        df = pd.read_parquet(per_key_pq)
    elif per_key_csv.exists():
        df = pd.read_csv(per_key_csv)
    else:
        return pd.read_parquet(_latest("verl_env_history_*.parquet"))
    for c in df.columns:
        if c in ("run_id", "run_name"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def snake_run_table() -> pd.DataFrame:
    """Return per-run table: run_id, group, bucket, H, scale + final metrics."""
    cfg = load_configs()
    summ = load_summaries()
    snake_cfg = cfg[cfg["is_snake"]].copy()

    # Keep only the eval/entropy/reward columns we care about.
    metric_cols = [ENTROPY_KEY] + list(EVAL_SPLITS.values()) + list(EVAL_SCORES.values())
    metric_cols = [c for c in metric_cols if c in summ.columns]
    summ_small = summ[["run_id"] + metric_cols].copy()

    merged = snake_cfg.merge(summ_small, on="run_id", how="left")
    return merged


def snake_history_table() -> pd.DataFrame:
    """Return long-form per-run history with bucket/H/scale columns attached."""
    hist = load_history()
    cfg = load_configs()[["run_id", "group", "run_name", "bucket", "H", "scale", "is_snake"]]
    merged = hist.merge(cfg, on="run_id", how="left")
    return merged[merged["is_snake"].fillna(False)].copy()


def aggregate_by_method(table: pd.DataFrame, metric: str, scale: str = "4B") -> pd.DataFrame:
    """Mean / std / N of ``metric`` per (bucket, H) within a scale."""
    sub = table[(table["scale"] == scale) & table[metric].notna()].copy()
    grouped = (
        sub.groupby(["bucket", "H"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped


def baseline_h_curves(history: pd.DataFrame, key: str, scale: str = "4B") -> pd.DataFrame:
    """Wide table: rows = step, cols = (H, run_id) for the baseline_H bucket."""
    sub = history[(history["scale"] == scale) & (history["bucket"] == "baseline_H")].copy()
    sub = sub[sub[key].notna()]
    if "_step" not in sub.columns:
        return pd.DataFrame()
    return sub[["_step", "run_id", "H", key]].rename(columns={key: "value"})
