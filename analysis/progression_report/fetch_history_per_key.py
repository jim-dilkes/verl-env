"""Fetch history per metric (separate API call per key), then merge.

``wandb.Api().run(...).history(keys=[a, b])`` does an inner join on all keys —
if any key is missing on any logged step, the row is dropped. Several Snake
metrics log at different cadences (training every step vs eval every N steps),
so passing them together yields 0 rows. Fetching per-key and merging on
``_step`` recovers the dense per-metric data.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.fetch import WandBFetcher  # noqa: E402
from analysis.progression_report.categorize import bucket_for  # noqa: E402
from analysis.progression_report.fetch_data import HISTORY_KEYS, PROJECT  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "verl_env_history_per_key.parquet"

# Limit to the curves we actually plot — keeps the fetch under ~5 min.
CURVE_KEYS = [
    "critic/rewards/mean",
    "actor/entropy",
    "eval_FastSnake-Default-Greedy/rewards_mean",
    "eval_Snake-20Step-Greedy/rewards_mean",
]


def fetch_one_run(api, project_path: str, run_id: str, keys: list[str]) -> pd.DataFrame:
    """Fetch each metric separately and outer-merge on _step."""
    run = api.run(f"{project_path}/{run_id}")
    merged: pd.DataFrame | None = None
    for key in keys:
        if key == "_step":
            continue
        try:
            df = run.history(keys=["_step", key], samples=2000)
        except Exception as exc:
            print(f"    {run_id} {key}: error {exc}", file=sys.stderr)
            continue
        if df.empty or key not in df.columns:
            continue
        df = df[["_step", key]].dropna(subset=[key])
        merged = df if merged is None else merged.merge(df, on="_step", how="outer")
        time.sleep(0.05)
    if merged is None:
        return pd.DataFrame()
    merged["run_id"] = run_id
    return merged


def main(refresh: bool = False, scale_filter: str | None = "4B",
         buckets: tuple[str, ...] = ("baseline_H", "adaptive", "decay", "clipcov")) -> None:
    if not refresh and OUT_PATH.exists():
        print(f"already cached at {OUT_PATH}; pass --refresh to redo")
        return

    entity, project = PROJECT.split("/", 1)
    fetcher = WandBFetcher(entity=entity, project=project, verbose=True)
    runs = fetcher.fetch_runs(states=["finished"])
    runs = [r for r in runs if bucket_for(r.group) in buckets]
    if scale_filter:
        runs = [r for r in runs if f"_{scale_filter}_" in r.group]
    print(f">>> fetching history for {len(runs)} runs × {len(CURVE_KEYS)} metrics "
          f"= {len(runs)*len(CURVE_KEYS)} API calls")

    api = wandb.Api()
    dfs: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for i, r in enumerate(runs, start=1):
        print(f"  [{i}/{len(runs)}] {r.id}  group={r.group}")
        df = fetch_one_run(api, fetcher.project_path, r.id, CURVE_KEYS)
        if not df.empty:
            dfs.append(df)
            print(f"      {len(df)} rows, "
                  f"cols={[c for c in df.columns if c not in ('_step','run_id')]}")

    if not dfs:
        print("no history fetched")
        return
    combined = pd.concat(dfs, ignore_index=True)
    # Write CSV first as a safety net; parquet fails silently in some pyarrow/pyarrow
    # version mismatches with mixed-dtype frames.
    csv_path = OUT_PATH.with_suffix(".csv")
    try:
        combined.to_csv(csv_path, index=False)
        print(f"    csv checkpoint -> {csv_path} ({len(combined)} rows)")
    except Exception as exc:
        print(f"    csv write failed: {exc}")
    try:
        combined.to_parquet(OUT_PATH)
        print(f">>> wrote {OUT_PATH} ({len(combined)} rows) in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:
        print(f"!!! parquet write failed: {exc}")
        print(f"!!! keep the CSV at {csv_path} and re-derive")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--scale", default=None, help="Filter by scale token (e.g. '4B')")
    args = p.parse_args()
    main(refresh=args.refresh, scale_filter=args.scale)
