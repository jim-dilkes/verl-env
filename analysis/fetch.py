"""WandB API wrapper with caching and rate limiting."""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb

from .config import (
    CACHE_DIR,
    DEFAULT_CONFIG_ALLOWLIST,
    IGNORE_CONFIG_PREFIXES,
    API_CALL_DELAY,
    LARGE_REQUEST_THRESHOLD,
)


@dataclass
class RunInfo:
    """Lightweight run reference."""

    id: str
    name: str
    group: str
    tags: list[str]
    state: str  # running, finished, crashed, failed, killed
    created_at: str
    entity: str = ""
    project: str = ""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, RunInfo) and self.id == other.id


def flatten_config(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict with dot notation.

    Handles JSON strings that wandb sometimes returns for nested configs.
    """
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except (json.JSONDecodeError, TypeError):
            return {}

    if not isinstance(d, dict):
        return {}

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def filter_config_keys(
    config: dict,
    allowlist: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
) -> dict:
    """Filter config keys by allowlist and ignore prefixes."""
    if ignore_prefixes is None:
        ignore_prefixes = IGNORE_CONFIG_PREFIXES

    # First, remove ignored prefixes
    filtered = {
        k: v
        for k, v in config.items()
        if not any(k.startswith(prefix) for prefix in ignore_prefixes)
    }

    # If allowlist specified, keep only those keys
    if allowlist:
        filtered = {k: v for k, v in filtered.items() if k in allowlist}

    return filtered


class WandBFetcher:
    """Wrapper around wandb.Api with caching and rate limiting."""

    def __init__(
        self,
        entity: str,
        project: str,
        cache_dir: Path | None = None,
        verbose: bool = False,
    ):
        self.entity = entity
        self.project = project
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._api = None

    @property
    def api(self) -> wandb.Api:
        if self._api is None:
            self._api = wandb.Api()
        return self._api

    @property
    def project_path(self) -> str:
        return f"{self.entity}/{self.project}"

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[analysis] {msg}", file=sys.stderr, flush=True)

    def _log_progress(self, i: int, total: int, label: str, run_id: str = "") -> None:
        """Log progress for per-run loops (verbose-only).

        Keeps output readable by only logging the first item, last item,
        and every 10th item.
        """
        if not self.verbose:
            return

        if total <= 0:
            return

        should_log = (i == 1) or (i == total) or (i % 10 == 0)
        if should_log:
            suffix = f" ({run_id})" if run_id else ""
            self._log(f"{label}: {i}/{total}{suffix}")

    def _cache_path(self, name: str, suffix: str = ".parquet") -> Path:
        return self.cache_dir / f"{self.project}_{name}{suffix}"

    def _hash_params(self, **params) -> str:
        """Create stable hash from parameters."""
        serialized = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:8]

    def fetch_runs(
        self,
        filters: dict | None = None,
        groups: list[str] | None = None,
        tags: list[str] | None = None,
        name_regex: str | None = None,
        states: list[str] | None = None,
    ) -> list[RunInfo]:
        """Fetch runs matching filters.

        Args:
            filters: Raw wandb filter dict
            groups: Filter by group names (supports wildcards via fnmatch)
            tags: Filter by tags (runs must have ALL specified tags)
            name_regex: Filter by run name pattern
            states: Filter by run state (finished, running, crashed, etc.)

        Returns:
            List of RunInfo objects
        """
        t0 = time.perf_counter()
        self._log(f"Fetching runs from {self.project_path}...")

        # Build wandb filters
        wandb_filters = filters.copy() if filters else {}

        # Fetch all runs (wandb API filtering is limited)
        runs = list(self.api.runs(self.project_path, filters=wandb_filters))
        self._log(f"Fetched {len(runs)} runs from API; applying local filters...")

        if len(runs) > LARGE_REQUEST_THRESHOLD:
            self._log(
                f"Warning: Fetched {len(runs)} runs. Consider adding filters."
            )

        # Convert to RunInfo and apply additional filters
        result = []
        for run in runs:
            info = RunInfo(
                id=run.id,
                name=run.name,
                group=run.group or "",
                tags=list(run.tags) if run.tags else [],
                state=run.state,
                created_at=run.created_at,
                entity=self.entity,
                project=self.project,
            )

            # Apply group filter (supports wildcards)
            if groups:
                import fnmatch

                if not any(fnmatch.fnmatch(info.group, g) for g in groups):
                    continue

            # Apply tag filter (must have ALL tags)
            if tags:
                if not all(t in info.tags for t in tags):
                    continue

            # Apply name regex
            if name_regex:
                if not re.search(name_regex, info.name):
                    continue

            # Apply state filter
            if states:
                if info.state not in states:
                    continue

            result.append(info)
            time.sleep(API_CALL_DELAY)

        self._log(f"Found {len(result)} matching runs")
        self._log(f"fetch_runs completed in {time.perf_counter() - t0:.2f}s")
        return result

    def fetch_summaries(
        self,
        runs: list[RunInfo],
        keys: list[str] | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch run summaries (final metrics).

        Args:
            runs: List of RunInfo objects
            keys: Specific keys to fetch (None = all)
            refresh: Bypass cache

        Returns:
            DataFrame with run_id as index, metrics as columns
        """
        cache_key = self._hash_params(
            runs=[r.id for r in runs], keys=keys
        )
        cache_path = self._cache_path(f"summaries_{cache_key}")

        if not refresh and cache_path.exists():
            self._log(f"Cache hit (summaries): {cache_path}")
            return pd.read_parquet(cache_path)

        if refresh:
            self._log("Refresh enabled; bypassing summaries cache")
        else:
            self._log(f"Cache miss (summaries): will write {cache_path}")

        t0 = time.perf_counter()
        self._log(f"Fetching summaries for {len(runs)} runs...")
        records = []

        for i, run_info in enumerate(runs, start=1):
            self._log_progress(i, len(runs), label="Summaries", run_id=run_info.id)
            try:
                run = self.api.run(f"{self.project_path}/{run_info.id}")

                # Get summary dict safely
                summary_data = getattr(run.summary, "_json_dict", run.summary)
                if isinstance(summary_data, str):
                    try:
                        summary_data = json.loads(summary_data)
                    except (json.JSONDecodeError, TypeError):
                        summary_data = {}
                if not isinstance(summary_data, dict):
                    summary_data = {}

                # Filter keys if specified
                if keys:
                    summary_data = {k: v for k, v in summary_data.items() if k in keys}

                summary_data["run_id"] = run_info.id
                summary_data["run_name"] = run_info.name
                summary_data["group"] = run_info.group
                records.append(summary_data)

            except Exception as e:
                self._log(f"Error fetching summary for {run_info.id}: {e}")

            time.sleep(API_CALL_DELAY)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("run_id")

        # Cache result
        df.to_parquet(cache_path)
        self._log(f"Cached summaries to {cache_path} ({len(df)} rows)")
        self._log(f"fetch_summaries completed in {time.perf_counter() - t0:.2f}s")

        return df

    def fetch_history(
        self,
        runs: list[RunInfo],
        keys: list[str],
        sample_rate: int = 1,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch run history (metrics over time).

        Args:
            runs: List of RunInfo objects
            keys: Metric keys to fetch
            sample_rate: Downsample factor (1 = all points, 10 = every 10th)
            refresh: Bypass cache

        Returns:
            DataFrame with columns: run_id, _step, and requested metrics
        """
        cache_key = self._hash_params(
            runs=[r.id for r in runs], keys=keys, sample_rate=sample_rate
        )
        cache_path = self._cache_path(f"history_{cache_key}")

        if not refresh and cache_path.exists():
            self._log(f"Cache hit (history): {cache_path}")
            return pd.read_parquet(cache_path)

        if refresh:
            self._log("Refresh enabled; bypassing history cache")
        else:
            self._log(f"Cache miss (history): will write {cache_path}")

        t0 = time.perf_counter()
        self._log(f"Fetching history for {len(runs)} runs (keys: {keys}, sample_rate={sample_rate})...")
        dfs = []

        for i, run_info in enumerate(runs, start=1):
            self._log_progress(i, len(runs), label="History", run_id=run_info.id)
            try:
                run = self.api.run(f"{self.project_path}/{run_info.id}")
                history = run.history(keys=keys, samples=10000)

                if sample_rate > 1:
                    history = history.iloc[::sample_rate]

                history["run_id"] = run_info.id
                history["run_name"] = run_info.name
                dfs.append(history)
                self._log(f"  {run_info.id}: {len(history)} rows")

            except Exception as e:
                self._log(f"Error fetching history for {run_info.id}: {e}")

            time.sleep(API_CALL_DELAY)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()

        # Cache result
        df.to_parquet(cache_path)
        self._log(f"Cached history to {cache_path} ({len(df)} rows)")
        self._log(f"fetch_history completed in {time.perf_counter() - t0:.2f}s")

        return df

    def fetch_configs(
        self,
        runs: list[RunInfo],
        allowlist: list[str] | None = None,
        include_metadata: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch run configurations.

        Args:
            runs: List of RunInfo objects
            allowlist: Config keys to include (None = default allowlist)
            include_metadata: Fetch host/GPU from wandb-metadata.json
            refresh: Bypass cache

        Returns:
            DataFrame with run_id as index, config keys as columns
        """
        if allowlist is None:
            allowlist = DEFAULT_CONFIG_ALLOWLIST

        cache_key = self._hash_params(
            runs=[r.id for r in runs],
            allowlist=allowlist,
            include_metadata=include_metadata,
        )
        cache_path = self._cache_path(f"configs_{cache_key}")

        if not refresh and cache_path.exists():
            self._log(f"Cache hit (configs): {cache_path}")
            return pd.read_parquet(cache_path)

        if refresh:
            self._log("Refresh enabled; bypassing configs cache")
        else:
            self._log(f"Cache miss (configs): will write {cache_path}")

        t0 = time.perf_counter()
        self._log(f"Fetching configs for {len(runs)} runs (allowlist={len(allowlist)}, include_metadata={include_metadata})...")
        records = []

        for i, run_info in enumerate(runs, start=1):
            self._log_progress(i, len(runs), label="Configs", run_id=run_info.id)
            try:
                run = self.api.run(f"{self.project_path}/{run_info.id}")

                # Flatten and filter config
                flat_config = flatten_config(run.config)
                filtered_config = filter_config_keys(flat_config, allowlist=allowlist)

                # Add run metadata
                filtered_config["run_id"] = run_info.id
                filtered_config["run_name"] = run_info.name
                filtered_config["group"] = run_info.group
                filtered_config["tags"] = ",".join(run_info.tags)
                filtered_config["state"] = run_info.state
                filtered_config["created_at"] = run_info.created_at

                # Fetch host/GPU metadata
                if include_metadata:
                    meta = self._fetch_run_metadata(run)
                    filtered_config["hostname"] = meta.get("host", "")
                    filtered_config["gpu_type"] = meta.get("gpu_type", "")

                records.append(filtered_config)

            except Exception as e:
                self._log(f"Error fetching config for {run_info.id}: {e}")

            time.sleep(API_CALL_DELAY)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("run_id")

        # Cache result
        df.to_parquet(cache_path)
        self._log(f"Cached configs to {cache_path} ({len(df)} rows)")
        self._log(f"fetch_configs completed in {time.perf_counter() - t0:.2f}s")

        return df

    def _fetch_run_metadata(self, run) -> dict:
        """Fetch wandb-metadata.json for a run."""
        try:
            metadata_file = run.file("wandb-metadata.json").download(replace=True)
            with open(metadata_file.name, "r") as f:
                meta = json.load(f)
            os.remove(metadata_file.name)
            return meta
        except Exception:
            return {}

    def clear_cache(self) -> None:
        """Clear all cached data for this project."""
        for f in self.cache_dir.glob(f"{self.project}_*"):
            f.unlink()
            self._log(f"Removed {f}")
