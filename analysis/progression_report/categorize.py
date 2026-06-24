"""Map WandB groups → method buckets for the PhD progression-report analysis.

Buckets focus on the entropy-control / exploration story:
- baseline-H: H sweep with no extra method (FS_PPO_4B_Hpt*, FS_PPO_14B_Hpt*, FS_PPO_pt5B_Hpt*).
- clipcov / klcov / topP / cosine / decay / adaptive: each a different
  loss-mediated exploration tweak from the literature.
- respHist: context-mediated baseline (longer response history). Kept
  separate — it's not a loss-mediated entropy method but is in the
  same comparison family.
- multi-action / 150tok / 128tok / 175tok / fuse_remPad / noBlock /
  scratch / small_vecenv: throughput / format variants — excluded from
  the entropy-method comparison.
- OC_*: Overcooked runs — excluded from the Snake analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    name: str
    description: str


BUCKETS = {
    "baseline_H": Bucket("baseline_H", "Entropy regularisation H sweep, no extra method."),
    "clipcov": Bucket("clipcov", "Covariance clipping (ARPO/DAPO-style)."),
    "klcov": Bucket("klcov", "KL covariance clipping."),
    "topP": Bucket("topP", "Top-p / nucleus-sampling restriction."),
    "cosine": Bucket("cosine", "Cosine LR / entropy schedule."),
    "decay": Bucket("decay", "Linear / step entropy decay schedule."),
    "adaptive": Bucket("adaptive", "Adaptive entropy controller (target-entropy band)."),
    "respHist": Bucket("respHist", "Response-history captioner (context-mediated)."),
    "multi_action": Bucket("multi_action", "Multi-action format with epsilon-greedy."),
    "throughput": Bucket("throughput", "Token-budget / fuse / no-block / vecenv variants — performance plumbing."),
    "overcooked": Bucket("overcooked", "Overcooked runs — not Snake."),
    "other": Bucket("other", "Uncategorised."),
}


_H_RE = re.compile(r"_Hpt(\d+)")


def _parse_h(group: str) -> float | None:
    """Parse the H coefficient from a group string like ``FS_PPO_4B_Hpt001`` → 0.001."""
    m = _H_RE.search(group or "")
    if not m:
        return None
    digits = m.group(1)
    return float("0." + digits)


def _parse_scale(group: str) -> str | None:
    """Parse model scale: 4B / 14B / pt5B → '4B' / '14B' / '0.5B'."""
    if "_pt5B_" in group:
        return "0.5B"
    if "_14B_" in group:
        return "14B"
    if "_4B_" in group:
        return "4B"
    return None


def bucket_for(group: str) -> str:
    """Return the bucket key for a WandB group string.

    Order matters — a group like ``FS_PPO_4B_Hpt01_topP_C`` is a topP run,
    not a baseline_H run, so topP must win.
    """
    g = group or ""
    if g.startswith("OC_") or "Overcooked" in g:
        return "overcooked"
    if "_respHist" in g:
        return "respHist"
    if "_topP_" in g:
        return "topP"
    if "_clipcov" in g or "_CLpt" in g:
        return "clipcov"
    if "_klcov" in g:
        return "klcov"
    if "_cos_" in g or g.endswith("_cos"):
        return "cosine"
    if "_decay_" in g:
        return "decay"
    if "_adapt_" in g:
        return "adaptive"
    if "multiact" in g or "multi_action" in g:
        return "multi_action"
    if any(tok in g for tok in ("_150tok", "_128tok", "_175tok", "_fuse_", "_noBlock",
                                "_scratch", "_SMALL_VECENV", "_NT", "test_")):
        return "throughput"
    if _H_RE.search(g):
        return "baseline_H"
    return "other"


def categorize_row(row) -> dict:
    """Return ``{'bucket': str, 'H': float|None, 'scale': str|None}`` for a config-row."""
    group = row.get("group") if hasattr(row, "get") else row["group"]
    return {
        "bucket": bucket_for(group),
        "H": _parse_h(group),
        "scale": _parse_scale(group),
    }


# Run-level filters — apply BEFORE bucketing.
def is_snake_run(row) -> bool:
    """Filter to Snake runs only."""
    env = row.get("envs.env_name") if hasattr(row, "get") else row.get("envs.env_name")
    if env is not None and isinstance(env, str):
        return env.lower() in ("fastsnake", "snake")
    # fallback: group prefix
    g = (row.get("group") or "") if hasattr(row, "get") else (row["group"] or "")
    return g.startswith("FS_PPO_") or g.startswith("test_multi_action")
