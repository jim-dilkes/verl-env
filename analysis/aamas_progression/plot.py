"""One-figure summary for the progression-report tail.

Story: standard entropy-control methods (entropy reg H sweep, covariance clip,
KL-covariance clip) raise token entropy only slightly, don't move converged
reward, and don't increase behavioural validity. Motivates the shift to
context-mediated exploration in the following section.

Source: ``wandb.ai/jimdilkes/AAMAS_msrl`` (the AAMAS submission run set;
Snake/FrozenLake/BabyAI; here we slice to Snake at 3B PPO).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_DIR / "data"
FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "aamas_progression"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_SLUG = "AAMAS_msrl"

# Method buckets for the 3B PPO cut.
METHOD_ORDER = ["baseline_default", "Hpt002", "Hpt005", "Hpt01", "clipcov", "klcov"]
METHOD_LABELS = {
    "baseline_default": "Baseline\n(H=0.001)",
    "Hpt002": "H=0.002",
    "Hpt005": "H=0.005",
    "Hpt01": "H=0.01",
    "clipcov": "Cov-clip",
    "klcov": "KL-cov",
}
METHOD_COLORS = {
    "baseline_default": "#2C7BB6",
    "Hpt002": "#80B1D3",
    "Hpt005": "#FDB462",
    "Hpt01": "#FB8072",
    "clipcov": "#D73027",
    "klcov": "#A50026",
}

# Scale prefix → human label.
SCALE_LABELS = {"3B": "Qwen2.5-3B PPO", "Q3_4B": "Qwen3-4B PPO"}

# Key metrics in the wandb summary.
TRAIN_REWARD_KEY = "critic/rewards/mean"
EVAL_REWARD_KEY = "eval_FastSnake-Default/rewards_mean"
ENTROPY_KEY = "actor/entropy"
VALID_KEY = "behavior/valid_action_ratio"


_H_RE = re.compile(r"_Hpt(\d+)")


def _latest(pattern: str) -> Path:
    candidates = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no cache files matching {pattern}")
    return candidates[-1]


def _bucket(group: str) -> str | None:
    """Map ``FS_PPO_3B``-family group strings to method buckets. Returns None
    if the group is outside the 3B Snake PPO comparison set."""
    if not (group.startswith("FS_PPO_3B") or group.startswith("FS_PPO_Q3_4B")):
        return None
    if "_128tok" in group or "_LoRA" in group or "_clipcov" in group and "_128" in group:
        # exclude format / capacity variants — they confound the entropy story
        pass
    # tail after the scale prefix
    if "_clipcov" in group:
        return "clipcov"
    if "_klcov" in group:
        return "klcov"
    m = _H_RE.search(group)
    if m:
        suffix = m.group(1)
        return f"Hpt{suffix}"
    # exact baseline name (no method suffix)
    if group in {"FS_PPO_3B", "FS_PPO_Q3_4B"}:
        return "baseline_default"
    return None


def _scale(group: str) -> str:
    if group.startswith("FS_PPO_Q3_4B"):
        return "Q3_4B"
    if group.startswith("FS_PPO_3B"):
        return "3B"
    return "?"


def load() -> pd.DataFrame:
    """Load configs + summaries, attach bucket / scale, filter to the comparison set."""
    cfg = pd.read_parquet(_latest("AAMAS_msrl_configs_*.parquet"))
    summ = pd.read_parquet(_latest("AAMAS_msrl_summaries_*.parquet"))
    if cfg.index.name == "run_id":
        cfg = cfg.reset_index()
    if summ.index.name == "run_id":
        summ = summ.reset_index()
    cfg["bucket"] = cfg["group"].apply(_bucket)
    cfg["scale"] = cfg["group"].apply(_scale)
    keep_cols = ["run_id", "group", "bucket", "scale", "run_name", "tags"]
    keep_cols = [c for c in keep_cols if c in cfg.columns]
    merged = cfg[keep_cols].merge(summ, on="run_id", how="left")
    merged = merged[merged["bucket"].notna()].copy()
    return merged


def _meanse(values: pd.Series) -> tuple[float, float, int]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return (np.nan, 0.0, 0)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return (m, se, len(v))


def _bar(ax, data: pd.DataFrame, metric: str, title: str, ylabel: str,
        baseline_label: str = "baseline_default") -> None:
    """Bar plot with seed dots + baseline reference line."""
    rng = np.random.default_rng(0)
    means, ses, ns, colors = [], [], [], []
    jx, jy = [], []
    for j, b in enumerate(METHOD_ORDER):
        vals = pd.to_numeric(data[data["bucket"] == b][metric], errors="coerce").dropna()
        m, se, n = _meanse(vals)
        means.append(m); ses.append(se); ns.append(n); colors.append(METHOD_COLORS[b])
        if n:
            jx.extend(rng.normal(j, 0.05, size=n))
            jy.extend(vals.values)
    xs = np.arange(len(METHOD_ORDER))
    ax.bar(xs, means, color=colors, yerr=ses, capsize=2.5, alpha=0.88,
           edgecolor="black", lw=0.4)
    ax.scatter(jx, jy, s=14, color="black", alpha=0.55, zorder=3)
    base_m = means[0]
    if np.isfinite(base_m):
        ax.axhline(base_m, ls="--", color=METHOD_COLORS[baseline_label], lw=1, alpha=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{METHOD_LABELS[b]}\n(n={n})"
                        for b, n in zip(METHOD_ORDER, ns)], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.tick_params(axis="y", labelsize=8)


def make_figure(scale: str = "3B") -> Path:
    """Three-panel figure for a single model scale: train reward, token entropy,
    valid-action ratio (behavioural-validity proxy)."""
    data = load()
    sub = data[data["scale"] == scale].copy()
    if sub.empty:
        raise RuntimeError(f"no data for scale={scale}")

    panels = [
        (TRAIN_REWARD_KEY, "Converged train reward", "reward"),
        (ENTROPY_KEY, "Final token entropy", "entropy (nats)"),
        (VALID_KEY, "Final valid-action rate", "share of valid actions"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    for ax, (metric, title, ylabel) in zip(axes, panels):
        _bar(ax, sub, metric, title, ylabel)
    fig.suptitle(
        f"Snake — {SCALE_LABELS.get(scale, scale)}: entropy-control methods vs baseline",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / f"fig_methods_{scale}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_table(scale: str = "3B") -> Path:
    data = load()
    sub = data[data["scale"] == scale]
    rows = [["method", "n", "train reward", "default eval", "token entropy", "valid-action rate"]]
    for b in METHOD_ORDER:
        sub_b = sub[sub["bucket"] == b]
        if sub_b.empty:
            continue

        def fmt(metric: str) -> str:
            m, se, n = _meanse(sub_b[metric])
            return "—" if n == 0 else f"{m:.3f} ± {se:.3f}"

        rows.append([
            METHOD_LABELS[b].replace("\n", " "),
            str(len(sub_b)),
            fmt(TRAIN_REWARD_KEY),
            fmt(EVAL_REWARD_KEY),
            fmt(ENTROPY_KEY),
            fmt(VALID_KEY),
        ])

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    md_lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(rows[0], widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ] + ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows[1:]]
    out = FIG_DIR / f"table_methods_{scale}.md"
    out.write_text("\n".join(md_lines) + "\n")
    return out


if __name__ == "__main__":
    for sc in ("3B", "Q3_4B"):
        try:
            f = make_figure(sc)
            t = write_summary_table(sc)
            print(f"wrote {f}\nwrote {t}")
        except Exception as exc:
            print(f"skipping scale={sc}: {exc}")
