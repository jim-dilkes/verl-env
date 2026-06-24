"""Performance-overview figures for the PhD progression report.

Two figures:
1. ``fig_perf_overview.png`` — training reward + token entropy curves for the
   baseline-H sweep at the 4B scale (mean ± SE across seeds, per H).
2. ``fig_perf_heldout.png`` — final-step reward across held-out eval splits
   (Default-Greedy / 20Step-Greedy / PoisonApple) for each H, with seeds shown.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .process import (
    EVAL_SPLITS,
    ENTROPY_KEY,
    TRAIN_REWARD_KEY,
    aggregate_by_method,
    baseline_h_curves,
    snake_history_table,
    snake_run_table,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "progression_report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colour-blind-safe palette ordered by H low → high.
H_COLORS = {
    0.0005: "#0072B2",
    0.001: "#009E73",
    0.002: "#56B4E9",
    0.005: "#E69F00",
    0.01: "#D55E00",
    0.02: "#CC79A7",
    0.05: "#7B1FA2",
}

SE = lambda s: s.std(ddof=1) / np.sqrt(max(s.count(), 1))


def _bin_steps(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    """Bin _step into quantile bins so seeds with different stride align."""
    if df.empty:
        return df
    df = df.copy()
    df["bin"] = pd.cut(df["_step"], bins=bins, labels=False, include_lowest=True)
    return df


def plot_curves_per_H(scale: str = "4B") -> Path:
    """4-panel: train reward, token entropy, default-eval, 20step-eval — per H.

    Drops the diverged H=0.05 run (n=1, entropy ≈11 nats) so the y-axes show the
    meaningful H sweep.
    """
    hist = snake_history_table()
    hist = hist[hist["H"] != 0.05]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        (axes[0, 0], TRAIN_REWARD_KEY, "Training reward (critic/rewards/mean)", "reward"),
        (axes[0, 1], ENTROPY_KEY, "Token entropy (actor/entropy)", "entropy (nats)"),
        (axes[1, 0], EVAL_SPLITS["default"], "Held-out: Default-Greedy", "reward"),
        (axes[1, 1], EVAL_SPLITS["20step"], "Held-out: 20Step-Greedy (horizon shift)", "reward"),
    ]
    for ax, key, title, ylabel in panels:
        data = baseline_h_curves(hist, key=key, scale=scale)
        if data.empty:
            ax.set_title(f"{title}\n(no data)")
            continue
        binned = _bin_steps(data, bins=40)
        # Mean ± SE per H bucket.
        agg = (
            binned.groupby(["H", "bin"])
            .agg(value_mean=("value", "mean"),
                 value_se=("value", SE),
                 step=("_step", "mean"),
                 n=("value", "count"))
            .reset_index()
        )
        for H_val, grp in agg.groupby("H"):
            grp = grp.sort_values("step")
            c = H_COLORS.get(round(H_val, 4), "grey")
            ax.plot(grp["step"], grp["value_mean"], color=c, lw=1.8, label=f"H={H_val:g}")
            ax.fill_between(grp["step"],
                            grp["value_mean"] - grp["value_se"],
                            grp["value_mean"] + grp["value_se"],
                            color=c, alpha=0.18, lw=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, lw=0.5)
    axes[1, 0].set_xlabel("training step")
    axes[1, 1].set_xlabel("training step")
    axes[0, 0].legend(loc="best", fontsize=8, frameon=False, title="entropy coeff")
    fig.suptitle(f"Snake — baseline H sweep ({scale}), mean ± SE across seeds", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "fig_perf_overview.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heldout_bars(scale: str = "4B") -> Path:
    """Grouped bar chart: final reward by eval split, per H value (with seed dots)."""
    table = snake_run_table()
    base = table[(table["bucket"] == "baseline_H") & (table["scale"] == scale)].copy()
    if base.empty:
        raise RuntimeError("no baseline_H runs at this scale")

    split_keys = {
        "Default-Greedy (train dist.)": EVAL_SPLITS["default"],
        "20Step-Greedy (horizon ↑)": EVAL_SPLITS["20step"],
        "PoisonApple (instruction flip)": EVAL_SPLITS["poison"],
    }
    # Subset to columns we have.
    split_keys = {label: col for label, col in split_keys.items() if col in base.columns}

    H_values = sorted([h for h in base["H"].dropna().unique()])
    n_splits = len(split_keys)
    n_H = len(H_values)
    bar_w = 0.8 / max(n_splits, 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, col) in enumerate(split_keys.items()):
        means, ses, jitters_y, jitters_x = [], [], [], []
        for j, H_val in enumerate(H_values):
            vals = base.loc[base["H"] == H_val, col].dropna().values
            if len(vals) == 0:
                means.append(np.nan); ses.append(0); continue
            means.append(vals.mean())
            ses.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            x_centre = j + (i - (n_splits - 1) / 2) * bar_w
            jitters_x.extend(np.random.default_rng(j * 10 + i).normal(x_centre, 0.015, size=len(vals)))
            jitters_y.extend(vals)
        xs = np.arange(n_H) + (i - (n_splits - 1) / 2) * bar_w
        bars = ax.bar(xs, means, width=bar_w * 0.85, label=label,
                      yerr=ses, capsize=2.5, alpha=0.85)
        ax.scatter(jitters_x, jitters_y, s=10, color="black", alpha=0.4, zorder=3)
    ax.set_xticks(np.arange(n_H))
    ax.set_xticklabels([f"H={h:g}\n(n={base[base['H']==h].shape[0]})" for h in H_values])
    ax.set_ylabel("final-step reward")
    ax.set_title(f"Snake — held-out eval performance vs entropy coefficient ({scale} baseline)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_perf_heldout.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = plot_curves_per_H()
    print(f"wrote {p1}")
    p2 = plot_heldout_bars()
    print(f"wrote {p2}")
