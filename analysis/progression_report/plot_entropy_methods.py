"""Entropy-method comparison figure for the PhD progression report.

Each method bucket (clipcov, klcov, topP, cosine, decay, adaptive, plus a
baseline-H reference) is summarised by three final-step metrics:

- Held-out reward (Default-Greedy as the canonical eval split)
- Held-out 20Step reward (horizon shift)
- Token entropy (actor/entropy)

We also include a scatter of (final entropy, final reward) annotated by
method — this is the headline "entropy doesn't buy you reward" plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .process import (
    ENTROPY_KEY,
    EVAL_SPLITS,
    snake_run_table,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "progression_report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "baseline_H",
    "clipcov",
    "klcov",
    "topP",
    "cosine",
    "decay",
    "adaptive",
]
METHOD_LABELS = {
    "baseline_H": "Entropy reg (H)",
    "clipcov": "Cov clip",
    "klcov": "KL cov",
    "topP": "Top-p restriction",
    "cosine": "Cosine schedule",
    "decay": "Entropy decay",
    "adaptive": "Adaptive entropy",
}
METHOD_COLORS = {
    "baseline_H": "#2C7BB6",
    "clipcov": "#D73027",
    "klcov": "#A50026",
    "topP": "#F46D43",
    "cosine": "#FDAE61",
    "decay": "#FEE090",
    "adaptive": "#5E3C99",
}


def _final_table(scale: str = "4B") -> pd.DataFrame:
    """Per-run (bucket-row) table of final-step metrics + reference H baseline.

    Drops the H=0.05 baseline_H run — n=1 catastrophic divergence (token entropy
    ≈11 nats, reward ≈−0.6). Keeping it would distort the baseline pool and
    swamp the entropy y-axis with a single outlier.
    """
    table = snake_run_table()
    sub = table[(table["scale"] == scale) & table["bucket"].isin(METHOD_ORDER)].copy()
    sub = sub[~((sub["bucket"] == "baseline_H") & (sub["H"] == 0.05))]
    return sub


BEST_BASELINE_H = 0.001  # densest baseline (n=18) and the highest per-H mean


def _baseline_best_H(table: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Pick the H that gives the highest mean ``metric`` from baseline_H. Return (H, mean)."""
    base = table[table["bucket"] == "baseline_H"]
    if base.empty:
        return (np.nan, np.nan)
    means = base.groupby("H")[metric].mean()
    if means.empty:
        return (np.nan, np.nan)
    best_H = means.idxmax()
    return (best_H, means[best_H])


def plot_method_bars(scale: str = "4B") -> Path:
    """3-panel bar chart: methods × {default reward, 20step reward, token entropy}."""
    table = _final_table(scale)
    metrics = [
        (EVAL_SPLITS["default"], "Default-Greedy reward"),
        (EVAL_SPLITS["20step"], "20Step-Greedy reward"),
        (ENTROPY_KEY, "final token entropy"),
    ]
    metrics = [(k, lbl) for k, lbl in metrics if k in table.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.3))
    if len(metrics) == 1:
        axes = [axes]

    ent_cap = 3.5
    for ax, (metric, label) in zip(axes, metrics):
        means, ses, ns, xticks, jitter_x, jitter_y, bar_colors = [], [], [], [], [], [], []
        clipped_pts: list[tuple[float, float]] = []
        for j, bucket in enumerate(METHOD_ORDER):
            sub_full = table[(table["bucket"] == bucket)][metric].dropna()
            if metric == ENTROPY_KEY:
                # Compute bar/SE on in-range only — outliers shown as ↑ annotations.
                in_range = sub_full[sub_full <= ent_cap]
                out_range = sub_full[sub_full > ent_cap]
            else:
                in_range = sub_full
                out_range = pd.Series(dtype=float)
            if len(in_range) == 0:
                means.append(np.nan); ses.append(0); ns.append(len(sub_full))
                xticks.append(METHOD_LABELS[bucket]); bar_colors.append(METHOD_COLORS[bucket])
                continue
            means.append(in_range.mean())
            ses.append(in_range.std(ddof=1) / np.sqrt(len(in_range)) if len(in_range) > 1 else 0)
            ns.append(len(sub_full))
            xticks.append(METHOD_LABELS[bucket]); bar_colors.append(METHOD_COLORS[bucket])
            rng = np.random.default_rng(j)
            jitter_x.extend(rng.normal(j, 0.06, size=len(in_range)))
            jitter_y.extend(in_range.values)
            for val in out_range.values:
                clipped_pts.append((j, val))
        bars = ax.bar(np.arange(len(METHOD_ORDER)), means, color=bar_colors,
                      yerr=ses, capsize=2.5, alpha=0.85, edgecolor="black", lw=0.4)
        ax.scatter(jitter_x, jitter_y, s=12, color="black", alpha=0.55, zorder=3)
        # Reference line at the best-H (densest) baseline mean.
        ref_vals = table[(table["bucket"] == "baseline_H") & (table["H"] == BEST_BASELINE_H)][metric].dropna()
        if len(ref_vals):
            ref_mean = ref_vals.mean()
            ax.axhline(ref_mean, ls="--", color="#2C7BB6", lw=1, alpha=0.6,
                       label=f"H={BEST_BASELINE_H:g} mean = {ref_mean:.3f} (n={len(ref_vals)})")
            ax.legend(loc="best", fontsize=7, frameon=False)
        ax.set_xticks(np.arange(len(METHOD_ORDER)))
        ax.set_xticklabels([f"{METHOD_LABELS[b]}\n(n={n})" for b, n in zip(METHOD_ORDER, ns)],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        if metric == ENTROPY_KEY:
            ax.set_ylim(0, ent_cap)
            for x, y in clipped_pts:
                ax.annotate(f"↑ {y:.1f}", (x, ent_cap), ha="center", va="top",
                            fontsize=7, color="red")
    fig.suptitle(f"Snake {scale} — entropy-control methods vs baseline H", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_entropy_methods_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_entropy_vs_reward(scale: str = "4B") -> Path:
    """Scatter of final token entropy vs Default-Greedy reward, colour-coded by method."""
    table = _final_table(scale)
    metric_x = ENTROPY_KEY
    metric_y = EVAL_SPLITS["default"]
    if metric_x not in table.columns or metric_y not in table.columns:
        raise RuntimeError("required metrics missing")

    fig, ax = plt.subplots(figsize=(7, 5))
    ent_cap = 3.0  # diverged runs sit far above this; show as triangles on x=cap.
    for bucket in METHOD_ORDER:
        sub = table[table["bucket"] == bucket][[metric_x, metric_y, "H"]].dropna(subset=[metric_x, metric_y])
        if sub.empty:
            continue
        c = METHOD_COLORS[bucket]
        in_range = sub[sub[metric_x] <= ent_cap]
        clipped = sub[sub[metric_x] > ent_cap]
        ax.scatter(in_range[metric_x], in_range[metric_y], s=42, color=c,
                   edgecolor="black", lw=0.4, alpha=0.85,
                   label=f"{METHOD_LABELS[bucket]} (n={len(sub)})")
        if len(clipped):
            ax.scatter([ent_cap] * len(clipped), clipped[metric_y], s=80, color=c,
                       edgecolor="black", lw=0.4, alpha=0.9, marker=">")
            for _, row in clipped.iterrows():
                ax.annotate(f"H={row[metric_x]:.1f}", (ent_cap, row[metric_y]),
                            xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7, color=c)
    ax.set_xlim(0, ent_cap * 1.05)
    ax.set_xlabel("final token entropy (actor/entropy)  [▶ = diverged, entropy past cap]")
    ax.set_ylabel("Default-Greedy reward (final)")
    ax.set_title(f"Snake {scale} — final entropy vs reward, by method")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_entropy_vs_reward.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_table(scale: str = "4B") -> Path:
    """CSV: bucket × {n, default mean ± SE, 20step mean ± SE, entropy mean ± SE}."""
    table = _final_table(scale)
    rows = []
    metrics = {
        "default_reward": EVAL_SPLITS["default"],
        "twenty_step_reward": EVAL_SPLITS["20step"],
        "poison_reward": EVAL_SPLITS.get("poison"),
        "final_entropy": ENTROPY_KEY,
    }
    for bucket in METHOD_ORDER:
        sub = table[table["bucket"] == bucket]
        row = {"bucket": bucket, "n": len(sub)}
        for label, col in metrics.items():
            if col is None or col not in sub.columns:
                row[f"{label}_mean"] = np.nan
                row[f"{label}_se"] = np.nan
                row[f"{label}_n"] = 0
                continue
            vals = sub[col].dropna()
            row[f"{label}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{label}_se"] = (vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0
            row[f"{label}_n"] = len(vals)
        rows.append(row)
    out = FIG_DIR / "summary_methods.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    p1 = plot_method_bars()
    print(f"wrote {p1}")
    p2 = plot_entropy_vs_reward()
    print(f"wrote {p2}")
    p3 = write_summary_table()
    print(f"wrote {p3}")
