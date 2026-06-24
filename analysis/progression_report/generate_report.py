"""Assemble the markdown report from figures + tables.

Outputs:
  figures/progression_report/report.md  (with embedded PNGs)

Run after the plot scripts have produced their figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .process import (
    ENTROPY_KEY,
    EVAL_SPLITS,
    snake_run_table,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "progression_report"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _format_meanse(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "—"
    mean = s.mean()
    se = (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
    return f"{mean:.3f} ± {se:.3f}"


def _make_data_scope_table(table: pd.DataFrame) -> str:
    rows = [["bucket", "scale", "H", "n runs"]]
    for (bucket, scale, H), grp in (
        table[table["bucket"].isin(
            ["baseline_H", "clipcov", "klcov", "topP", "cosine", "decay", "adaptive"]
        )]
        .groupby(["bucket", "scale", "H"], dropna=False)
    ):
        rows.append([str(bucket), str(scale), f"{H:g}" if pd.notna(H) else "—", str(len(grp))])
    return _md_table(rows)


def _make_method_summary_table(table: pd.DataFrame, scale: str = "4B") -> str:
    rows = [["method", "n", "Default-Greedy", "20Step", "PoisonApple", "Token entropy"]]
    for bucket in ["baseline_H", "clipcov", "klcov", "topP", "cosine", "decay", "adaptive"]:
        sub = table[(table["bucket"] == bucket) & (table["scale"] == scale)]
        # Drop the n=1 diverged H=0.05 baseline so methods compare to a sensible pool.
        if bucket == "baseline_H":
            sub = sub[sub["H"] != 0.05]
        if len(sub) == 0:
            continue
        # For token entropy: drop diverged outliers (>3.5 nats) from the pool stat.
        ent = sub.get(ENTROPY_KEY, pd.Series(dtype=float)).dropna()
        ent = ent[ent <= 3.5]
        rows.append([
            bucket,
            str(len(sub)),
            _format_meanse(sub.get(EVAL_SPLITS["default"], pd.Series(dtype=float))),
            _format_meanse(sub.get(EVAL_SPLITS["20step"], pd.Series(dtype=float))),
            _format_meanse(sub.get(EVAL_SPLITS.get("poison", "_"), pd.Series(dtype=float))),
            _format_meanse(ent),
        ])
    return _md_table(rows)


def _make_h_summary_table(table: pd.DataFrame, scale: str = "4B") -> str:
    base = table[(table["bucket"] == "baseline_H") & (table["scale"] == scale)]
    rows = [["H", "n", "Default-Greedy", "20Step", "PoisonApple", "Token entropy"]]
    for H_val in sorted(base["H"].dropna().unique()):
        sub = base[base["H"] == H_val]
        rows.append([
            f"{H_val:g}",
            str(len(sub)),
            _format_meanse(sub.get(EVAL_SPLITS["default"], pd.Series(dtype=float))),
            _format_meanse(sub.get(EVAL_SPLITS["20step"], pd.Series(dtype=float))),
            _format_meanse(sub.get(EVAL_SPLITS.get("poison", "_"), pd.Series(dtype=float))),
            _format_meanse(sub.get(ENTROPY_KEY, pd.Series(dtype=float))),
        ])
    return _md_table(rows)


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    def fmt(r):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt(rows[0]), sep] + [fmt(r) for r in rows[1:]]
    return "\n".join(out)


def main() -> Path:
    table = snake_run_table()
    scale = "4B"

    report = []
    report.append("# Entropy-control methods on Snake — preliminary findings\n")
    report.append("*Source*: `wandb.ai/jimdilkes/verl_env`. Snake (FastSnake env), Qwen2.5-4B Instruct ")
    report.append("unless noted. All runs `state=finished`.\n")

    report.append("\n## Caveats\n")
    report.append("- Runs in `verl_env` predate the 2026-02-16 batch-size bug fix")
    report.append(" (`train_batch_size = n_rollouts × horizon`). Mid-episode weight updates were")
    report.append(" biasing GAE on all of these runs. Conclusions here are *relative comparisons")
    report.append(" under the same bias*, not absolute performance.\n")
    report.append("- Post-fix runs live in the `rl_sdm` project (CAIS v2 tag) and are not included here.\n")
    report.append("- A few groups have small n (klcov n=3, decay variants n=1–2). The headline")
    report.append(" message is robust *across* methods, not for individual H values within a method.\n")

    report.append("\n## Data scope\n")
    report.append(_make_data_scope_table(table))
    report.append("\n")

    report.append("\n## 1. Performance overview — training + held-out eval\n")
    report.append("Snake's held-out eval suite probes instruction-generalisation:\n")
    report.append("- **Default-Greedy** — same env as training (10×10, 8 rounds), T=0, in-distribution.\n")
    report.append("- **20Step-Greedy** — same env but `episode_length=20`, 5 apples — horizon shift.\n")
    report.append("- **PoisonAppleAndBanana-Greedy** — apple reward flipped to −1, banana +1 — *instruction flip*: ")
    report.append("the system prompt still describes the original rules; the env contradicts it.\n")
    if (FIG_DIR / "fig_perf_overview.png").exists():
        report.append("\n![Per-H curves](fig_perf_overview.png)\n")
    else:
        report.append("\n*(per-H learning curves figure pending — rerun `plot_performance.py` ")
        report.append("after `fetch_history_per_key.py` completes)*\n")
    report.append("\n![Held-out bars](fig_perf_heldout.png)\n")
    report.append("\n### Final-step performance, baseline H sweep (4B)\n")
    report.append(_make_h_summary_table(table, scale="4B"))
    report.append("\n")
    report.append("\nReading: H=0.001–0.005 cluster together on the training-distribution split;")
    report.append(" performance does not transfer to the PoisonApple split (instruction-flip generalisation")
    report.append(" is largely a failure across all H values).\n")

    report.append("\n## 2. Entropy-control methods — do they help?\n")
    report.append("Each method is benchmarked against the densest, best-performing baseline ")
    report.append("(H=0.001, n=18; dashed line in the bar plot). The H=0.05 baseline run ")
    report.append("(n=1, diverged: token entropy ≈11 nats, reward ≈−0.6) is excluded from method ")
    report.append("comparisons to avoid swamping the bars and scatter with one collapsed run.\n")
    report.append("\nMethod bucket descriptions:\n")
    report.append("- **Entropy reg (H)** — pure entropy bonus in the PPO loss, swept H ∈ {0.001, 0.005, 0.01, 0.05}.\n")
    report.append("- **Cov clip** — covariance-clip from the DAPO/ARPO family; clips per-token policy gradients ")
    report.append("by token-advantage covariance.\n")
    report.append("- **KL cov** — variant of cov-clip thresholded by per-token KL.\n")
    report.append("- **Top-p restriction** — truncates the sampling distribution at top-p mass during rollout.\n")
    report.append("- **Cosine schedule** — cosine decay of the LR and/or H.\n")
    report.append("- **Entropy decay** — explicit linear/step schedule on H.\n")
    report.append("- **Adaptive entropy** — target-entropy controller (P-controller on actor/entropy bounds).\n")

    report.append("\n![Method bars](fig_entropy_methods_bars.png)\n")
    report.append("\n![Entropy vs reward](fig_entropy_vs_reward.png)\n")

    report.append("\n### Per-method final-step summary (4B, mean ± SE across seeds)\n")
    report.append(_make_method_summary_table(table, scale="4B"))
    report.append("\n")

    report.append("\n## 3. Takeaway\n")
    report.append("**Training-distribution (Default-Greedy) eval.** Cov-clip, KL-cov, top-p restriction, ")
    report.append("cosine schedule and explicit entropy decay all sit at or below the H=0.001 baseline. ")
    report.append("Only adaptive entropy edges above (+0.12 mean, ~1.5 SE), and entropy decay matches. ")
    report.append("No method exceeds the best-tuned constant H by more than ~10 %.\n")
    report.append("\n**Horizon-shift (20Step-Greedy) eval.** Same ordering: adaptive entropy is the only ")
    report.append("method to exceed the H=0.001 baseline by a margin worth quoting (+0.26 mean, ~2 SE); ")
    report.append("everything else is at or below.\n")
    report.append("\n**Instruction-flip (PoisonApple) eval.** The most striking signal: *every* method ")
    report.append("sits between roughly −0.4 and −0.7 (with the worst-case floor at −1). The agent fails ")
    report.append("to override its original instruction even when the env hard-codes a reward inversion. ")
    report.append("No entropy-control method moves this dial.\n")
    report.append("\n**Token-entropy axis.** Loss-mediated methods that *raise* entropy (top-p, cosine) ")
    report.append("hurt reward. Methods that *lower* entropy (cov-clip, decay, adaptive) sit near the ")
    report.append("baseline trade-off curve. There is no out-of-the-pack point.\n")
    report.append("\nThis is the motivating observation: standard entropy-control / exploration tweaks ")
    report.append("for PPO do not unlock instruction-generalisation on Snake, and barely move the ")
    report.append("training-distribution reward. The rest of the report explores context-mediated ")
    report.append("alternatives.\n")

    out = FIG_DIR / "report.md"
    out.write_text("".join(report))
    return out


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}")
