"""Group-based eval score aggregation: pass@k / best-of-group.

Eval rollouts are organised into groups of ``seed_group_size`` rollouts that share a
resampled initial environment seed (so within-group trajectories are directly comparable).
The standard evaluator logs only per-group / global mean and std of the task score. These
helpers add the diversity-vs-commitment view: how good the BEST trajectory in a group is,
and how that scales with the number of attempts k.

We record everything (the harness philosophy: never re-run for a forgotten metric) and let
the paper choose which to report:
  - ``best_of_group``           : mean over groups of the max score in the group (= exp_best_at_{G}).
  - ``expected_best_of_k``      : unbiased E[max of a random k-subset], continuous score.
  - ``pass_at_k_binary``        : Codex-style solve-rate pass@k on a binary success signal.

All inputs are flat per-rollout arrays in GLOBAL rollout order (rollout i -> group
``i // group_size``), length ``n_groups * group_size``.
"""

import math
from typing import Dict, Sequence

import numpy as np

DEFAULT_PASS_AT_KS = (1, 2, 4, 8, 16)


def _as_groups(values: Sequence[float], n_groups: int, group_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != n_groups * group_size:
        raise ValueError(
            f"eval_metrics: expected {n_groups * group_size} values "
            f"({n_groups} groups x {group_size}), got {arr.size}."
        )
    return arr.reshape(n_groups, group_size)


def best_of_group(values: Sequence[float], n_groups: int, group_size: int) -> float:
    """Mean over groups of the maximum score within each group (= expected_best_of_k at k=group_size)."""
    return float(_as_groups(values, n_groups, group_size).max(axis=1).mean())


def _expected_best_of_k_single(sorted_asc: np.ndarray, k: int) -> float:
    """Unbiased E[max of k samples drawn without replacement] from one group.

    With group scores sorted ascending s_0..s_{n-1}, the probability that s_i is the maximum
    of a uniformly-random k-subset is C(i, k-1) / C(n, k) (the other k-1 picks come from the
    i strictly-smaller elements), so E[max_k] = sum_{i>=k-1} s_i * C(i, k-1) / C(n, k).
    """
    n = sorted_asc.shape[0]
    k = min(k, n)
    denom = math.comb(n, k)
    total = 0.0
    for i in range(k - 1, n):
        total += float(sorted_asc[i]) * math.comb(i, k - 1)
    return total / denom


def expected_best_of_k(
    values: Sequence[float], n_groups: int, group_size: int, ks: Sequence[int] = DEFAULT_PASS_AT_KS
) -> Dict[int, float]:
    """Continuous pass@k: mean over groups of the unbiased expected best-of-k score."""
    groups = _as_groups(values, n_groups, group_size)
    sorted_groups = np.sort(groups, axis=1)
    out: Dict[int, float] = {}
    for k in ks:
        if k > group_size:
            continue
        out[int(k)] = float(np.mean([_expected_best_of_k_single(sorted_groups[g], k) for g in range(n_groups)]))
    return out


def pass_at_k_binary(
    successes: Sequence[float], n_groups: int, group_size: int, ks: Sequence[int] = DEFAULT_PASS_AT_KS
) -> Dict[int, float]:
    """Codex-style pass@k on a binary success signal, averaged over groups.

    Per group with c successes among n rollouts: pass@k = 1 - C(n-c, k)/C(n, k) (the chance that
    a random k-subset contains at least one success). ``successes`` may be 0/1 or any >0 flag.
    """
    groups = _as_groups([1.0 if float(x) > 0 else 0.0 for x in successes], n_groups, group_size)
    n = group_size
    out: Dict[int, float] = {}
    for k in ks:
        if k > group_size:
            continue
        vals = []
        for g in range(n_groups):
            c = int(round(float(groups[g].sum())))
            if n - c < k:
                vals.append(1.0)
            else:
                vals.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
        out[int(k)] = float(np.mean(vals))
    return out


def compute_group_score_metrics(
    n_groups: int,
    group_size: int,
    *,
    continuous_values: Sequence[float] = None,
    binary_successes: Sequence[float] = None,
    ks: Sequence[int] = DEFAULT_PASS_AT_KS,
    prefix: str = "passk",
) -> Dict[str, float]:
    """Assemble the full pass@k / best-of-group metric dict (records every variant).

    continuous_values: per-rollout task score (or reward) -> best_of_group + exp_best_at_{k}.
    binary_successes:  per-rollout >0 success flag        -> solve_at_{k}.
    Returns {} for any input that is None / wrong length (skip silently, never crash an eval).
    """
    metrics: Dict[str, float] = {}
    expected = n_groups * group_size
    if continuous_values is not None and len(continuous_values) == expected and group_size >= 1:
        metrics[f"{prefix}/best_of_group_mean"] = best_of_group(continuous_values, n_groups, group_size)
        for k, v in expected_best_of_k(continuous_values, n_groups, group_size, ks).items():
            metrics[f"{prefix}/exp_best_at_{k}"] = v
    if binary_successes is not None and len(binary_successes) == expected and group_size >= 1:
        for k, v in pass_at_k_binary(binary_successes, n_groups, group_size, ks).items():
            metrics[f"{prefix}/solve_at_{k}"] = v
    return metrics
