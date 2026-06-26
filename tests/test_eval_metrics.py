"""Unit tests for pass@k / best-of-group eval aggregation. CPU, no GPU/model."""

import math

import numpy as np
import pytest

from verl.trainer.ppo.eval_metrics import (
    best_of_group,
    expected_best_of_k,
    pass_at_k_binary,
    compute_group_score_metrics,
)


# --- best_of_group ------------------------------------------------------------

def test_best_of_group_takes_max_per_group_then_mean():
    # 2 groups of 3: maxes 3 and 9 -> mean 6
    assert best_of_group([1, 2, 3, 4, 5, 9], n_groups=2, group_size=3) == pytest.approx(6.0)


def test_best_of_group_single_group():
    assert best_of_group([1.0, 7.0, 3.0], n_groups=1, group_size=3) == pytest.approx(7.0)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        best_of_group([1, 2, 3], n_groups=2, group_size=3)


# --- expected_best_of_k (continuous, unbiased) --------------------------------

def test_exp_best_at_1_equals_mean():
    vals = [0.0, 1.0, 2.0, 3.0]
    out = expected_best_of_k(vals, n_groups=1, group_size=4, ks=[1])
    assert out[1] == pytest.approx(np.mean(vals))


def test_exp_best_at_group_size_equals_best_of_group():
    vals = [0.0, 5.0, 2.0, 9.0, 1.0, 1.0]
    out = expected_best_of_k(vals, n_groups=2, group_size=3, ks=[3])
    assert out[3] == pytest.approx(best_of_group(vals, 2, 3))


def test_exp_best_monotonic_in_k():
    vals = list(range(8))  # one group of 8
    out = expected_best_of_k(vals, n_groups=1, group_size=8, ks=[1, 2, 4, 8])
    seq = [out[k] for k in (1, 2, 4, 8)]
    assert all(a <= b for a, b in zip(seq, seq[1:]))
    assert out[8] == pytest.approx(7.0)  # best-of-all = max


def test_exp_best_of_k_matches_brute_force():
    # brute-force E[max of k-subset] over all C(n,k) subsets
    rng = np.random.default_rng(0)
    vals = rng.normal(size=6).tolist()
    from itertools import combinations
    for k in (1, 2, 3, 6):
        brute = np.mean([max(c) for c in combinations(vals, k)])
        got = expected_best_of_k(vals, n_groups=1, group_size=6, ks=[k])[k]
        assert got == pytest.approx(brute)


def test_exp_best_skips_k_above_group_size():
    out = expected_best_of_k([1, 2, 3], n_groups=1, group_size=3, ks=[1, 2, 4, 8])
    assert set(out) == {1, 2}


# --- pass_at_k_binary ---------------------------------------------------------

def test_pass_at_1_equals_success_rate():
    succ = [1, 0, 0, 0]  # 1/4 success
    assert pass_at_k_binary(succ, n_groups=1, group_size=4, ks=[1])[1] == pytest.approx(0.25)


def test_pass_at_k_all_success_is_one():
    succ = [1, 1, 1, 1]
    out = pass_at_k_binary(succ, n_groups=1, group_size=4, ks=[1, 2, 4])
    assert all(v == pytest.approx(1.0) for v in out.values())


def test_pass_at_k_no_success_is_zero():
    succ = [0, 0, 0, 0]
    out = pass_at_k_binary(succ, n_groups=1, group_size=4, ks=[1, 2, 4])
    assert all(v == pytest.approx(0.0) for v in out.values())


def test_pass_at_k_codex_formula():
    # n=4, c=1: pass@2 = 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    succ = [1, 0, 0, 0]
    assert pass_at_k_binary(succ, n_groups=1, group_size=4, ks=[2])[2] == pytest.approx(0.5)


def test_pass_at_k_treats_any_positive_as_success():
    assert pass_at_k_binary([0.0, 2.5, -1.0, 0.0], n_groups=1, group_size=4, ks=[1])[1] == pytest.approx(0.25)


def test_pass_at_k_monotonic_in_k():
    succ = [1, 0, 0, 0, 0, 0, 0, 0]
    out = pass_at_k_binary(succ, n_groups=1, group_size=8, ks=[1, 2, 4, 8])
    seq = [out[k] for k in (1, 2, 4, 8)]
    assert all(a <= b for a, b in zip(seq, seq[1:]))
    assert out[8] == pytest.approx(1.0)


# --- compute_group_score_metrics (assembly) -----------------------------------

def test_compute_assembles_all_keys():
    cont = [0.0, 1.0, 2.0, 3.0]
    binar = [0, 1, 0, 1]
    m = compute_group_score_metrics(1, 4, continuous_values=cont, binary_successes=binar, ks=[1, 2, 4])
    assert "passk/best_of_group_mean" in m
    assert {"passk/exp_best_at_1", "passk/exp_best_at_2", "passk/exp_best_at_4"} <= set(m)
    assert {"passk/solve_at_1", "passk/solve_at_2", "passk/solve_at_4"} <= set(m)
    assert m["passk/best_of_group_mean"] == pytest.approx(3.0)


def test_compute_skips_wrong_length_silently():
    m = compute_group_score_metrics(2, 4, continuous_values=[1, 2, 3], binary_successes=None)
    assert m == {}


def test_compute_handles_none_inputs():
    m = compute_group_score_metrics(1, 4, continuous_values=None, binary_successes=None)
    assert m == {}
