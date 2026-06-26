"""Unit tests for advantage-weighted KL_S (AWR, V2 part C). CPU, no GPU."""

import math
import pytest
import torch

from verl.trainer.ppo.distill_kl import compute_awr_weights, masked_sample_mean


# --- compute_awr_weights -------------------------------------------------------

def test_equal_returns_give_uniform_weights():
    w = compute_awr_weights([1.0, 1.0, 1.0, 1.0], [True, True, True, True], temp=1.0)
    assert all(abs(x - 1.0) < 1e-9 for x in w)  # std=0 -> adv=0 -> exp(0)=1


def test_higher_return_gets_higher_weight_and_positive():
    w = compute_awr_weights([0.0, 1.0, 2.0], [True, True, True], temp=1.0)
    assert w[0] < w[1] < w[2]
    assert all(x > 0 for x in w)  # always positive (never anti-distill)


def test_non_instructed_get_zero_weight():
    w = compute_awr_weights([5.0, 0.0, 9.0], [True, False, True], temp=1.0)
    assert w[1] == 0.0 and w[0] > 0 and w[2] > 0


def test_zscore_is_over_instructed_only():
    # env1 (non-instructed, huge return) must not shift the instructed mean/std
    w_with = compute_awr_weights([0.0, 100.0, 2.0], [True, False, True], temp=1.0)
    w_without = compute_awr_weights([0.0, 2.0], [True, True], temp=1.0)
    assert w_with[0] == pytest.approx(w_without[0])
    assert w_with[2] == pytest.approx(w_without[1])


def test_large_temp_approaches_uniform():
    w = compute_awr_weights([0.0, 1.0, 5.0], [True, True, True], temp=1e6)
    assert all(abs(x - 1.0) < 1e-3 for x in w)


def test_cap_limits_weight():
    w = compute_awr_weights([0.0, 10.0], [True, True], temp=0.5, cap=2.0)
    assert max(w) <= 2.0 + 1e-9


def test_empty_instructed_all_zero():
    assert compute_awr_weights([1.0, 2.0], [False, False], temp=1.0) == [0.0, 0.0]


def test_temp_must_be_positive():
    with pytest.raises(ValueError):
        compute_awr_weights([0.0, 1.0], [True, True], temp=0.0)
    with pytest.raises(ValueError):
        compute_awr_weights([0.0, 1.0], [True, True], temp=-1.0)


def test_cap_must_be_positive():
    with pytest.raises(ValueError):
        compute_awr_weights([0.0, 1.0], [True, True], temp=1.0, cap=0.0)
    with pytest.raises(ValueError):
        compute_awr_weights([0.0, 1.0], [True, True], temp=1.0, cap=-2.0)


def test_small_temp_large_return_no_overflow():
    # tiny temp + large z-advantage must NOT raise OverflowError; weights stay finite and
    # non-negative (the low one may underflow to 0 = the only-best limit, never negative).
    w = compute_awr_weights([0.0, 1000.0], [True, True], temp=1e-3)
    assert all(math.isfinite(x) and x >= 0 for x in w)
    assert w[1] > w[0] and w[1] > 0  # max-advantage sample keeps a positive (>=1) weight


# --- weighted masked_sample_mean ----------------------------------------------

def test_weighted_mean_none_is_uniform_regression():
    seq = torch.tensor([1.0, 2.0, 3.0])
    keep = torch.tensor([1.0, 1.0, 1.0])
    assert masked_sample_mean(seq, keep).item() == pytest.approx(2.0)
    assert masked_sample_mean(seq, keep, weights=None).item() == pytest.approx(2.0)


def test_weighted_mean_preserves_scale_on_constant():
    # weighted mean of a constant seq == that constant (scale preserved)
    seq = torch.tensor([5.0, 5.0, 5.0])
    keep = torch.tensor([1.0, 1.0, 1.0])
    w = torch.tensor([0.1, 1.0, 9.0])
    assert masked_sample_mean(seq, keep, weights=w).item() == pytest.approx(5.0)


def test_weighted_mean_emphasizes_high_weight():
    seq = torch.tensor([0.0, 10.0])
    keep = torch.tensor([1.0, 1.0])
    hi = masked_sample_mean(seq, keep, weights=torch.tensor([1.0, 9.0])).item()
    assert hi > masked_sample_mean(seq, keep).item()  # > unweighted mean (5.0)
    assert hi == pytest.approx((0*1 + 10*9) / (1 + 9))


def test_weighted_mean_respects_keep_mask():
    seq = torch.tensor([1.0, 100.0, 3.0])
    keep = torch.tensor([1.0, 0.0, 1.0])  # drop the big one
    w = torch.tensor([1.0, 50.0, 1.0])
    assert masked_sample_mean(seq, keep, weights=w).item() == pytest.approx(2.0)


def test_weighted_mean_all_zero_keep_is_zero():
    seq = torch.tensor([1.0, 2.0])
    assert masked_sample_mean(seq, torch.zeros(2), weights=torch.tensor([3.0, 4.0])).item() == 0.0
