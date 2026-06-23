"""Unit tests for DIME Asymmetric-RL/SD variation primitives (CPU, no GPU).

Run: python -m pytest tests/test_dime_distill.py -q
"""

import math

import pytest
import torch

from verl.trainer.ppo.distill_kl import (
    compute_distill_kl_mean_logprob,
    masked_sample_mean,
    compute_kl_filter_keep,
    validate_kl_estimator_config,
)
from verl.envs.environments.focus_instructions import (
    assign_focus_deterministic,
    validate_deterministic_assignment,
    has_dime_instructions,
)


# --- mean-logprob KL estimator -------------------------------------------------

def test_kl_mean_logprob_values():
    # B=2, T=3, full mask. mean over tokens then diff.
    student = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0]])
    teacher = torch.tensor([[-0.5, -0.5, -0.5], [-2.0, -2.0, -2.0]])
    mask = torch.ones(2, 3)
    kl_t, kl_s = compute_distill_kl_mean_logprob(student, teacher, mask)
    mean_s = student.mean(1)   # [-2, -1]
    mean_t = teacher.mean(1)   # [-0.5, -2]
    assert torch.allclose(kl_t, mean_t - mean_s)   # teacher branch
    assert torch.allclose(kl_s, mean_t - mean_s)   # student branch (same value, diff grad)


def test_kl_mean_logprob_respects_mask():
    student = torch.tensor([[-1.0, -2.0, 99.0]])  # 3rd token masked out
    teacher = torch.tensor([[-0.5, -0.5, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    kl_t, kl_s = compute_distill_kl_mean_logprob(student, teacher, mask)
    assert torch.allclose(kl_s, torch.tensor([(-0.5 + -0.5) / 2 - (-1.0 + -2.0) / 2]))


def test_kl_zero_token_row_is_zero_not_nan():
    # Terminal/empty row (all-zero mask) must yield finite 0 KL (verl masked_mean +1e-8),
    # so it can be safely excluded by a valid-row mask without poisoning the sum.
    student = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    teacher = torch.tensor([[-0.5, -0.5], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # row 1 empty
    kl_t, kl_s = compute_distill_kl_mean_logprob(student, teacher, mask)
    assert torch.isfinite(kl_s).all() and torch.isfinite(kl_t).all()
    assert kl_s[1].item() == 0.0


def test_dilution_fix_excludes_empty_rows():
    # Regression for P1: an empty instructed row (KL 0) must NOT dilute the per-sample
    # mean. keep weighting by valid-rows recovers the true mean over real rows.
    kl_s_seq = torch.tensor([2.0, 0.0])      # row1 is the empty terminal row
    keep = torch.tensor([1.0, 1.0])          # both instructed (broadcast over steps)
    valid = torch.tensor([1.0, 0.0])         # row1 has no response tokens
    assert masked_sample_mean(kl_s_seq, keep).item() == pytest.approx(1.0)        # diluted
    assert masked_sample_mean(kl_s_seq, keep * valid).item() == pytest.approx(2.0)  # fixed


def test_kl_grad_directions():
    student = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -0.5]], requires_grad=True)
    mask = torch.ones(1, 2)
    kl_t, kl_s = compute_distill_kl_mean_logprob(student, teacher, mask)
    kl_s.sum().backward(retain_graph=True)
    assert student.grad is not None and student.grad.abs().sum() > 0  # student-branch grad
    assert teacher.grad is None                                       # teacher detached in KL_S

    student.grad = None
    kl_t.sum().backward()
    assert teacher.grad is not None and teacher.grad.abs().sum() > 0  # teacher-branch grad
    assert student.grad is None                                       # student detached in KL_T


# --- source registration invariant (motivates config defaults) -----------------

def test_generic_source_always_available():
    assert has_dime_instructions("snake", "generic") is True
    assert has_dime_instructions("babyai", "generic") is True


def test_specific_source_only_for_registered_envs():
    # Only overcooked registers specific instructions; non-overcooked prompt configs
    # must default to source=generic or DIME-enable raises immediately.
    assert has_dime_instructions("overcooked", "specific") is True
    assert has_dime_instructions("snake", "specific") is False
    assert has_dime_instructions("babyai", "specific") is False


# --- estimator/teacher-KL config validation ------------------------------------

def test_validate_kl_estimator_rejects_unknown():
    with pytest.raises(ValueError):
        validate_kl_estimator_config("bogus", beta_teacher=0.0)


def test_validate_kl_estimator_rejects_mean_logprob_with_teacher_kl():
    # mean_logprob teacher term has the wrong minimization direction (reverse KL needs
    # a score-function term) — must be rejected, not silently run.
    with pytest.raises(ValueError):
        validate_kl_estimator_config("mean_logprob", beta_teacher=0.1)


def test_validate_kl_estimator_allows_valid_combos():
    validate_kl_estimator_config("mean_logprob", beta_teacher=0.0)  # student-only KL
    validate_kl_estimator_config("k3", beta_teacher=0.1)            # k3 handles teacher dir
    validate_kl_estimator_config("k3", beta_teacher=0.0)


# --- k3 student KL argument orientation -----------------------------------------

def test_k3_student_kl_orientation_bounded():
    # KL_S must call kl_penalty_forward(teacher.detach(), student, 'k3'): k3 estimates
    # KL(A||B) for samples from A, and samples are teacher rollouts. With this order the
    # student gradient is exp(student-teacher)-1 ∈ (-1,0) when student<teacher (bounded,
    # pulls student up). The swapped order gives 1-exp(teacher-student), which EXPLODES
    # (≈ -1+π_T/π_S) for student≪teacher — this test fails under the wrong arg order.
    from verl.trainer.ppo.core_algos import kl_penalty_forward
    teacher = torch.tensor([-0.5])
    student = torch.tensor([-5.0], requires_grad=True)  # far below teacher
    kl_penalty_forward(teacher.detach(), student, 'k3').sum().backward()
    g = student.grad.item()
    assert -1.0 < g < 0.0


# --- masked_sample_mean (filter application) -----------------------------------

def test_masked_sample_mean():
    seq = torch.tensor([1.0, 2.0, 3.0, 4.0])
    keep = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert masked_sample_mean(seq, keep).item() == pytest.approx((1.0 + 3.0) / 2)


def test_masked_sample_mean_all_zero_is_zero():
    seq = torch.tensor([5.0, 6.0])
    keep = torch.zeros(2)
    assert masked_sample_mean(seq, keep).item() == 0.0


# --- KL_S filter (per-episode keep) --------------------------------------------

def test_filter_none_keeps_all_instructed():
    has_inst = [True, False, True, True]
    keep = compute_kl_filter_keep([0.0, 5.0, -1.0, 2.0], has_inst, "none")
    assert keep == [True, False, True, True]  # non-instructed always False


def test_filter_return_positive():
    has_inst = [True, True, True, False]
    keep = compute_kl_filter_keep([0.5, -0.1, 0.0, 9.0], has_inst, "return_positive")
    assert keep == [True, False, False, False]  # >0 only, env3 not instructed


def test_filter_top_pct():
    has_inst = [True, True, True, True]
    # 4 instructed, top 50% -> ceil(0.5*4)=2 highest returns kept
    keep = compute_kl_filter_keep([1.0, 4.0, 3.0, 2.0], has_inst, "top_pct", top_pct=0.5)
    assert keep == [False, True, True, False]


def test_filter_top_pct_ignores_non_instructed():
    has_inst = [True, False, True]
    # 2 instructed (idx 0,2); top 50% -> ceil(1)=1
    keep = compute_kl_filter_keep([9.0, 100.0, 1.0], has_inst, "top_pct", top_pct=0.5)
    assert keep == [True, False, False]  # env1 (highest return) excluded: not instructed


def test_filter_bad_mode_raises():
    with pytest.raises(ValueError):
        compute_kl_filter_keep([1.0], [True], "bogus")


def test_filter_bad_top_pct_raises():
    with pytest.raises(ValueError):
        compute_kl_filter_keep([1.0], [True], "top_pct", top_pct=0.0)


# --- deterministic assignment --------------------------------------------------

def test_deterministic_coverage():
    instr = ["a", "b", "c"]
    out = assign_focus_deterministic(8, instr, n_duplicates=2, n_no_instruction=2, seed=0)
    assert len(out) == 8
    assert out.count("a") == 2 and out.count("b") == 2 and out.count("c") == 2
    assert out.count(None) == 2


def test_deterministic_reproducible_and_shuffled():
    instr = ["a", "b", "c"]
    out1 = assign_focus_deterministic(8, instr, 2, 2, seed=42)
    out2 = assign_focus_deterministic(8, instr, 2, 2, seed=42)
    out3 = assign_focus_deterministic(8, instr, 2, 2, seed=43)
    assert out1 == out2          # reproducible
    # different seed almost surely reorders (not guaranteed, but check it's a valid perm)
    assert sorted(out3, key=lambda x: (x is None, x)) == sorted(out1, key=lambda x: (x is None, x))


def test_deterministic_group_size_validation():
    with pytest.raises(ValueError):
        validate_deterministic_assignment(n_rollouts=7, n_instructions=3, n_duplicates=2, n_no_instruction=2)
    # 3*2 + 2 = 8 ok
    validate_deterministic_assignment(n_rollouts=8, n_instructions=3, n_duplicates=2, n_no_instruction=2)
