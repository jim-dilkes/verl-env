"""Paper-faithful sampled-action teacher/student KL for ICE (ICE π-distill).

The branch's default KL estimator is Schulman ``k3`` (per-token, via
``kl_penalty_forward``). This module adds the paper's **mean-logprob** estimator
(EMNLP-26 ICE π-distill / OpenRLHF-DICE ``openrlhf/dice/distill_kl.py``): a
per-sample mean over response tokens of the teacher/student logprob difference.

Stop-gradient directions:

    teacher branch:  KL_T = mean_t[ log π_T(a_t) - sg(log π_S(a_t)) ]   (grad → teacher)
    student branch:  KL_S = mean_t[ sg(log π_T(a_t)) - log π_S(a_t) ]   (grad → student)

In dp_actor the student and teacher forwards share the SAME response tokens and the
SAME response_mask (only the prompt — observation ± focus text — differs), so the
two per-branch masked means are taken over one shared index set and subtracted.
"""

import math
from typing import List, Sequence, Tuple

import torch

from verl.utils.torch_functional import masked_mean


def compute_kl_filter_keep(
    episode_returns: Sequence[float],
    has_instruction: Sequence[bool],
    kl_filter: str,
    top_pct: float = 0.5,
) -> List[bool]:
    """Per-env keep mask selecting which teacher rollouts contribute the student KL_S.

    Selects among INSTRUCTED episodes only (the teacher rollouts); non-instructed
    envs are always False (the actor masks KL_S by has_instruction regardless).
    Returned per-env list[bool] is later broadcast to all steps of that env.

    Modes:
      none            — keep all instructed episodes.
      return_positive — instructed episodes with episode_return > 0.
      top_pct         — top ceil(top_pct * n_instructed) instructed episodes by return.
    """
    n = len(has_instruction)
    returns = [float(episode_returns[i]) for i in range(n)]
    inst_idx = [i for i in range(n) if has_instruction[i]]
    keep = [False] * n

    if kl_filter == "none":
        for i in inst_idx:
            keep[i] = True
    elif kl_filter == "return_positive":
        for i in inst_idx:
            keep[i] = returns[i] > 0.0
    elif kl_filter == "top_pct":
        if not (0.0 < top_pct <= 1.0):
            raise ValueError(f"ice.kl_filter_top_pct={top_pct} must be in (0, 1].")
        n_keep = math.ceil(top_pct * len(inst_idx)) if inst_idx else 0
        for i in sorted(inst_idx, key=lambda j: returns[j], reverse=True)[:n_keep]:
            keep[i] = True
    else:
        raise ValueError(
            f"ice.kl_filter={kl_filter!r} must be 'none', 'return_positive', or 'top_pct'."
        )
    return keep


def validate_kl_estimator_config(kl_estimator: str, beta_teacher: float) -> None:
    """Reject illegal kl_estimator / teacher-KL combinations.

    The mean_logprob estimator is only correct for the STUDENT (forward) KL:
    KL_S is estimated on teacher samples as mean(sg[logπ_T] − logπ_S), whose
    gradient w.r.t. the student is behaviour-cloning toward teacher actions
    (bounded, correct). The TEACHER term is a reverse KL D_KL(π_T ‖ sg[π_S])
    whose sampling distribution depends on the teacher params; the analogous
    mean(logπ_T) − sg(mean logπ_S) is NOT a valid reverse-KL gradient — descending
    it merely lowers logπ_T on the sampled actions (degenerate), so we forbid it.
    Use kl_estimator=k3 for teacher-side KL: its pathwise gradient (1 − π_S/π_T) is
    at least directionally sensible (pushes π_T toward π_S). NOTE k3-on-teacher is
    itself only a heuristic — a faithful reverse-KL gradient needs the score-function
    term too — so kl_beta_teacher>0 is exploratory; the paper's Asymmetric-RL/SD uses
    kl_beta_teacher=0. See "Future work (V2)" in docs-agent/training/ice-implementation.md.
    """
    if kl_estimator not in ("k3", "mean_logprob"):
        raise ValueError(
            f"ice.kl_estimator={kl_estimator!r} must be 'k3' or 'mean_logprob'."
        )
    if kl_estimator == "mean_logprob" and beta_teacher > 0:
        raise ValueError(
            "ice.kl_estimator='mean_logprob' is incompatible with kl_beta_teacher>0: "
            "the mean-logprob estimator only yields a correct gradient for the student "
            "(forward) KL. For teacher-side KL use kl_estimator='k3'."
        )


def compute_distill_kl_mean_logprob(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mean-logprob KL_T and KL_S.

    NOTE: only kl_S_seq is a valid loss term (forward KL, grad → student =
    behaviour-cloning toward teacher). kl_T_seq is returned for diagnostics/symmetry
    only; its gradient is NOT a correct reverse-KL estimator (see
    validate_kl_estimator_config), so it must not be added to the loss.

    Args:
        student_log_prob: (B, T) student-context per-token logprobs (grad-tracking).
        teacher_log_prob: (B, T) teacher-context per-token logprobs (grad-tracking).
        response_mask: (B, T) 1 at response (action) token positions, shared by both.

    Returns:
        kl_T_seq: (B,) per-sample teacher-branch KL (grad through teacher).
        kl_S_seq: (B,) per-sample student-branch KL (grad through student).
    """
    mask = response_mask.float()
    mean_log_pi_S = masked_mean(student_log_prob, mask, axis=1)
    mean_log_pi_T = masked_mean(teacher_log_prob, mask, axis=1)
    kl_T_seq = mean_log_pi_T - mean_log_pi_S.detach()
    kl_S_seq = mean_log_pi_T.detach() - mean_log_pi_S
    return kl_T_seq, kl_S_seq


def masked_sample_mean(seq: torch.Tensor, keep_mask: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """(Weighted) mean of per-sample values over kept samples; 0 if none kept.

    Args:
        seq: (B,) per-sample scalars.
        keep_mask: (B,) 0/1 mask of samples to include (e.g. instructed & filtered).
        weights: optional (B,) non-negative per-sample weights (e.g. AWR). When given,
            a proper weighted mean over kept samples is returned — `sum(w·keep·seq) /
            sum(w·keep)` — which auto-normalizes, preserving the loss SCALE (a convex
            combination) while reallocating emphasis. None ⇒ uniform.

    Returns:
        scalar. All-zero kept set ⇒ 0 (no grad).
    """
    keep = keep_mask.to(seq.dtype)
    if weights is None:
        return (seq * keep).sum() / keep.sum().clamp_min(1.0)
    w = weights.to(seq.dtype) * keep
    return (seq * w).sum() / w.sum().clamp_min(1e-8)


def compute_awr_weights(episode_returns, instructed_mask, temp: float = 1.0, cap: float = None):
    """Positive, advantage-weighted (AWR/RWR) per-episode weights for KL_S.

    `A_i = (R_i - mean) / std` z-scored over the INSTRUCTED episodes (the teacher
    rollouts — the KL_S target set), then `w_i = exp(A_i / temp)`, optionally capped at
    `cap`. Always > 0, emphasising higher-return teacher rollouts; `temp→∞` → uniform,
    `temp→0+` → approaches only-the-best (the exponent is clamped to avoid `math.exp`
    overflow). Non-instructed entries get weight 0 (KL_S ignores them anyway). Applied as
    a *weighted mean* downstream, so absolute scale is preserved. Returns list[float].

    `temp` must be > 0 and `cap` must be None or > 0 (enforced): signed advantages enter
    only inside `exp` so weights stay strictly positive — we never multiply KL_S by a
    negative coefficient (that would be unbounded anti-distillation).
    """
    import math
    if temp <= 0:
        raise ValueError(f"ice.kl_weight_temp={temp} must be > 0.")
    if cap is not None and cap <= 0:
        raise ValueError(
            f"ice.kl_weight_cap={cap} must be > 0 (or null); a non-positive cap would "
            "break the strictly-positive-weight invariant (negative coeff = unbounded "
            "anti-distillation)."
        )
    _MAX_EXP = 50.0  # overflow guard: math.exp RAISES above ~709. exp(50)~5e21 is already
                     # the effective only-best limit, so clamp the exponent here.
    n = len(instructed_mask)
    returns = [float(episode_returns[i]) for i in range(n)]
    inst_idx = [i for i in range(n) if instructed_mask[i]]
    weights = [0.0] * n
    if not inst_idx:
        return weights
    vals = [returns[i] for i in inst_idx]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    for i in inst_idx:
        adv = (returns[i] - mean) / std if std > 1e-8 else 0.0
        w = math.exp(min(adv / temp, _MAX_EXP))
        if cap is not None:
            w = min(w, cap)
        weights[i] = w
    return weights
