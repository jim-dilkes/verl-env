"""Paper-faithful sampled-action teacher/student KL for DIME (ICE π-distill).

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
            raise ValueError(f"dime.kl_filter_top_pct={top_pct} must be in (0, 1].")
        n_keep = math.ceil(top_pct * len(inst_idx)) if inst_idx else 0
        for i in sorted(inst_idx, key=lambda j: returns[j], reverse=True)[:n_keep]:
            keep[i] = True
    else:
        raise ValueError(
            f"dime.kl_filter={kl_filter!r} must be 'none', 'return_positive', or 'top_pct'."
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
    kl_beta_teacher=0. See "Future work (V2)" in docs-agent/training/dime-implementation.md.
    """
    if kl_estimator not in ("k3", "mean_logprob"):
        raise ValueError(
            f"dime.kl_estimator={kl_estimator!r} must be 'k3' or 'mean_logprob'."
        )
    if kl_estimator == "mean_logprob" and beta_teacher > 0:
        raise ValueError(
            "dime.kl_estimator='mean_logprob' is incompatible with kl_beta_teacher>0: "
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


def masked_sample_mean(seq: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Mean of per-sample values over kept samples; 0 if none kept.

    Args:
        seq: (B,) per-sample scalars.
        keep_mask: (B,) 0/1 mask of samples to include (e.g. instructed & filtered).

    Returns:
        scalar = sum(seq * keep) / max(sum(keep), 1). All-zero mask → 0 (no grad).
    """
    keep = keep_mask.to(seq.dtype)
    return (seq * keep).sum() / keep.sum().clamp_min(1.0)
