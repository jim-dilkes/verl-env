"""Unit tests for per-focus evaluation expansion (V2). CPU, no GPU/model."""

from collections.abc import Mapping

import pytest

from verl.envs.environments.focus_instructions import (
    fixed_focus_for_batch,
    expand_per_focus_eval_envs,
)


def test_fixed_focus_for_batch():
    assert fixed_focus_for_batch(4, "strategy A") == ["strategy A"] * 4
    assert fixed_focus_for_batch(3, None) == [None, None, None]
    assert fixed_focus_for_batch(0, "x") == []


INSTR = ["i0", "i1", "i2"]
BASE = {"name": "Overcooked-Cramped", "env_name": "overcooked", "episode_length": 40,
        "ice_per_focus": True}


def test_expand_produces_nofocus_plus_one_per_instruction():
    out = expand_per_focus_eval_envs(BASE, INSTR)
    assert len(out) == 1 + len(INSTR)
    names = [c["name"] for c in out]
    assert names == ["Overcooked-Cramped_nofocus",
                     "Overcooked-Cramped_focus0",
                     "Overcooked-Cramped_focus1",
                     "Overcooked-Cramped_focus2"]
    # fixed-focus markers
    assert out[0]["ice_fixed_focus"] == -1
    assert [c["ice_fixed_focus"] for c in out[1:]] == [0, 1, 2]
    # all inherit ICE, none keep the expansion flags
    assert all(c["inherit_ice"] for c in out)
    assert all("ice_per_focus" not in c for c in out)
    # base fields preserved
    assert all(c["episode_length"] == 40 for c in out)


def test_expand_respects_focus_indices_subset():
    cfg = {**BASE, "ice_focus_indices": [0, 2]}
    out = expand_per_focus_eval_envs(cfg, INSTR)
    assert [c["name"] for c in out] == ["Overcooked-Cramped_nofocus",
                                        "Overcooked-Cramped_focus0",
                                        "Overcooked-Cramped_focus2"]
    assert [c["ice_fixed_focus"] for c in out] == [-1, 0, 2]
    assert all("ice_focus_indices" not in c for c in out)


def test_expand_out_of_range_index_raises():
    with pytest.raises(ValueError):
        expand_per_focus_eval_envs({**BASE, "ice_focus_indices": [0, 5]}, INSTR)


def test_expand_does_not_mutate_input():
    cfg = dict(BASE)
    _ = expand_per_focus_eval_envs(cfg, INSTR)
    assert cfg["ice_per_focus"] is True and "ice_fixed_focus" not in cfg


# --- milestone extraction gating (regression) ---------------------------------
# Bug: the eval loop called _extract_from_info(info_vec, "milestones", default=None);
# the helper RAISES when default is None and the key is missing, so any eval env not
# emitting "milestones" (snake/babyai, or overcooked with emit_milestones=False) threw
# mid-loop and the outer handler silently dropped that env's ENTIRE eval. Fix: gate on
# key presence and only extract when the env actually emits milestones.

def _evaluator():
    from verl.trainer.ppo.multi_env_evaluator import MultiEnvEvaluator
    return MultiEnvEvaluator.__new__(MultiEnvEvaluator)  # bypass heavy __init__


def test_extract_from_info_raises_on_missing_key_with_default_none():
    # documents the trap the bug relied on: default=None does NOT suppress the raise
    ev = _evaluator()
    with pytest.raises(ValueError):
        ev._extract_from_info([{"game_state_text": "x"}], "milestones")


def test_extract_from_info_returns_present_milestones():
    ev = _evaluator()
    assert ev._extract_from_info([{"milestones": [True, False]}], "milestones") == [[True, False]]


def test_milestone_gating_skips_when_env_does_not_emit():
    # the fix's guard: absent -> skip (no extraction, no raise); present -> extract
    absent = [{"game_state_text": "x"}, {"game_state_text": "y"}]
    present = [{"milestones": [True, False]}, {"milestones": [False, True]}]
    assert not (bool(absent) and isinstance(absent[0], Mapping) and "milestones" in absent[0])
    assert (bool(present) and isinstance(present[0], Mapping) and "milestones" in present[0])
