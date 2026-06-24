"""Unit tests for per-focus evaluation expansion (V2). CPU, no GPU/model."""

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
