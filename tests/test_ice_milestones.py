"""Unit tests for Overcooked task-milestone derivation (V2 part B). CPU, no JAX."""

from verl.envs.environments.overcooked.milestones import MILESTONE_NAMES, compute_milestones


def _ing(counts):  # raw ingredient(s) in hand
    return {"plate": False, "cooked": False, "ingredient_counts": counts}


def _empty_plate():
    return {"plate": True, "cooked": False, "ingredient_counts": {}}


def _soup():
    return {"plate": True, "cooked": True, "ingredient_counts": {0: 3}}


def _pot(counts=None, timer=0, cooked=False):
    contents = None
    if counts or cooked:
        contents = {"plate": False, "cooked": cooked, "ingredient_counts": counts or {}}
    return {"contents": contents, "timer": timer, "pos": (0, 0)}


def test_names_length():
    assert MILESTONE_NAMES == ["hold_ingredient", "ingredient_in_pot", "pot_cooking",
                               "hold_dish", "hold_soup", "delivered"]


def test_idle_all_false():
    assert compute_milestones(None, [_pot()], 0.0) == [False] * 6


def test_hold_ingredient():
    ms = compute_milestones(_ing({0: 1}), [_pot()], 0.0)
    assert ms[0] is True and ms[1:] == [False] * 5


def test_ingredient_in_pot():
    ms = compute_milestones(None, [_pot({0: 1})], 0.0)
    assert ms[1] is True and ms[2] is False


def test_pot_cooking_via_timer_or_cooked():
    assert compute_milestones(None, [_pot({0: 3}, timer=5)], 0.0)[2] is True
    assert compute_milestones(None, [_pot(cooked=True)], 0.0)[2] is True


def test_hold_dish_vs_soup():
    dish = compute_milestones(_empty_plate(), [_pot()], 0.0)
    soup = compute_milestones(_soup(), [_pot()], 0.0)
    assert dish[3] is True and dish[4] is False        # empty plate = dish, not soup
    assert soup[4] is True and soup[3] is False         # cooked soup = soup, not dish


def test_delivered_on_sparse_reward():
    assert compute_milestones(None, [_pot()], 20.0)[5] is True
    assert compute_milestones(None, [_pot()], 0.0)[5] is False


def test_holding_ingredient_is_not_dish():
    # raw ingredient in hand must not trip the "hold_dish" (plate) milestone
    ms = compute_milestones(_ing({0: 1}), [_pot()], 0.0)
    assert ms[3] is False and ms[4] is False
