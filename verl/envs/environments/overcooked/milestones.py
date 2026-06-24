"""Ordered, recipe-agnostic Overcooked task milestones.

Pure (no JAX) so it can be unit-tested and imported by the evaluator for metric
naming without pulling the JAX env. The env wrapper decodes its state into the
`held` / `pots` structures (via its existing `_decode_item` / `_get_pot_info`
helpers) and calls `compute_milestones`; the evaluator OR-accumulates the per-step
flags per trajectory and reports the furthest-reached fraction per milestone.
"""

from typing import List, Optional

MILESTONE_NAMES = [
    "hold_ingredient",    # 0: controlled agent holding a raw ingredient
    "ingredient_in_pot",  # 1: >=1 ingredient placed in a pot
    "pot_cooking",        # 2: a pot is cooking or cooked
    "hold_dish",          # 3: holding a dish/plate (not yet a completed soup)
    "hold_soup",          # 4: holding a cooked soup on a plate
    "delivered",          # 5: a soup delivered this step (sparse reward > 0)
]


def compute_milestones(held: Optional[dict], pots: List[dict], sparse_reward: float) -> List[bool]:
    """Per-step milestone flags in MILESTONE_NAMES order.

    Args:
        held: decoded controlled-agent inventory (``_decode_item`` output) or None.
            Shape: {"plate": bool, "cooked": bool, "ingredient_counts": {idx: count}}.
        pots: list of pot dicts (``_get_pot_info`` output), each
            {"contents": decoded-or-None, "timer": int, ...}.
        sparse_reward: pre-shaping reward this step (>0 == a delivery).
    """
    def _pot_total(p):
        return sum(p["contents"]["ingredient_counts"].values()) if p["contents"] else 0

    held_raw = held is not None and bool(held["ingredient_counts"]) and not held["plate"]
    held_dish = held is not None and held["plate"] and not held["cooked"]
    held_soup = held is not None and held["plate"] and held["cooked"]
    pot_has_ingredient = any(_pot_total(p) >= 1 for p in pots)
    pot_cooking = any(
        p["contents"] is not None and (p["contents"]["cooked"] or p["timer"] > 0)
        for p in pots
    )

    return [
        bool(held_raw),
        bool(pot_has_ingredient),
        bool(pot_cooking),
        bool(held_dish),
        bool(held_soup),
        bool(sparse_reward > 0),
    ]
