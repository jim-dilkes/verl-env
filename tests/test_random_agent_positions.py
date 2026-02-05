"""Tests for random_agent_positions in Overcooked via OvercookedGymWrapper."""
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2 import OvercookedV2
from jaxmarl.environments.overcooked_v2.common import StaticObject
from jaxmarl.environments.overcooked_v2.utils import compute_enclosed_spaces
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts

from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper


def get_agent_positions(env):
    """Extract (x, y) for each agent from env state."""
    pos_x = np.array(env._state.agents.pos.x)
    pos_y = np.array(env._state.agents.pos.y)
    return [(int(pos_x[i]), int(pos_y[i])) for i in range(len(pos_x))]


def get_empty_tiles(layout_name):
    """Get set of (x, y) coordinates that are EMPTY in the layout."""
    layout = overcooked_v2_layouts[layout_name]
    static = np.array(layout.static_objects)
    h, w = static.shape
    tiles = set()
    for y in range(h):
        for x in range(w):
            if static[y, x] == StaticObject.EMPTY:
                tiles.add((x, y))
    return tiles


def get_enclosed_space_map(layout_name):
    """Get enclosed_spaces grid and return {room_id: set of (x,y)} mapping."""
    layout = overcooked_v2_layouts[layout_name]
    empty_mask = layout.static_objects == StaticObject.EMPTY
    spaces = np.array(compute_enclosed_spaces(empty_mask))
    rooms = {}
    h, w = spaces.shape
    for y in range(h):
        for x in range(w):
            rid = int(spaces[y, x])
            if rid >= 0:
                rooms.setdefault(rid, set()).add((x, y))
    return spaces, rooms


# ─── Test 1: Positions actually vary across resets (2-player, single room) ───

def test_positions_vary_cramped_room():
    """cramped_room is a single open room. Both agents should get different
    positions across resets with different seeds."""
    positions_seen = set()
    n_resets = 20

    for seed in range(n_resets):
        env = OvercookedGymWrapper(
            layout="cramped_room", max_steps=50,
            partner_policy="noop", seed=seed,
            random_agent_positions=True,
        )
        env.reset()
        pos = get_agent_positions(env)
        positions_seen.add(tuple(pos))

    # With 6 empty tiles and 2 agents, there are 30 possible combos.
    # Over 20 seeds we should see more than 1 unique arrangement.
    assert len(positions_seen) > 1, (
        f"Expected varied positions across resets, got only {positions_seen}"
    )
    print(f"  PASS: saw {len(positions_seen)} distinct arrangements over {n_resets} resets")


# ─── Test 2: Positions always land on empty tiles ───

def test_positions_on_empty_tiles():
    """All spawned positions must be on EMPTY tiles."""
    empty = get_empty_tiles("cramped_room")

    for seed in range(30):
        env = OvercookedGymWrapper(
            layout="cramped_room", max_steps=50,
            partner_policy="noop", seed=seed,
            random_agent_positions=True,
        )
        env.reset()
        for i, pos in enumerate(get_agent_positions(env)):
            assert pos in empty, (
                f"Seed {seed}: agent {i} at {pos} is not an empty tile. "
                f"Empty tiles: {empty}"
            )
    print("  PASS: all positions on empty tiles (30 seeds)")


# ─── Test 3: Agents never overlap ───

def test_no_agent_overlap():
    """Two agents should never spawn on the same tile."""
    for seed in range(50):
        env = OvercookedGymWrapper(
            layout="cramped_room", max_steps=50,
            partner_policy="noop", seed=seed,
            random_agent_positions=True,
        )
        env.reset()
        positions = get_agent_positions(env)
        assert positions[0] != positions[1], (
            f"Seed {seed}: both agents at {positions[0]}"
        )
    print("  PASS: no overlaps across 50 seeds")


# ─── Test 4: Room confinement (two_rooms layout) ───

def test_room_confinement_two_rooms():
    """In two_rooms, agents start in separate rooms and should stay
    confined to their respective rooms when positions are randomized."""
    spaces_grid, rooms = get_enclosed_space_map("two_rooms")

    # Determine which room each agent starts in (from layout defaults)
    layout = overcooked_v2_layouts["two_rooms"]
    agent_positions = layout.agent_positions
    agent_rooms = []
    for ax, ay in agent_positions:
        rid = int(spaces_grid[ay, ax])
        agent_rooms.append(rid)

    print(f"  Layout 'two_rooms': {len(rooms)} rooms, agents start in rooms {agent_rooms}")
    print(f"  Room sizes: {[(rid, len(tiles)) for rid, tiles in rooms.items()]}")

    # These should be different rooms
    assert agent_rooms[0] != agent_rooms[1], (
        f"Expected agents in different rooms, both in room {agent_rooms[0]}"
    )

    for seed in range(50):
        env = OvercookedGymWrapper(
            layout="two_rooms", max_steps=50,
            partner_policy="noop", seed=seed,
            random_agent_positions=True,
        )
        env.reset()
        positions = get_agent_positions(env)

        for i, pos in enumerate(positions):
            expected_room = agent_rooms[i]
            actual_room = int(spaces_grid[pos[1], pos[0]])
            assert actual_room == expected_room, (
                f"Seed {seed}: agent {i} at {pos} is in room {actual_room}, "
                f"expected room {expected_room}"
            )

    print("  PASS: room confinement holds across 50 seeds")


# ─── Test 5: Solo mode (partner=none) + random positions ───

def test_solo_mode_random_positions():
    """Solo mode should randomize agent_0 position; agent_1 should be at (-1,-1)."""
    positions_seen = set()

    for seed in range(20):
        env = OvercookedGymWrapper(
            layout="cramped_room", max_steps=50,
            partner_policy="none", seed=seed,
            random_agent_positions=True,
        )
        env.reset()
        positions = get_agent_positions(env)

        # Partner should be off-map
        assert positions[1] == (-1, -1), (
            f"Seed {seed}: solo partner at {positions[1]}, expected (-1, -1)"
        )

        # Controlled agent should be on a valid tile
        empty = get_empty_tiles("cramped_room")
        assert positions[0] in empty, (
            f"Seed {seed}: agent_0 at {positions[0]} not in empty tiles"
        )
        positions_seen.add(positions[0])

    assert len(positions_seen) > 1, (
        f"Solo agent position never varied: {positions_seen}"
    )
    print(f"  PASS: solo mode works, saw {len(positions_seen)} distinct positions")


# ─── Test 6: Deterministic when random_agent_positions=False ───

def test_deterministic_without_flag():
    """Without random_agent_positions, same seed should give same positions."""
    env = OvercookedGymWrapper(
        layout="cramped_room", max_steps=50,
        partner_policy="noop", seed=42,
        random_agent_positions=False,
    )
    env.reset()
    pos1 = get_agent_positions(env)

    env2 = OvercookedGymWrapper(
        layout="cramped_room", max_steps=50,
        partner_policy="noop", seed=99,
        random_agent_positions=False,
    )
    env2.reset()
    pos2 = get_agent_positions(env2)

    # Without randomization, layout-defined positions should be identical
    assert pos1 == pos2, (
        f"Without random_agent_positions, positions should match layout: {pos1} vs {pos2}"
    )
    print(f"  PASS: positions deterministic without flag: {pos1}")


# ─── Test 7: Same seed = same random positions (reproducibility) ───

def test_reproducibility():
    """Same seed should produce identical random positions."""
    for seed in [0, 7, 42, 123]:
        env1 = OvercookedGymWrapper(
            layout="cramped_room", max_steps=50,
            partner_policy="noop", seed=seed,
            random_agent_positions=True,
        )
        env1.reset()
        pos1 = get_agent_positions(env1)

        env2 = OvercookedGymWrapper(
            layout="cramped_room", max_steps=50,
            partner_policy="noop", seed=seed,
            random_agent_positions=True,
        )
        env2.reset()
        pos2 = get_agent_positions(env2)

        assert pos1 == pos2, (
            f"Seed {seed}: positions not reproducible: {pos1} vs {pos2}"
        )
    print("  PASS: same seed = same positions (4 seeds tested)")


# ─── Run all ───

if __name__ == "__main__":
    tests = [
        ("Positions vary (cramped_room, 2P)", test_positions_vary_cramped_room),
        ("Positions on empty tiles", test_positions_on_empty_tiles),
        ("No agent overlap", test_no_agent_overlap),
        ("Room confinement (two_rooms)", test_room_confinement_two_rooms),
        ("Solo mode + random positions", test_solo_mode_random_positions),
        ("Deterministic without flag", test_deterministic_without_flag),
        ("Reproducibility (same seed)", test_reproducibility),
    ]

    print("=" * 60)
    print("Testing random_agent_positions")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
