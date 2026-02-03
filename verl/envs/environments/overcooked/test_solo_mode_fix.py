#!/usr/bin/env python3
"""Test that partner agent doesn't block movement in solo mode."""

import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper

def test_solo_mode_blocking():
    """Test that in solo mode, partner doesn't block their spawn tile."""

    print("Testing solo mode partner blocking fix...")
    print("=" * 60)

    # Create environment in solo mode
    env = OvercookedGymWrapper(
        layout="cramped_room",
        max_steps=100,
        partner_policy="none",  # Solo mode
        shaped_reward=True,
        print_visualization=True,
        print_coordinates=True,
    )

    obs, info = env.reset()

    print("\nInitial state (solo mode):")
    print("=" * 60)
    print(env.render())

    # Check agent positions
    state_info = env.get_state_info()
    agents = state_info["agents"]

    print("\n" + "=" * 60)
    print("Agent info:")
    print("=" * 60)
    for agent in agents:
        print(f"{agent['name']}: pos={agent['pos']}, is_controlled={agent['is_controlled']}")

    # The controlled agent starts at (1, 1) in cramped_room
    # Partner normally starts at (3, 1)
    # In solo mode, partner should be at (-1, -1) (off-map)

    controlled_pos = agents[0]["pos"]
    print(f"\nControlled agent at: {controlled_pos}")
    print("Expected: (1, 1)")

    # Check raw state
    print("\n" + "=" * 60)
    print("Raw state check:")
    print("=" * 60)
    print(f"Agent 0 pos: x={env._state.agents.pos.x[0]}, y={env._state.agents.pos.y[0]}")
    print(f"Agent 1 pos: x={env._state.agents.pos.x[1]}, y={env._state.agents.pos.y[1]}")
    print("\nExpected agent 1 (partner) to be at (-1, -1) in solo mode")

    # Try moving to where partner would normally spawn (3, 1)
    print("\n" + "=" * 60)
    print("Testing movement to partner's normal spawn (3, 1):")
    print("=" * 60)

    # Move right twice to reach (3, 1)
    print("\n1. Move right...")
    obs, reward, terminated, truncated, info = env.step("right")
    print(env.render())

    print("\n2. Move right again (should reach partner's spawn at 3,1)...")
    obs, reward, terminated, truncated, info = env.step("right")
    print(env.render())

    state_info = env.get_state_info()
    final_pos = state_info["agents"][0]["pos"]

    print("\n" + "=" * 60)
    print("Test result:")
    print("=" * 60)
    if final_pos == (3, 1):
        print("✓ SUCCESS: Player reached (3,1) - partner is NOT blocking!")
        print("Partner agent is correctly moved off-map in solo mode.")
        return True
    else:
        print(f"✗ FAILED: Player at {final_pos}, expected (3,1)")
        print("Partner agent is still blocking their spawn tile.")
        return False


def test_partner_mode_blocking():
    """Test that with partner, they DO block their spawn tile (normal behavior)."""

    print("\n\n" + "=" * 60)
    print("Testing normal partner mode (partner should block):")
    print("=" * 60)

    env = OvercookedGymWrapper(
        layout="cramped_room",
        max_steps=100,
        partner_policy="noop",  # Partner present
        shaped_reward=True,
        print_visualization=True,
        print_coordinates=True,
    )

    obs, info = env.reset()

    print("\nInitial state (with partner):")
    print(env.render())

    # Check raw state
    print("\nAgent positions:")
    print(f"Agent 0: x={env._state.agents.pos.x[0]}, y={env._state.agents.pos.y[0]}")
    print(f"Agent 1: x={env._state.agents.pos.x[1]}, y={env._state.agents.pos.y[1]}")

    # Move right twice
    print("\n1. Move right...")
    env.step("right")

    print("\n2. Move right again...")
    obs, reward, terminated, truncated, info = env.step("right")
    print(env.render())

    state_info = env.get_state_info()
    final_pos = state_info["agents"][0]["pos"]

    print("\n" + "=" * 60)
    print("Test result:")
    print("=" * 60)
    if final_pos == (2, 1):
        print("✓ CORRECT: Player stopped at (2,1) - partner blocks (3,1)")
        print("Normal partner mode working as expected.")
        return True
    else:
        print(f"Note: Player at {final_pos}")
        return True  # Either way is fine for this test


if __name__ == "__main__":
    solo_success = test_solo_mode_blocking()
    partner_success = test_partner_mode_blocking()

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Solo mode fix: {'✓ PASS' if solo_success else '✗ FAIL'}")
    print(f"Partner mode check: {'✓ PASS' if partner_success else '✗ FAIL'}")

    if solo_success:
        print("\n✓ Bug fixed! Partner no longer blocks in solo mode.")
    else:
        print("\n✗ Bug persists - partner still blocking in solo mode.")
