#!/usr/bin/env python3
"""Quick test that custom layout works with interactive_play setup."""

import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
from verl.envs.environments.overcooked.custom_layouts import CUSTOM_LAYOUTS

def test_interactive_setup():
    """Test that custom layout loads correctly with interactive_play settings."""

    layout_name = "cramped_room_mixed"
    layout = CUSTOM_LAYOUTS[layout_name]

    print(f"Testing custom layout: {layout_name}")
    print(f"Recipe: {layout.possible_recipes[0]}\n")

    env = OvercookedGymWrapper(
        layout=layout,
        max_steps=200,
        partner_policy="noop",
        seed=0,
        shaped_reward=True,
        print_visualization=True,
        print_coordinates=True,
        pot_cook_time=None,
    )

    print("Environment initialized successfully!")
    print(f"Cook time: {env.pot_cook_time} ticks\n")

    obs, info = env.reset()

    print("=" * 60)
    print(env.render())
    print("=" * 60)

    # Take a few test actions
    print("\nTesting actions...")

    # Move left (toward tomato pile)
    print("\n1. Moving left...")
    obs, reward, terminated, truncated, info = env.step("left")
    print(env.render())

    # Interact (pick up tomato)
    print("\n2. Interacting (should pick up tomato)...")
    obs, reward, terminated, truncated, info = env.step("interact")
    print(env.render())
    if reward != 0:
        print(f"Reward: {reward:+.2f}")

    print("\n✓ Custom layout works with interactive_play setup!")
    print("\nTo play interactively:")
    print("  python -m verl.envs.environments.overcooked.interactive_play --layout cramped_room_mixed")


if __name__ == "__main__":
    test_interactive_setup()
