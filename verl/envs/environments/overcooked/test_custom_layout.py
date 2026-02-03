#!/usr/bin/env python3
"""Test script for custom Overcooked layouts."""

import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
from verl.envs.environments.overcooked.custom_layouts import get_cramped_room_mixed

def test_custom_layout():
    """Test the cramped_room_mixed custom layout."""
    print("Testing cramped_room_mixed layout...")
    print("Expected: Left pile=tomato (1), Right pile=onion (0), Recipe=[0,0,1]\n")

    layout = get_cramped_room_mixed()

    # Print layout info
    print(f"Layout size: {layout.width}x{layout.height}")
    print(f"Num ingredients: {layout.num_ingredients}")
    print(f"Possible recipes: {layout.possible_recipes}")
    print(f"Agent positions: {layout.agent_positions}\n")

    # Create environment
    env = OvercookedGymWrapper(
        layout=layout,
        max_steps=100,
        partner_policy="noop",
        shaped_reward=True,
        print_visualization=True,
        print_coordinates=True,
    )

    print("=" * 60)
    print("Initial state:")
    print("=" * 60)

    obs, info = env.reset()
    print(env.render())

    print("\n" + "=" * 60)
    print("Recipe verification:")
    print("=" * 60)
    print(f"Current recipe (encoded): {env._state.recipe}")

    # Decode recipe from state
    from jaxmarl.environments.overcooked_v2.common import DynamicObject
    import jax.numpy as jnp
    recipe_encoded = env._state.recipe
    # Convert to jax array if needed
    if not isinstance(recipe_encoded, jnp.ndarray):
        recipe_encoded = jnp.array(recipe_encoded)
    ingredient_list = DynamicObject.get_ingredient_idx_list(recipe_encoded)
    print(f"Recipe ingredients: {ingredient_list}")
    print(f"Expected: [0, 0, 1] (2 onions + 1 tomato)")

    if ingredient_list == [0, 0, 1]:
        print("✓ Recipe matches expected!")
    else:
        print("✗ Recipe does not match!")

    print("\n" + "=" * 60)
    print("Instructions:")
    print("=" * 60)
    print("To use in config:")
    print("  envs.overcooked_kwargs.layout_name=cramped_room_mixed")
    print("\nGameplay:")
    print("  1. Pick up 2 onions from RIGHT pile (0)")
    print("  2. Drop into pot")
    print("  3. Pick up 1 tomato from LEFT pile (1)")
    print("  4. Drop into pot")
    print("  5. Wait for cooking, get plate, serve")


if __name__ == "__main__":
    test_custom_layout()
