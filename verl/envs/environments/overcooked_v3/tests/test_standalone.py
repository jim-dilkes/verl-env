"""Standalone tests for pure Python Overcooked implementation.

These tests verify game mechanics independently without requiring JaxMARL.
"""

import pytest
import numpy as np

from verl.envs.environments.overcooked_v3 import (
    ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION, DIRECTION_NAMES,
    StaticObject, DynamicObject, Direction, DIR_TO_VEC,
    DEFAULT_POT_COOK_TIME, DELIVERY_REWARD
)
from verl.envs.environments.overcooked_v3.layouts import (
    Layout, get_layout, BUILTIN_LAYOUTS, CUSTOM_LAYOUTS
)
from verl.envs.environments.overcooked_v3.game_engine import (
    OvercookedEngine, GameState, Agent, Position
)
from verl.envs.environments.overcooked_v3.gym_wrapper import OvercookedGymWrapper


class TestConstants:
    """Test that constants are correctly defined."""

    def test_actions_defined(self):
        assert len(ACTIONS) == 6
        assert "right" in ACTIONS
        assert "down" in ACTIONS
        assert "left" in ACTIONS
        assert "up" in ACTIONS
        assert "stay" in ACTIONS
        assert "interact" in ACTIONS

    def test_action_indices(self):
        assert ACTION_TO_IDX["right"] == 0
        assert ACTION_TO_IDX["down"] == 1
        assert ACTION_TO_IDX["left"] == 2
        assert ACTION_TO_IDX["up"] == 3
        assert ACTION_TO_IDX["stay"] == 4
        assert ACTION_TO_IDX["interact"] == 5

    def test_idx_to_action_inverse(self):
        for action, idx in ACTION_TO_IDX.items():
            assert IDX_TO_ACTION[idx] == action

    def test_direction_names(self):
        assert DIRECTION_NAMES[0] == "UP"
        assert DIRECTION_NAMES[1] == "DOWN"
        assert DIRECTION_NAMES[2] == "RIGHT"
        assert DIRECTION_NAMES[3] == "LEFT"

    def test_direction_vectors(self):
        assert DIR_TO_VEC[Direction.UP] == (0, -1)
        assert DIR_TO_VEC[Direction.DOWN] == (0, 1)
        assert DIR_TO_VEC[Direction.RIGHT] == (1, 0)
        assert DIR_TO_VEC[Direction.LEFT] == (-1, 0)


class TestDynamicObjectEncoding:
    """Test dynamic object encoding/decoding."""

    def test_empty(self):
        assert DynamicObject.EMPTY == 0

    def test_plate(self):
        assert DynamicObject.PLATE == 1

    def test_cooked(self):
        assert DynamicObject.COOKED == 2

    def test_ingredient_encoding(self):
        # Onion (index 0)
        onion = DynamicObject.ingredient(0)
        assert onion == 4  # 1 << 2

        # Tomato (index 1)
        tomato = DynamicObject.ingredient(1)
        assert tomato == 16  # 1 << 4

        # Lettuce (index 2)
        lettuce = DynamicObject.ingredient(2)
        assert lettuce == 64  # 1 << 6

    def test_is_ingredient(self):
        onion = DynamicObject.ingredient(0)
        assert DynamicObject.is_ingredient(onion) is True

        plate = DynamicObject.PLATE
        assert DynamicObject.is_ingredient(plate) is False

        empty = DynamicObject.EMPTY
        assert DynamicObject.is_ingredient(empty) is False

        # Plate with ingredient - not a pure ingredient
        plate_with_onion = DynamicObject.PLATE | onion
        assert DynamicObject.is_ingredient(plate_with_onion) is False

    def test_ingredient_count(self):
        # 3 onions
        three_onions = 3 * DynamicObject.ingredient(0)
        assert DynamicObject.ingredient_count(three_onions) == 3

        # 2 onions + 1 tomato
        mixed = 2 * DynamicObject.ingredient(0) + DynamicObject.ingredient(1)
        assert DynamicObject.ingredient_count(mixed) == 3

        # Empty
        assert DynamicObject.ingredient_count(0) == 0

    def test_recipe_encoding(self):
        # [0, 0, 0] = 3 onions
        recipe_3_onions = DynamicObject.get_recipe_encoding([0, 0, 0])
        assert DynamicObject.ingredient_count(recipe_3_onions) == 3
        # Should be 4 + 4 + 4 = 12
        assert recipe_3_onions == 12

        # [0, 0, 1] = 2 onions + 1 tomato
        recipe_mixed = DynamicObject.get_recipe_encoding([0, 0, 1])
        # 4 + 4 + 16 = 24
        assert recipe_mixed == 24


class TestLayoutParsing:
    """Test layout parsing."""

    def test_cramped_room_dimensions(self):
        layout = get_layout("cramped_room")
        assert layout.width == 5
        assert layout.height == 4

    def test_cramped_room_agents(self):
        layout = get_layout("cramped_room")
        # JaxMARL swaps agents for cramped_room
        assert len(layout.agent_positions) == 2
        # Both agents should be on row 1 (y=1)
        for x, y in layout.agent_positions:
            assert y == 1

    def test_cramped_room_recipe(self):
        layout = get_layout("cramped_room")
        assert layout.possible_recipes == [[0, 0, 0]]

    def test_custom_layout_mixed(self):
        layout = get_layout("cramped_room_mixed")
        assert layout.possible_recipes == [[0, 0, 1]]
        assert layout.num_ingredients == 2

    def test_layout_static_objects(self):
        layout = get_layout("cramped_room")

        # Top row should have wall and pot
        assert layout.static_objects[0, 0] == StaticObject.WALL
        assert layout.static_objects[0, 2] == StaticObject.POT
        assert layout.static_objects[0, 4] == StaticObject.WALL

        # Bottom row
        assert layout.static_objects[3, 1] == StaticObject.PLATE_PILE
        assert layout.static_objects[3, 3] == StaticObject.GOAL


class TestPosition:
    """Test Position class."""

    def test_move(self):
        pos = Position(2, 2)

        assert pos.move(Direction.UP) == Position(2, 1)
        assert pos.move(Direction.DOWN) == Position(2, 3)
        assert pos.move(Direction.LEFT) == Position(1, 2)
        assert pos.move(Direction.RIGHT) == Position(3, 2)

    def test_move_in_bounds(self):
        # Corner position
        pos = Position(0, 0)

        # Up and left should clip to 0
        assert pos.move_in_bounds(Direction.UP, 5, 4) == Position(0, 0)
        assert pos.move_in_bounds(Direction.LEFT, 5, 4) == Position(0, 0)

        # Down and right should work
        assert pos.move_in_bounds(Direction.DOWN, 5, 4) == Position(0, 1)
        assert pos.move_in_bounds(Direction.RIGHT, 5, 4) == Position(1, 0)

    def test_equality(self):
        assert Position(1, 2) == Position(1, 2)
        assert Position(1, 2) != Position(2, 1)


class TestGameEngine:
    """Test game engine logic."""

    def test_reset(self):
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=200)

        state = engine.reset()

        assert state.time == 0
        assert state.terminal is False
        assert len(state.agents) == 2
        assert state.grid.shape == (4, 5, 3)

    def test_movement_basic(self):
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=200)
        state = engine.reset()

        # Find agent on floor (not blocked)
        agent = state.agents[0]
        initial_pos = Position(agent.pos.x, agent.pos.y)

        # Move down (should work on cramped_room)
        state, _, _ = engine.step(state, [1, 4])  # agent 0: down, agent 1: stay

        # Agent should have moved or be blocked by wall
        # In cramped_room, row 1 agents can move down to row 2

    def test_movement_blocked_by_wall(self):
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=200)
        state = engine.reset()

        # Get initial position
        agent = state.agents[0]
        initial_pos = Position(agent.pos.x, agent.pos.y)

        # Try to move up (into wall/pot row)
        state, _, _ = engine.step(state, [3, 4])  # agent 0: up, agent 1: stay

        # Agent should stay in place (blocked)
        assert state.agents[0].pos == initial_pos

    def test_cooking_timer(self):
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=200, pot_cook_time=5)
        state = engine.reset()

        # Manually place 3 ingredients in pot
        pot_y, pot_x = 0, 2  # Pot position in cramped_room
        three_onions = 3 * DynamicObject.ingredient(0)
        state.grid[pot_y, pot_x, 1] = three_onions
        state.grid[pot_y, pot_x, 2] = 5  # Start timer

        # Step 5 times
        for i in range(5):
            state, _, _ = engine.step(state, [4, 4])  # Both stay

        # Timer should be 0 and soup should be cooked
        assert state.grid[pot_y, pot_x, 2] == 0
        assert state.grid[pot_y, pot_x, 1] & DynamicObject.COOKED

    def test_delivery_reward(self):
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=200)
        state = engine.reset()

        # Give agent a completed soup
        recipe = DynamicObject.get_recipe_encoding([0, 0, 0])
        plated_soup = recipe | DynamicObject.PLATE | DynamicObject.COOKED

        # Find serving counter position
        serving_x, serving_y = 3, 3  # In cramped_room

        # Position agent next to serving counter facing it
        state.agents[0].pos = Position(serving_x, serving_y - 1)  # Above serving counter
        state.agents[0].direction = Direction.DOWN
        state.agents[0].inventory = plated_soup

        # Interact to deliver
        state, reward, _ = engine.step(state, [5, 4])  # agent 0: interact

        assert reward == DELIVERY_REWARD
        assert state.agents[0].inventory == 0

    def test_max_steps_termination(self):
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=5)
        state = engine.reset()

        # Step 5 times
        for i in range(5):
            state, _, _ = engine.step(state, [4, 4])

        assert state.terminal is True


class TestGymWrapper:
    """Test gymnasium wrapper."""

    def test_reset(self):
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert "state" in info

    def test_step(self):
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        obs, info = env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(4)  # stay

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_render(self):
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        env.reset(seed=42)

        render_output = env.render()

        assert isinstance(render_output, str)
        assert "Kitchen" in render_output or "Layout" in render_output

    def test_action_string(self):
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        env.reset(seed=42)

        # Step with string action
        obs, reward, _, _, _ = env.step("stay")
        assert obs is not None

    def test_partner_noop(self):
        env = OvercookedGymWrapper(
            layout="cramped_room",
            max_steps=200,
            partner_policy="noop"
        )
        env.reset(seed=42)

        # Partner should always stay
        for _ in range(10):
            action = env._get_partner_action()
            assert action == 4  # stay

    def test_solo_mode(self):
        env = OvercookedGymWrapper(
            layout="cramped_room",
            max_steps=200,
            partner_policy="none"
        )
        obs, info = env.reset(seed=42)

        assert env.solo_mode is True

        # Partner should be at (-1, -1)
        partner_idx = 1  # Controlled is agent_0
        assert env._state.agents[partner_idx].pos.x == -1
        assert env._state.agents[partner_idx].pos.y == -1

    def test_pot_cook_time_override(self):
        env = OvercookedGymWrapper(
            layout="cramped_room",
            max_steps=200,
            pot_cook_time=10
        )
        assert env.pot_cook_time == 10

    def test_get_state_info(self):
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        env.reset(seed=42)

        info = env.get_state_info()

        assert "agents" in info
        assert "grid" in info
        assert "time" in info
        assert "recipe" in info

    def test_inventory_to_str(self):
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)

        assert env._inventory_to_str(0) == "NOTHING"
        assert env._inventory_to_str(DynamicObject.PLATE) == "empty plate"

        onion = DynamicObject.ingredient(0)
        assert "ONION" in env._inventory_to_str(onion).upper()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_interact_empty_space(self):
        """Interact with empty floor should do nothing."""
        env = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        env.reset(seed=42)

        initial_state = env.get_state_info()
        initial_inv = initial_state["agents"][0]["inventory"]

        # Interact (might be facing empty floor)
        env.step(5)

        state = env.get_state_info()
        # Inventory might or might not change depending on what's faced
        # This just tests it doesn't crash

    def test_double_pickup(self):
        """Can't pick up when already holding something."""
        layout = get_layout("cramped_room")
        engine = OvercookedEngine(layout, max_steps=200)
        state = engine.reset()

        # Give agent an item
        onion = DynamicObject.ingredient(0)
        state.agents[0].inventory = onion

        initial_inv = state.agents[0].inventory

        # Try to interact (pick up) - should fail
        # Position agent facing ingredient pile
        state.agents[0].pos = Position(2, 1)  # Center of cramped room
        state.agents[0].direction = Direction.LEFT  # Face left toward onion pile

        state, _, _ = engine.step(state, [5, 4])

        # Inventory should be unchanged
        assert state.agents[0].inventory == initial_inv

    def test_invalid_layout_name(self):
        with pytest.raises(ValueError):
            get_layout("nonexistent_layout")


class TestSeedDeterminism:
    """Test that same seed produces same results."""

    def test_same_seed_same_result(self):
        env1 = OvercookedGymWrapper(layout="cramped_room", max_steps=200)
        env2 = OvercookedGymWrapper(layout="cramped_room", max_steps=200)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Initial states should match
        assert np.allclose(obs1, obs2)

        # Same actions should produce same results
        actions = [0, 1, 5, 3, 4, 5, 2]
        for action in actions:
            obs1, r1, _, _, _ = env1.step(action)
            obs2, r2, _, _, _ = env2.step(action)

            assert np.allclose(obs1, obs2)
            assert r1 == r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
