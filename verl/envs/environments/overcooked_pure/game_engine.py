"""Pure Python/NumPy implementation of Overcooked game logic.

This module implements the core Overcooked mechanics without any JAX dependencies,
maintaining behavioral parity with JaxMARL's OvercookedV2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import copy

from . import (
    StaticObject, DynamicObject, Direction, DIR_TO_VEC,
    ACTION_TO_DIRECTION, DEFAULT_POT_COOK_TIME, DELIVERY_REWARD, SHAPED_REWARDS
)
from .layouts import Layout


@dataclass
class Position:
    """2D position in the grid."""
    x: int
    y: int

    def move(self, direction: int) -> 'Position':
        """Return new position after moving in direction."""
        dx, dy = DIR_TO_VEC[direction]
        return Position(self.x + dx, self.y + dy)

    def move_in_bounds(self, direction: int, width: int, height: int) -> 'Position':
        """Move in direction, clipping to grid bounds."""
        new_pos = self.move(direction)
        new_x = max(0, min(width - 1, new_pos.x))
        new_y = max(0, min(height - 1, new_pos.y))
        return Position(new_x, new_y)

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Agent:
    """Agent state."""
    pos: Position
    direction: int  # 0=UP, 1=DOWN, 2=RIGHT, 3=LEFT
    inventory: int  # Bit-packed item encoding

    def get_fwd_pos(self) -> Position:
        """Get position agent is facing."""
        return self.pos.move(self.direction)

    def copy(self) -> 'Agent':
        return Agent(Position(self.pos.x, self.pos.y), self.direction, self.inventory)


@dataclass
class GameState:
    """Complete game state."""
    agents: List[Agent]
    grid: np.ndarray  # [height, width, 3]
    time: int
    recipe: int  # Bit-packed recipe encoding
    terminal: bool
    max_steps: int

    def copy(self) -> 'GameState':
        """Deep copy the state."""
        return GameState(
            agents=[a.copy() for a in self.agents],
            grid=self.grid.copy(),
            time=self.time,
            recipe=self.recipe,
            terminal=self.terminal,
            max_steps=self.max_steps,
        )


class OvercookedEngine:
    """Pure Python Overcooked game engine.

    Implements all game mechanics matching JaxMARL's OvercookedV2:
    - Movement with collision resolution
    - Interact action (pickup, place, cook, deliver)
    - Cooking timers
    - Rewards (base and shaped)
    """

    def __init__(
        self,
        layout: Layout,
        max_steps: int = 400,
        pot_cook_time: int = DEFAULT_POT_COOK_TIME,
        random_agent_positions: bool = False,
        start_cooking_interaction: bool = False,
        negative_rewards: bool = False,
    ):
        """Initialize the engine.

        Args:
            layout: Layout object defining the kitchen
            max_steps: Maximum steps per episode
            pot_cook_time: Ticks to cook a pot
            random_agent_positions: Randomize agent spawn positions
            start_cooking_interaction: Require interact to start cooking (default: auto-cook)
            negative_rewards: Enable negative rewards for wrong deliveries
        """
        self.layout = layout
        self.max_steps = max_steps
        self.pot_cook_time = pot_cook_time
        self.random_agent_positions = random_agent_positions
        self.start_cooking_interaction = start_cooking_interaction
        self.negative_rewards = negative_rewards

        self.height = layout.height
        self.width = layout.width
        self.num_agents = len(layout.agent_positions)

    def reset(self, rng: Optional[np.random.Generator] = None) -> GameState:
        """Reset to initial state.

        Args:
            rng: NumPy random generator for stochastic elements

        Returns:
            Initial GameState
        """
        if rng is None:
            rng = np.random.default_rng()

        # Initialize grid
        grid = np.zeros((self.height, self.width, 3), dtype=np.int32)
        grid[:, :, 0] = self.layout.static_objects

        # Sample recipe
        recipe_idx = rng.integers(0, len(self.layout.possible_recipes))
        recipe_list = self.layout.possible_recipes[recipe_idx]
        recipe = DynamicObject.get_recipe_encoding(recipe_list)

        # Initialize agents
        if self.random_agent_positions:
            agent_positions = self._sample_agent_positions(rng)
        else:
            agent_positions = [Position(x, y) for x, y in self.layout.agent_positions]

        agents = []
        for pos in agent_positions:
            agents.append(Agent(
                pos=pos,
                direction=Direction.UP,
                inventory=0,
            ))

        return GameState(
            agents=agents,
            grid=grid,
            time=0,
            recipe=recipe,
            terminal=False,
            max_steps=self.max_steps,
        )

    def _sample_agent_positions(self, rng: np.random.Generator) -> List[Position]:
        """Sample random agent positions on empty floor tiles."""
        # Find all empty floor positions
        empty_mask = self.layout.static_objects == StaticObject.EMPTY
        empty_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if empty_mask[y, x]:
                    empty_positions.append((x, y))

        if len(empty_positions) < self.num_agents:
            raise ValueError("Not enough empty positions for agents")

        # Sample without replacement
        indices = rng.choice(len(empty_positions), size=self.num_agents, replace=False)
        positions = [Position(*empty_positions[i]) for i in indices]

        return positions

    def step(
        self,
        state: GameState,
        actions: List[int],
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[GameState, float, List[float]]:
        """Execute one step.

        Args:
            state: Current game state
            actions: List of action indices for each agent

        Returns:
            Tuple of (new_state, total_reward, shaped_rewards_per_agent)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Copy state for modification
        new_state = state.copy()

        # Process movement for all agents
        new_state = self._process_movement(new_state, actions)

        # Process interact actions
        total_reward = 0.0
        shaped_rewards = [0.0] * self.num_agents

        for i, action in enumerate(actions):
            if action == 5:  # interact
                reward, shaped = self._process_interact(new_state, i)
                total_reward += reward
                shaped_rewards[i] = shaped

        # Update cooking timers
        new_state = self._update_timers(new_state)

        # Increment time
        new_state.time += 1

        # Check terminal
        if new_state.time >= new_state.max_steps:
            new_state.terminal = True

        return new_state, total_reward, shaped_rewards

    def _process_movement(self, state: GameState, actions: List[int]) -> GameState:
        """Process movement actions with collision resolution."""
        # Calculate intended new positions
        intended_positions = []
        new_directions = []

        for i, agent in enumerate(state.agents):
            action = actions[i]
            direction = ACTION_TO_DIRECTION.get(action, -1)

            if direction != -1:
                # Movement action - update direction and try to move
                new_dir = direction
                new_pos = agent.pos.move_in_bounds(direction, self.width, self.height)

                # Check if blocked by static object
                static = state.grid[new_pos.y, new_pos.x, 0]
                if static != StaticObject.EMPTY:
                    new_pos = agent.pos  # Stay in place
            else:
                # Stay or interact - keep position and direction
                new_dir = agent.direction
                new_pos = agent.pos

            intended_positions.append(new_pos)
            new_directions.append(new_dir)

        # Resolve collisions (iteratively until stable)
        final_positions = list(intended_positions)
        changed = True
        while changed:
            changed = False
            # Check for collisions
            position_counts = {}
            for i, pos in enumerate(final_positions):
                key = (pos.x, pos.y)
                if key not in position_counts:
                    position_counts[key] = []
                position_counts[key].append(i)

            # If collision, revert to original positions
            for pos_key, agent_indices in position_counts.items():
                if len(agent_indices) > 1:
                    for idx in agent_indices:
                        if final_positions[idx] != state.agents[idx].pos:
                            final_positions[idx] = state.agents[idx].pos
                            changed = True

        # Prevent swapping: if A→B and B→A, both stay
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                # Check if agents swapped
                if (final_positions[i] == state.agents[j].pos and
                    final_positions[j] == state.agents[i].pos):
                    final_positions[i] = state.agents[i].pos
                    final_positions[j] = state.agents[j].pos

        # Update agent states
        for i, agent in enumerate(state.agents):
            agent.pos = final_positions[i]
            agent.direction = new_directions[i]

        return state

    def _process_interact(self, state: GameState, agent_idx: int) -> Tuple[float, float]:
        """Process interact action for an agent.

        Returns:
            Tuple of (base_reward, shaped_reward)
        """
        agent = state.agents[agent_idx]
        fwd_pos = agent.get_fwd_pos()

        # Check bounds
        if not (0 <= fwd_pos.x < self.width and 0 <= fwd_pos.y < self.height):
            return 0.0, 0.0

        grid = state.grid
        cell_static = grid[fwd_pos.y, fwd_pos.x, 0]
        cell_dynamic = grid[fwd_pos.y, fwd_pos.x, 1]
        cell_timer = grid[fwd_pos.y, fwd_pos.x, 2]

        inventory = agent.inventory
        plated_recipe = state.recipe | DynamicObject.PLATE | DynamicObject.COOKED

        reward = 0.0
        shaped_reward = 0.0

        # Determine what's being interacted with
        is_plate_pile = cell_static == StaticObject.PLATE_PILE
        is_ingredient_pile = StaticObject.is_ingredient_pile(cell_static)
        is_pile = is_plate_pile or is_ingredient_pile
        is_pot = cell_static == StaticObject.POT
        is_goal = cell_static == StaticObject.GOAL
        is_wall = cell_static == StaticObject.WALL  # counters are walls

        cell_empty = cell_dynamic == 0
        inventory_empty = inventory == 0
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        # Pot states
        pot_is_cooking = is_pot and cell_timer > 0
        pot_is_cooked = is_pot and (cell_dynamic & DynamicObject.COOKED) != 0
        pot_is_idle = is_pot and not pot_is_cooking and not pot_is_cooked
        pot_full = DynamicObject.ingredient_count(cell_dynamic) == 3

        # === Pickup from pile ===
        if is_pile and inventory_empty:
            if is_plate_pile:
                agent.inventory = DynamicObject.PLATE
            else:
                pile_ingredient = StaticObject.get_ingredient(cell_static)
                agent.inventory = pile_ingredient
            return 0.0, 0.0

        # === Pick up cooked soup with plate ===
        if pot_is_cooked and inventory_is_plate:
            merged = cell_dynamic + inventory
            agent.inventory = merged
            grid[fwd_pos.y, fwd_pos.x, 1] = 0
            grid[fwd_pos.y, fwd_pos.x, 2] = 0

            # Shaped reward for useful soup pickup
            if merged == plated_recipe:
                shaped_reward += SHAPED_REWARDS["DISH_PICKUP"]

            return 0.0, shaped_reward

        # === Add ingredient to pot ===
        if pot_is_idle and inventory_is_ingredient and not pot_full:
            merged = cell_dynamic + inventory
            grid[fwd_pos.y, fwd_pos.x, 1] = merged
            agent.inventory = 0

            # Check if this placement is useful (matches recipe direction)
            ingredient_selector = inventory | (inventory << 1)
            is_useful = (cell_dynamic & ingredient_selector) < (state.recipe & ingredient_selector)
            if is_useful:
                shaped_reward += SHAPED_REWARDS["PLACEMENT_IN_POT"]

            # Auto-start cooking if pot becomes full
            new_count = DynamicObject.ingredient_count(merged)
            if new_count == 3 and not self.start_cooking_interaction:
                grid[fwd_pos.y, fwd_pos.x, 2] = self.pot_cook_time

            return 0.0, shaped_reward

        # === Start cooking manually (if enabled) ===
        if (self.start_cooking_interaction and pot_is_idle and
            not cell_empty and inventory_empty):
            pot_count = DynamicObject.ingredient_count(cell_dynamic)
            if pot_count == 3:
                grid[fwd_pos.y, fwd_pos.x, 2] = self.pot_cook_time
                # Shaped reward if matches recipe
                if cell_dynamic == state.recipe:
                    shaped_reward += SHAPED_REWARDS["POT_START_COOKING"]
            return 0.0, shaped_reward

        # === Deliver to serving counter ===
        if is_goal and inventory_is_dish:
            is_correct = inventory == plated_recipe
            if is_correct:
                reward = DELIVERY_REWARD
            elif self.negative_rewards:
                reward = -DELIVERY_REWARD

            agent.inventory = 0
            return reward, 0.0

        # === Place item on counter ===
        if is_wall and cell_empty and not inventory_empty:
            grid[fwd_pos.y, fwd_pos.x, 1] = inventory
            agent.inventory = 0
            return 0.0, 0.0

        # === Pick up from counter ===
        if is_wall and not cell_empty and inventory_empty:
            agent.inventory = cell_dynamic
            grid[fwd_pos.y, fwd_pos.x, 1] = 0

            # Shaped reward for plate pickup if there's a pot with stuff
            if cell_dynamic == DynamicObject.PLATE:
                # Count plates in all inventories
                num_plates = sum(1 for a in state.agents if a.inventory == DynamicObject.PLATE)
                # Count non-empty pots
                num_nonempty_pots = 0
                for y in range(self.height):
                    for x in range(self.width):
                        if grid[y, x, 0] == StaticObject.POT and grid[y, x, 1] != 0:
                            num_nonempty_pots += 1

                # Check no plates on counters (prevent reward hacking)
                plates_on_counters = 0
                for y in range(self.height):
                    for x in range(self.width):
                        if grid[y, x, 1] == DynamicObject.PLATE:
                            plates_on_counters += 1

                if plates_on_counters == 0 and num_plates < num_nonempty_pots:
                    shaped_reward += SHAPED_REWARDS["PLATE_PICKUP"]

            return 0.0, shaped_reward

        # === Pick up plate from pile ===
        # (Already handled above in is_pile case)

        return 0.0, 0.0

    def _update_timers(self, state: GameState) -> GameState:
        """Update cooking timers for all pots."""
        grid = state.grid

        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x, 0] == StaticObject.POT:
                    timer = grid[y, x, 2]
                    if timer > 0:
                        timer -= 1
                        grid[y, x, 2] = timer
                        # If just finished cooking, mark as cooked
                        if timer == 0:
                            grid[y, x, 1] |= DynamicObject.COOKED

        return state

    def get_valid_positions(self) -> List[Tuple[int, int]]:
        """Get all walkable floor positions."""
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout.static_objects[y, x] == StaticObject.EMPTY:
                    positions.append((x, y))
        return positions
