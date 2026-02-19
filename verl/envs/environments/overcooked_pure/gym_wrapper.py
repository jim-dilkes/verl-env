"""Gymnasium wrapper for pure Python Overcooked environment.

This module provides the same interface as jaxmarl_wrapper.py but uses
the pure Python game engine instead of JaxMARL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from . import (
    ACTION_TO_IDX, IDX_TO_ACTION, DIRECTION_NAMES,
    StaticObject, DynamicObject, DEFAULT_POT_COOK_TIME
)
from .layouts import Layout, get_layout, BUILTIN_LAYOUTS, CUSTOM_LAYOUTS
from .game_engine import OvercookedEngine, GameState, Position


class OvercookedGymWrapper(gym.Env):
    """Gymnasium wrapper for pure Python Overcooked.

    Provides the same interface as the JaxMARL-based wrapper:
    - Single-agent control with configurable partner policy
    - Text rendering with grid visualization and coordinates
    - Shaped rewards
    - Solo mode support

    Partner policies:
        - "noop": Partner always stays in place
        - "random": Partner takes random actions
        - "none": Solo mode - partner hidden from observations/render
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        layout: str = "cramped_room",
        max_steps: int = 200,
        partner_policy: str = "noop",
        controlled_agent: str = "agent_0",
        seed: int = 0,
        shaped_reward: bool = True,
        print_visualization: bool = True,
        print_coordinates: bool = True,
        pot_cook_time: Optional[int] = None,
        random_agent_positions: bool = False,
    ):
        """Initialize the environment.

        Args:
            layout: Layout name or Layout object
            max_steps: Maximum steps per episode
            partner_policy: How partner behaves ("noop", "random", "none")
            controlled_agent: Which agent to control ("agent_0" or "agent_1")
            seed: Random seed
            shaped_reward: Whether to include shaped rewards
            print_visualization: Show ASCII grid in render
            print_coordinates: Show coordinate descriptions in render
            pot_cook_time: Override default cooking time (default: 20)
            random_agent_positions: Randomize agent spawn positions
        """
        super().__init__()

        self.layout_name = layout if isinstance(layout, str) else "custom"
        self.max_steps = max_steps
        self.partner_policy = partner_policy
        self.controlled_agent = controlled_agent
        self.partner_agent = "agent_1" if controlled_agent == "agent_0" else "agent_0"
        self.shaped_reward = shaped_reward
        self.print_visualization = print_visualization
        self.print_coordinates = print_coordinates
        self.solo_mode = partner_policy == "none"

        if not (print_visualization or print_coordinates):
            raise ValueError("At least one of print_visualization or print_coordinates must be True")

        self.pot_cook_time = pot_cook_time if pot_cook_time is not None else DEFAULT_POT_COOK_TIME

        # Get layout
        if isinstance(layout, str):
            self._layout = get_layout(layout)
        elif isinstance(layout, Layout):
            self._layout = layout
        else:
            raise ValueError(f"Invalid layout type: {type(layout)}")

        # Create engine
        self._engine = OvercookedEngine(
            layout=self._layout,
            max_steps=max_steps,
            pot_cook_time=self.pot_cook_time,
            random_agent_positions=random_agent_positions,
        )

        # RNG setup
        self._rng = np.random.default_rng(seed)
        self._state: Optional[GameState] = None
        self._step_count = 0
        self._last_event = ""

        # Cache for static objects (doesn't change during episode)
        self._cached_static_objects = None
        self._cached_pot_positions = None

        # Spaces
        self.action_space = spaces.Discrete(6)
        # Observation shape matches JaxMARL's default observation
        # For simplicity, we use a flat representation
        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32
        )

    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Get observation shape matching JaxMARL."""
        # JaxMARL uses a multi-channel grid observation
        # Channels: agent_pos(1) + agent_dir(4) + agent_inv(2+3*num_ing) +
        #           other_agent_pos(1) + other_agent_dir(4) + other_agent_inv(...) +
        #           static_layers(6) + ingredient_piles(num_ing) +
        #           ingredients(2+3*num_ing) + recipe(2+3*num_ing) +
        #           pot_timer(1)
        # This is complex - for now we'll use a simplified representation
        num_ing = self._layout.num_ingredients
        num_layers = 18 + 4 * (num_ing + 2)
        return (self._layout.height, self._layout.width, num_layers)

    def _get_partner_action(self) -> int:
        """Get action for partner agent."""
        if self.partner_policy == "noop" or self.partner_policy == "none":
            return 4  # stay
        elif self.partner_policy == "random":
            return int(self._rng.integers(0, 6))
        return 4

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional seed to reset RNG
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info dict)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = self._engine.reset(self._rng)
        self._step_count = 0
        self._last_event = "Episode started"

        # In solo mode, move partner off-map
        if self.solo_mode:
            partner_idx = 1 if self.controlled_agent == "agent_0" else 0
            self._state.agents[partner_idx].pos = Position(-1, -1)

        # Clear caches
        self._cached_static_objects = None
        self._cached_pot_positions = None

        obs = self._get_obs()
        return obs, {"state": self._state}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action index (0-5) or action name string

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to index
        if isinstance(action, str):
            action_idx = ACTION_TO_IDX.get(action.lower(), 4)
        else:
            action_idx = int(action)

        # Get partner action
        partner_action = self._get_partner_action()

        # Build actions list
        controlled_idx = 0 if self.controlled_agent == "agent_0" else 1
        partner_idx = 1 - controlled_idx

        actions = [0, 0]
        actions[controlled_idx] = action_idx
        actions[partner_idx] = partner_action

        # Execute step
        self._state, reward, shaped_rewards = self._engine.step(
            self._state, actions, self._rng
        )
        self._step_count += 1

        # Add shaped reward if enabled
        if self.shaped_reward:
            reward += shaped_rewards[controlled_idx]

        # Check termination
        terminated = self._state.terminal
        truncated = self._step_count >= self.max_steps and not terminated

        self._last_event = self._describe_step(action_idx, reward)

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, {
            "state": self._state,
            "step": self._step_count,
        }

    def _describe_step(self, action_idx: int, reward: float) -> str:
        """Generate description of the step."""
        action_name = IDX_TO_ACTION.get(action_idx, "unknown")
        if reward > 0:
            return f"Took action '{action_name}' and received reward {reward:.1f}!"
        return f"Took action '{action_name}'"

    def _get_obs(self) -> np.ndarray:
        """Get observation array.

        Returns a multi-channel grid observation matching JaxMARL format.
        """
        if self._state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        height = self._layout.height
        width = self._layout.width
        num_ing = self._layout.num_ingredients
        grid = self._state.grid

        layers = []

        # Controlled agent index
        ctrl_idx = 0 if self.controlled_agent == "agent_0" else 1
        partner_idx = 1 - ctrl_idx

        # Agent layers (position, direction, inventory)
        for agent_idx in [ctrl_idx, partner_idx]:
            agent = self._state.agents[agent_idx]

            # Position layer
            pos_layer = np.zeros((height, width), dtype=np.float32)
            if 0 <= agent.pos.y < height and 0 <= agent.pos.x < width:
                pos_layer[agent.pos.y, agent.pos.x] = 1
            layers.append(pos_layer)

            # Direction layers (one-hot)
            for d in range(4):
                dir_layer = np.zeros((height, width), dtype=np.float32)
                if 0 <= agent.pos.y < height and 0 <= agent.pos.x < width:
                    if agent.direction == d:
                        dir_layer[agent.pos.y, agent.pos.x] = 1
                layers.append(dir_layer)

            # Inventory layers
            inv = agent.inventory
            inv_layers = self._encode_item_layers(inv, height, width,
                                                  agent.pos.y, agent.pos.x, num_ing)
            layers.extend(inv_layers)

        # Static object layers
        static = grid[:, :, 0]
        static_types = [
            StaticObject.WALL,
            StaticObject.GOAL,
            StaticObject.POT,
            StaticObject.RECIPE_INDICATOR,
            StaticObject.BUTTON_RECIPE_INDICATOR,
            StaticObject.PLATE_PILE,
        ]
        for obj_type in static_types:
            layers.append((static == obj_type).astype(np.float32))

        # Ingredient pile layers
        for i in range(num_ing):
            pile_type = StaticObject.INGREDIENT_PILE_BASE + i
            layers.append((static == pile_type).astype(np.float32))

        # Dynamic item layers (ingredients on grid)
        dynamic = grid[:, :, 1]
        ing_layers = self._encode_grid_item_layers(dynamic, num_ing)
        layers.extend(ing_layers)

        # Recipe layers (at recipe indicator positions)
        recipe_mask = (static == StaticObject.RECIPE_INDICATOR) | \
                      ((static == StaticObject.BUTTON_RECIPE_INDICATOR) & (grid[:, :, 2] > 0))
        recipe_grid = np.where(recipe_mask, self._state.recipe, 0)
        recipe_layers = self._encode_grid_item_layers(recipe_grid.astype(np.int32), num_ing)
        layers.extend(recipe_layers)

        # Pot timer layer
        pot_mask = static == StaticObject.POT
        timer_layer = np.where(pot_mask, grid[:, :, 2], 0).astype(np.float32)
        layers.append(timer_layer)

        # Stack all layers
        obs = np.stack(layers, axis=-1)
        return obs

    def _encode_item_layers(
        self, item: int, height: int, width: int,
        y: int, x: int, num_ing: int
    ) -> List[np.ndarray]:
        """Encode item value as layers at a single position."""
        layers = []

        # Plate bit
        plate_layer = np.zeros((height, width), dtype=np.float32)
        if 0 <= y < height and 0 <= x < width:
            plate_layer[y, x] = float(item & 1)
        layers.append(plate_layer)

        # Cooked bit
        cooked_layer = np.zeros((height, width), dtype=np.float32)
        if 0 <= y < height and 0 <= x < width:
            cooked_layer[y, x] = float((item >> 1) & 1)
        layers.append(cooked_layer)

        # Ingredient counts
        for i in range(num_ing):
            count = (item >> (2 + 2 * i)) & 0x3
            ing_layer = np.zeros((height, width), dtype=np.float32)
            if 0 <= y < height and 0 <= x < width:
                ing_layer[y, x] = float(count)
            layers.append(ing_layer)

        return layers

    def _encode_grid_item_layers(self, items: np.ndarray, num_ing: int) -> List[np.ndarray]:
        """Encode grid of items as layers."""
        layers = []

        # Plate bit
        layers.append((items & 1).astype(np.float32))

        # Cooked bit
        layers.append(((items >> 1) & 1).astype(np.float32))

        # Ingredient counts
        for i in range(num_ing):
            layers.append(((items >> (2 + 2 * i)) & 0x3).astype(np.float32))

        return layers

    def get_state_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed state information for rendering."""
        if self._state is None:
            return None

        grid = self._state.grid

        agent_info = []
        agent_names = [self.controlled_agent] if self.solo_mode else [self.controlled_agent, self.partner_agent]

        for agent_name in agent_names:
            agent_idx = 0 if agent_name == "agent_0" else 1
            agent = self._state.agents[agent_idx]
            agent_info.append({
                "name": agent_name,
                "pos": (agent.pos.x, agent.pos.y),
                "direction": DIRECTION_NAMES.get(agent.direction, "UNKNOWN"),
                "inventory": agent.inventory,
                "is_controlled": agent_name == self.controlled_agent,
            })

        return {
            "agents": agent_info,
            "grid": grid,
            "time": self._state.time,
            "recipe": self._state.recipe,
            "terminal": self._state.terminal,
            "last_event": self._last_event,
            "solo_mode": self.solo_mode,
        }

    def render(self) -> str:
        """Render the environment as text."""
        if self._state is None:
            return "Environment not initialized. Call reset() first."
        return self._render_text()

    def _render_text(self) -> str:
        """Generate text representation."""
        info = self.get_state_info()
        if info is None:
            return "No state available"

        grid = info["grid"]
        height, width, _ = grid.shape
        lines = []

        if self.print_coordinates:
            lines.extend(self._render_coordinates(info, grid))

        if self.print_visualization:
            if lines:
                lines.append("")
            lines.extend(self._render_grid(info, grid))

        lines.append(f"\nStep: {info['time']}/{self.max_steps}")

        return "\n".join(lines)

    def _render_grid(self, info: Dict, grid: np.ndarray) -> List[str]:
        """Render ASCII grid visualization."""
        height, width, _ = grid.shape
        lines = []

        agent_positions = {
            (a["pos"][0], a["pos"][1]): ("@" if a["is_controlled"] else "X")
            for a in info["agents"]
        }

        lines.append(f"Kitchen ({self.layout_name}, {width}x{height}):")
        if self.solo_mode:
            lines.append("Legend: #=wall .=floor P=pot S=serve D=dish 0-9=ingredients @=you")
        else:
            lines.append("Legend: #=wall .=floor P=pot S=serve D=dish 0-9=ingredients @=you X=partner")

        for y in range(height):
            row = []
            for x in range(width):
                if (x, y) in agent_positions:
                    row.append(agent_positions[(x, y)])
                else:
                    cell = grid[y, x]
                    row.append(self._cell_to_char(cell))
            lines.append(" ".join(row))

        return lines

    def _render_coordinates(self, info: Dict, grid: np.ndarray) -> List[str]:
        """Render text coordinate descriptions."""
        height, width, _ = grid.shape
        lines = []

        lines.append(f"Layout: {self.layout_name} ({width}x{height})")
        lines.append("Coordinates: (x, y) where x=column, y=row from top-left (0,0)")

        # Agents
        lines.append("")
        for agent in info["agents"]:
            name = "You" if agent["is_controlled"] else "Partner"
            inv_str = self._inventory_to_str(agent["inventory"])
            lines.append(
                f"{name}: pos={agent['pos']}, "
                f"facing {agent['direction']}, holding: {inv_str}"
            )

        # Pots
        pots = self._get_pot_info(grid)
        if pots:
            lines.append("")
            for pot in pots:
                pot_status = self._pot_status_str(pot)
                lines.append(f"Pot at {pot['pos']}: {pot_status}")

        # Static objects
        lines.append("")
        objects = self._get_static_objects(grid)
        for obj_type, positions in objects.items():
            if positions:
                pos_str = ", ".join(str(p) for p in positions)
                lines.append(f"{obj_type}: {pos_str}")

        return lines

    def _get_static_objects(self, grid: np.ndarray) -> Dict[str, List]:
        """Get positions of static objects for coordinate display."""
        if self._cached_static_objects is not None:
            return self._cached_static_objects

        height, width, _ = grid.shape
        objects = {
            "Serving counter": [],
            "Dish pile": [],
            "Ingredient piles": [],
        }

        for y in range(height):
            for x in range(width):
                static = int(grid[y, x, 0])
                if static == StaticObject.GOAL:
                    objects["Serving counter"].append((x, y))
                elif static == StaticObject.PLATE_PILE:
                    objects["Dish pile"].append((x, y))
                elif static >= StaticObject.INGREDIENT_PILE_BASE:
                    idx = static - StaticObject.INGREDIENT_PILE_BASE
                    name = self._ingredient_name(idx)
                    objects["Ingredient piles"].append((x, y, name))

        self._cached_static_objects = objects
        return objects

    def _cell_to_char(self, cell: np.ndarray) -> str:
        """Convert grid cell to ASCII character."""
        static = cell[0]
        if static == StaticObject.EMPTY:
            return "."
        elif static == StaticObject.WALL:
            return "#"
        elif static == StaticObject.GOAL:
            return "S"
        elif static == StaticObject.POT:
            return "P"
        elif static == StaticObject.RECIPE_INDICATOR:
            return "R"
        elif static == StaticObject.BUTTON_RECIPE_INDICATOR:
            return "B"
        elif static == StaticObject.PLATE_PILE:
            return "D"
        elif static >= StaticObject.INGREDIENT_PILE_BASE:
            ingredient_idx = static - StaticObject.INGREDIENT_PILE_BASE
            return str(ingredient_idx)
        return "?"

    def _decode_item(self, val: int) -> Optional[Dict]:
        """Decode dynamic item encoding to components."""
        if val == 0:
            return None
        plate = bool(val & 1)
        cooked = bool(val & 2)
        ingredient_counts = {}
        for i in range(3):
            count = (val >> (2 + 2 * i)) & 3
            if count > 0:
                ingredient_counts[i] = count
        return {"plate": plate, "cooked": cooked, "ingredient_counts": ingredient_counts}

    def _ingredient_name(self, idx: int) -> str:
        """Get ingredient name by index."""
        names = ["onion", "tomato", "lettuce"]
        return names[idx] if idx < len(names) else f"ingredient_{idx}"

    def _inventory_to_str(self, inv: int) -> str:
        """Decode inventory to human-readable string."""
        decoded = self._decode_item(inv)
        if decoded is None:
            return "NOTHING"

        ing_counts = decoded["ingredient_counts"]
        ing_parts = []
        for idx, count in ing_counts.items():
            name = self._ingredient_name(idx)
            if count > 1:
                ing_parts.append(f"{count}x {name}")
            else:
                ing_parts.append(name)
        ing_str = "+".join(ing_parts)

        if decoded["plate"] and decoded["cooked"] and ing_counts:
            return f"SOUP ({ing_str}) on plate"
        elif decoded["plate"] and ing_counts:
            return f"plate with {ing_str}"
        elif decoded["plate"]:
            return "empty plate"
        elif ing_counts:
            return ing_str.upper()
        return "UNKNOWN"

    def _get_pot_info(self, grid: np.ndarray) -> List[Dict]:
        """Get pot status from grid."""
        if self._cached_pot_positions is None:
            height, width, _ = grid.shape
            self._cached_pot_positions = []
            for y in range(height):
                for x in range(width):
                    if grid[y, x, 0] == StaticObject.POT:
                        self._cached_pot_positions.append((x, y))

        pots = []
        for x, y in self._cached_pot_positions:
            contents = int(grid[y, x, 1])
            timer = int(grid[y, x, 2])
            decoded = self._decode_item(contents)
            pots.append({
                "pos": (x, y),
                "contents": decoded,
                "timer": timer,
            })
        return pots

    def _pot_status_str(self, pot: Dict) -> str:
        """Format pot status as string."""
        contents = pot["contents"]
        timer = pot["timer"]

        if contents is None:
            return "empty"

        ing_counts = contents["ingredient_counts"]
        total_count = sum(ing_counts.values())
        ing_parts = []
        for idx, count in ing_counts.items():
            name = self._ingredient_name(idx)
            ing_parts.append(f"{count}x {name}")
        ing_str = ", ".join(ing_parts)

        if contents["cooked"]:
            return f"READY! soup ({ing_str}) - pick up with plate"
        elif timer > 0:
            return f"cooking ({ing_str}) - {timer} ticks left"
        elif total_count > 0:
            return f"{ing_str} ({total_count}/3 ingredients)"
        return "empty"

    @property
    def last_event(self) -> str:
        """Get last event description."""
        return self._last_event
