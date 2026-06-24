import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp

import jaxmarl.environments.overcooked_v2.settings as overcooked_settings
import jaxmarl.environments.overcooked_v2.overcooked as overcooked_module
from jaxmarl.environments.overcooked_v2 import OvercookedV2
from jaxmarl.environments.overcooked_v2.overcooked import Actions, Direction

from verl.envs.environments.overcooked import ACTION_TO_IDX, IDX_TO_ACTION, DIRECTION_NAMES

from verl.envs.environments.overcooked.milestones import MILESTONE_NAMES, compute_milestones

JAXMARL_DEFAULT_COOK_TIME = 20


class OvercookedGymWrapper(gym.Env):
    """Gymnasium wrapper for JaxMARL Overcooked V2.

    Converts the functional JAX-based environment to a stateful gym interface.
    Supports single-agent mode where the controlled agent is agent_0 and agent_1
    follows a configurable policy (noop, random, none).

    Partner policies:
        - "noop": Partner always stays in place
        - "random": Partner takes random actions
        - "none": Solo mode - partner hidden from observations/render

    Game rule customization:
        - pot_cook_time: Override cooking duration (default 20 ticks)

    Note: JaxMARL enforces 3 ingredients per recipe. This cannot be changed.
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
        pot_cook_time: int = None,
        random_agent_positions: bool = False,
    ):
        super().__init__()

        self.layout = layout
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

        if pot_cook_time is not None:
            # Patch both settings and the module that imports it at load time
            overcooked_settings.POT_COOK_TIME = pot_cook_time
            overcooked_module.POT_COOK_TIME = pot_cook_time
            self.pot_cook_time = pot_cook_time
        else:
            self.pot_cook_time = JAXMARL_DEFAULT_COOK_TIME

        self._env = OvercookedV2(
            layout=layout, max_steps=max_steps,
            random_agent_positions=random_agent_positions,
        )
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._last_obs = None
        self._step_count = 0
        self._last_event = ""

        # Cached layout info (static objects don't move)
        self._cached_static_objects = None
        self._cached_pot_positions = None

        self.action_space = spaces.Discrete(6)
        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32
        )

    def _get_obs_shape(self):
        self._key, reset_key = jax.random.split(self._key)
        obs, _ = self._env.reset(reset_key)
        return obs[self.controlled_agent].shape

    def _get_partner_action(self):
        if self.partner_policy == "noop" or self.partner_policy == "none":
            return 4  # stay
        elif self.partner_policy == "random":
            return np.random.randint(0, 6)
        else:
            return 4

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)

        self._key, reset_key = jax.random.split(self._key)
        obs, self._state = self._env.reset(reset_key)
        self._last_obs = obs
        self._step_count = 0
        self._last_event = "Episode started"

        # In solo mode, move partner agent off-map so they don't block tiles
        if self.solo_mode:
            partner_idx = 1 if self.controlled_agent == "agent_0" else 0
            # Move partner to position (-1, -1) - off the grid
            new_pos_x = self._state.agents.pos.x.at[partner_idx].set(-1)
            new_pos_y = self._state.agents.pos.y.at[partner_idx].set(-1)
            new_pos = self._state.agents.pos.replace(x=new_pos_x, y=new_pos_y)
            new_agents = self._state.agents.replace(pos=new_pos)
            self._state = self._state.replace(agents=new_agents)

        # Clear caches - layout static objects need recomputing
        self._cached_static_objects = None
        self._cached_pot_positions = None

        obs_array = np.array(obs[self.controlled_agent])
        return obs_array, {"state": self._state}

    def step(self, action):
        if isinstance(action, str):
            action_idx = ACTION_TO_IDX.get(action.lower(), 4)
        else:
            action_idx = int(action)

        partner_action = self._get_partner_action()

        actions = {
            self.controlled_agent: action_idx,
            self.partner_agent: partner_action,
        }

        self._key, step_key = jax.random.split(self._key)
        obs, self._state, rewards, dones, info = self._env.step(
            step_key, self._state, actions
        )
        self._last_obs = obs
        self._step_count += 1

        obs_array = np.array(obs[self.controlled_agent])
        sparse_reward = float(rewards[self.controlled_agent])  # delivery reward (pre-shaping)
        reward = sparse_reward
        if self.shaped_reward and "shaped_reward" in info:
            reward += float(info["shaped_reward"][self.controlled_agent])

        terminated = bool(dones["__all__"])
        truncated = self._step_count >= self.max_steps and not terminated

        self._last_event = self._describe_step(action_idx, reward)

        return obs_array, reward, terminated, truncated, {
            "state": self._state,
            "step": self._step_count,
            "milestones": self._compute_milestones(sparse_reward),
        }

    def _compute_milestones(self, sparse_reward):
        """Per-step boolean flags for the ordered recipe-agnostic task chain.

        MILESTONE_NAMES gives the order: holding a raw ingredient -> >=1 ingredient in
        a pot -> a pot cooking/cooked -> holding a dish -> holding cooked soup ->
        delivered. Derived from the controlled agent's inventory + pot contents (same
        decode helpers used for rendering); delivery uses the sparse (pre-shaping)
        reward. The evaluator aggregates per-trajectory furthest-reached over an episode.
        """
        state = self._state
        grid = np.array(state.grid)
        inv_np = np.array(state.agents.inventory)
        agent_idx = 0 if self.controlled_agent == "agent_0" else 1
        held = self._decode_item(int(inv_np[agent_idx]))
        pots = self._get_pot_info(grid)
        return compute_milestones(held, pots, sparse_reward)

    def _describe_step(self, action_idx, reward):
        action_name = IDX_TO_ACTION.get(action_idx, "unknown")
        if reward > 0:
            return f"Took action '{action_name}' and received reward {reward:.1f}!"
        return f"Took action '{action_name}'"

    def get_state_info(self):
        if self._state is None:
            return None

        agents = self._state.agents
        grid = np.array(self._state.grid)

        # Batch JAX→NumPy conversions (one conversion per array, not per index)
        # This avoids expensive JAX dispatch for each array access
        pos_x_np = np.array(agents.pos.x)
        pos_y_np = np.array(agents.pos.y)
        dir_np = np.array(agents.dir)
        inv_np = np.array(agents.inventory)

        agent_info = []
        agent_names = [self.controlled_agent] if self.solo_mode else [self.controlled_agent, self.partner_agent]
        for agent_name in agent_names:
            agent_idx = 0 if agent_name == "agent_0" else 1
            agent_info.append({
                "name": agent_name,
                "pos": (int(pos_x_np[agent_idx]), int(pos_y_np[agent_idx])),
                "direction": DIRECTION_NAMES.get(int(dir_np[agent_idx]), "UNKNOWN"),
                "inventory": int(inv_np[agent_idx]),
                "is_controlled": agent_name == self.controlled_agent,
            })

        return {
            "agents": agent_info,
            "grid": grid,
            "time": int(self._state.time),
            "recipe": int(self._state.recipe),
            "terminal": bool(self._state.terminal),
            "last_event": self._last_event,
            "solo_mode": self.solo_mode,
        }

    def render(self):
        if self._state is None:
            return "Environment not initialized. Call reset() first."
        return self._render_text()

    def _render_text(self):
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

    def _render_grid(self, info, grid):
        """Render ASCII grid visualization."""
        height, width, _ = grid.shape
        lines = []

        agent_positions = {
            (a["pos"][0], a["pos"][1]): ("@" if a["is_controlled"] else "X")
            for a in info["agents"]
        }

        lines.append(f"Kitchen ({self.layout}, {width}x{height}):")
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

    def _render_coordinates(self, info, grid):
        """Render text coordinate descriptions."""
        height, width, _ = grid.shape
        lines = []

        lines.append(f"Layout: {self.layout} ({width}x{height})")
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

    def _get_static_objects(self, grid):
        """Get positions of static objects for coordinate display.

        Cached because static objects don't move during episode.
        """
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
                if static == 4:
                    objects["Serving counter"].append((x, y))
                elif static == 9:
                    objects["Dish pile"].append((x, y))
                elif static >= 10:
                    idx = static - 10
                    name = self._ingredient_name(idx)
                    objects["Ingredient piles"].append((x, y, name))

        self._cached_static_objects = objects
        return objects

    def _cell_to_char(self, cell):
        static = cell[0]
        if static == 0:
            return "."  # empty floor
        elif static == 1:
            return "#"  # wall
        elif static == 4:
            return "S"  # goal/serving counter
        elif static == 5:
            return "P"  # pot
        elif static == 6:
            return "R"  # recipe indicator
        elif static == 7:
            return "B"  # button recipe indicator
        elif static == 9:
            return "D"  # dish/plate pile
        elif static >= 10:
            ingredient_idx = static - 10
            return str(ingredient_idx)  # ingredient pile (0, 1, 2, ...)
        return "?"

    def _decode_item(self, val):
        """Decode dynamic item encoding to components.

        Encoding: bit 0 = plate, bit 1 = cooked, bits 2-7 = ingredient counts
        Each ingredient uses 2 bits for count (0-3), shifted by 2*idx from bit 2.
        """
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

    def _ingredient_name(self, idx):
        """Get ingredient name by index."""
        names = ["onion", "tomato", "lettuce"]
        return names[idx] if idx < len(names) else f"ingredient_{idx}"

    def _inventory_to_str(self, inv):
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

    def _get_pot_info(self, grid):
        """Get pot status from grid.

        Pot positions are cached; only contents/timer change.
        """
        # Cache pot positions on first call (positions don't change)
        if self._cached_pot_positions is None:
            height, width, _ = grid.shape
            self._cached_pot_positions = []
            for y in range(height):
                for x in range(width):
                    if grid[y, x, 0] == 5:  # POT
                        self._cached_pot_positions.append((x, y))

        # Only read contents at known pot positions
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

    def _pot_status_str(self, pot):
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
    def last_event(self):
        return self._last_event
