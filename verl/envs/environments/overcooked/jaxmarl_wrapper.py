import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2 import OvercookedV2
from jaxmarl.environments.overcooked_v2.overcooked import Actions, Direction

from verl.envs.environments.overcooked import ACTION_TO_IDX, IDX_TO_ACTION, DIRECTION_NAMES


class OvercookedGymWrapper(gym.Env):
    """Gymnasium wrapper for JaxMARL Overcooked V2.

    Converts the functional JAX-based environment to a stateful gym interface.
    Supports single-agent mode where the controlled agent is agent_0 and agent_1
    follows a configurable policy (noop, random, or controlled).
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
    ):
        super().__init__()

        self.layout = layout
        self.max_steps = max_steps
        self.partner_policy = partner_policy
        self.controlled_agent = controlled_agent
        self.partner_agent = "agent_1" if controlled_agent == "agent_0" else "agent_0"
        self.shaped_reward = shaped_reward

        self._env = OvercookedV2(layout=layout, max_steps=max_steps)
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._last_obs = None
        self._step_count = 0
        self._last_event = ""

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
        if self.partner_policy == "noop":
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
        reward = float(rewards[self.controlled_agent])
        if self.shaped_reward and "shaped_reward" in info:
            reward += float(info["shaped_reward"][self.controlled_agent])

        terminated = bool(dones["__all__"])
        truncated = self._step_count >= self.max_steps and not terminated

        self._last_event = self._describe_step(action_idx, reward)

        return obs_array, reward, terminated, truncated, {
            "state": self._state,
            "step": self._step_count,
        }

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

        agent_info = []
        for i, agent_name in enumerate([self.controlled_agent, self.partner_agent]):
            pos_x = int(agents.pos.x[i])
            pos_y = int(agents.pos.y[i])
            direction = int(agents.dir[i])
            inventory = int(agents.inventory[i])

            agent_info.append({
                "name": agent_name,
                "pos": (pos_x, pos_y),
                "direction": DIRECTION_NAMES.get(direction, "UNKNOWN"),
                "inventory": inventory,
                "is_controlled": agent_name == self.controlled_agent,
            })

        return {
            "agents": agent_info,
            "grid": grid,
            "time": int(self._state.time),
            "recipe": int(self._state.recipe),
            "terminal": bool(self._state.terminal),
            "last_event": self._last_event,
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

        agent_positions = {
            (a["pos"][0], a["pos"][1]): ("@" if a["is_controlled"] else "X")
            for a in info["agents"]
        }

        lines = [f"Kitchen ({self.layout}, {width}x{height}):"]
        for y in range(height):
            row = []
            for x in range(width):
                if (x, y) in agent_positions:
                    row.append(agent_positions[(x, y)])
                else:
                    cell = grid[y, x]
                    row.append(self._cell_to_char(cell))
            lines.append(" ".join(row))

        lines.append("")
        for agent in info["agents"]:
            marker = "(you)" if agent["is_controlled"] else "(partner)"
            inv_str = self._inventory_to_str(agent["inventory"])
            lines.append(
                f"Agent {agent['name']} {marker}: pos={agent['pos']}, "
                f"facing {agent['direction']}, holding: {inv_str}"
            )

        lines.append(f"\nStep: {info['time']}/{self.max_steps}")
        lines.append(f"Last: {info['last_event']}")

        return "\n".join(lines)

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

    def _inventory_to_str(self, inv):
        if inv == 0:
            return "NOTHING"
        if inv & 1:
            return "PLATE"
        return "INGREDIENT"

    @property
    def last_event(self):
        return self._last_event
