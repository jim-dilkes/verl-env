"""Pure Python Overcooked environment without JAX dependencies.

This module provides a JAX-free reimplementation of the Overcooked environment
for RL training. It maintains behavioral parity with the JaxMARL version.

Usage:
    from verl.envs.environments.overcooked_v3 import (
        OvercookedGymWrapper,
        OvercookedLLMAgentsWrapper,
        make_overcooked_env,
        ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION,
    )

    # Direct usage
    env = OvercookedGymWrapper(layout="cramped_room")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step("interact")

    # With LLM wrapper
    env = OvercookedLLMAgentsWrapper(OvercookedGymWrapper(layout="cramped_room"))
"""

# Action definitions - must match JaxMARL exactly
ACTIONS = {
    "right": "move right",
    "down": "move down",
    "left": "move left",
    "up": "move up",
    "stay": "stay in place (wait)",
    "interact": "interact with object in front of you (pick up, place, or use)",
}

ACTION_TO_IDX = {
    "right": 0,
    "down": 1,
    "left": 2,
    "up": 3,
    "stay": 4,
    "interact": 5,
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

# Direction encoding - must match JaxMARL exactly
# UP=0, DOWN=1, RIGHT=2, LEFT=3
DIRECTION_NAMES = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}


# Static object types (from JaxMARL common.py)
class StaticObject:
    """Static object type constants."""
    EMPTY = 0
    WALL = 1  # Also used for counters
    # AGENT = 2  # Only in observations
    # SELF_AGENT = 3  # Only in observations
    GOAL = 4  # Serving counter
    POT = 5
    RECIPE_INDICATOR = 6
    BUTTON_RECIPE_INDICATOR = 7
    PLATE_PILE = 9
    INGREDIENT_PILE_BASE = 10

    @staticmethod
    def is_ingredient_pile(obj: int) -> bool:
        return obj >= StaticObject.INGREDIENT_PILE_BASE

    @staticmethod
    def get_ingredient(obj: int) -> int:
        """Get encoded ingredient value from pile type."""
        idx = obj - StaticObject.INGREDIENT_PILE_BASE
        return DynamicObject.ingredient(idx)

    @staticmethod
    def ingredient_pile(idx: int) -> int:
        """Get pile type for ingredient index."""
        return StaticObject.INGREDIENT_PILE_BASE + idx


class DynamicObject:
    """Dynamic object encoding constants and helpers."""
    EMPTY = 0
    PLATE = 1 << 0  # bit 0
    COOKED = 1 << 1  # bit 1
    BASE_INGREDIENT = 1 << 2  # bits 2+ for ingredient counts

    @staticmethod
    def ingredient(idx: int) -> int:
        """Get encoded value for one unit of ingredient at index idx."""
        return DynamicObject.BASE_INGREDIENT << (2 * idx)

    @staticmethod
    def is_ingredient(obj: int) -> bool:
        """Check if object contains only ingredients (no plate)."""
        return ((obj >> 2) != 0) and ((obj & DynamicObject.PLATE) == 0)

    @staticmethod
    def ingredient_count(obj: int) -> int:
        """Count total ingredients in encoded object."""
        obj_shifted = obj >> 2
        count = 0
        while obj_shifted > 0:
            count += obj_shifted & 0x3
            obj_shifted >>= 2
        return count

    @staticmethod
    def get_recipe_encoding(recipe: list) -> int:
        """Encode a recipe [idx, idx, idx] as an integer."""
        total = 0
        for idx in recipe:
            total += DynamicObject.ingredient(idx)
        return total


# Direction constants
class Direction:
    """Direction constants matching JaxMARL."""
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


# Direction vectors: (dx, dy) for each direction
DIR_TO_VEC = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
}

# Action to direction mapping
# Movement actions set the facing direction to match the movement
ACTION_TO_DIRECTION = {
    0: Direction.RIGHT,   # right
    1: Direction.DOWN,    # down
    2: Direction.LEFT,    # left
    3: Direction.UP,      # up
    4: -1,                # stay - no direction change
    5: -1,                # interact - no direction change
}


# Game constants
DEFAULT_POT_COOK_TIME = 20
DELIVERY_REWARD = 20

# Shaped reward values (matching JaxMARL settings.py)
SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 3,
    "POT_START_COOKING": 5,
    "DISH_PICKUP": 5,
    "PLATE_PICKUP": 3,
}


# Lazy imports for main classes
def __getattr__(name):
    """Lazy import for main classes to avoid circular imports."""
    if name == "OvercookedGymWrapper":
        from .gym_wrapper import OvercookedGymWrapper
        return OvercookedGymWrapper
    elif name == "OvercookedLLMAgentsWrapper":
        from .base import OvercookedLLMAgentsWrapper
        return OvercookedLLMAgentsWrapper
    elif name == "make_overcooked_env":
        from .overcooked_env import make_overcooked_env
        return make_overcooked_env
    elif name == "Layout":
        from .layouts import Layout
        return Layout
    elif name == "get_layout":
        from .layouts import get_layout
        return get_layout
    elif name == "OvercookedEngine":
        from .game_engine import OvercookedEngine
        return OvercookedEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Constants
    "ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "DIRECTION_NAMES",
    "StaticObject",
    "DynamicObject",
    "Direction",
    "DIR_TO_VEC",
    "ACTION_TO_DIRECTION",
    "DEFAULT_POT_COOK_TIME",
    "DELIVERY_REWARD",
    "SHAPED_REWARDS",
    # Classes (lazy loaded)
    "OvercookedGymWrapper",
    "OvercookedLLMAgentsWrapper",
    "make_overcooked_env",
    "Layout",
    "get_layout",
    "OvercookedEngine",
]
