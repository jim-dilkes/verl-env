"""Layout definitions and parsing for Overcooked pure Python implementation.

Layouts are parsed from string representations matching JaxMARL's format.
"""

import itertools
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from . import StaticObject


# ============================================================================
# Layout strings from JaxMARL overcooked_v2
# ============================================================================

# Original Overcooked-AI layouts
cramped_room = """
WWPWW
OA AO
W   W
WBWXW
"""

asymm_advantages = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""

coord_ring = """
WWWPW
W A P
BAW W
O   W
WOXWW
"""

forced_coord = """
WWWPW
O WAP
OAW W
B W W
WWWXW
"""

counter_circuit = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""


# ============================================================================
# Custom layouts
# ============================================================================

# Cramped room with mixed recipe: 2 onions + 1 tomato
# Left pile: tomato (ingredient 1)
# Right pile: onion (ingredient 0)
cramped_room_mixed = """
WWPWW
1A A0
W   W
WBWXW
"""


# ============================================================================
# Layout dataclass
# ============================================================================

@dataclass
class Layout:
    """Represents an Overcooked kitchen layout.

    Attributes:
        agent_positions: List of (x, y) spawn points
        static_objects: NumPy array [height, width] of static object types
        num_ingredients: Number of distinct ingredient types
        possible_recipes: List of valid recipes, each a list of 3 ingredient indices
    """
    agent_positions: List[Tuple[int, int]]
    static_objects: np.ndarray
    num_ingredients: int
    possible_recipes: List[List[int]]

    def __post_init__(self):
        if len(self.agent_positions) == 0:
            raise ValueError("At least one agent position must be provided")
        if self.num_ingredients < 1:
            raise ValueError("At least one ingredient must be available")
        if self.possible_recipes is None:
            self.possible_recipes = self._get_all_possible_recipes(self.num_ingredients)

    @property
    def height(self) -> int:
        return self.static_objects.shape[0]

    @property
    def width(self) -> int:
        return self.static_objects.shape[1]

    @staticmethod
    def _get_all_possible_recipes(num_ingredients: int) -> List[List[int]]:
        """Generate all possible 3-ingredient recipes."""
        available_ingredients = list(range(num_ingredients)) * 3
        raw_combinations = itertools.combinations(available_ingredients, 3)
        unique_recipes = set(
            tuple(sorted(combination)) for combination in raw_combinations
        )
        return [list(recipe) for recipe in unique_recipes]

    @staticmethod
    def from_string(
        grid: str,
        possible_recipes: Optional[List[List[int]]] = None,
        swap_agents: bool = False
    ) -> 'Layout':
        """Parse a layout from string representation.

        Character meanings:
            W: wall/counter
            A: agent spawn point
            X: serving counter (goal)
            B: plate (bowl) pile
            P: pot
            R: recipe indicator
            L: button recipe indicator
            0-9: ingredient pile (index)
            ' ': empty floor
            O: onion pile (deprecated, same as '0')

        Args:
            grid: Multi-line string layout
            possible_recipes: List of [idx, idx, idx] recipes. If None, all are allowed.
            swap_agents: Reverse agent order (for JaxMARL compatibility)

        Returns:
            Layout object
        """
        rows = grid.strip().split("\n")
        
        # Remove empty rows
        rows = [r for r in rows if len(r) > 0]

        row_lens = [len(row) for row in rows]
        height = len(rows)
        width = max(row_lens)
        
        static_objects = np.zeros((height, width), dtype=np.int32)

        char_to_static = {
            " ": StaticObject.EMPTY,
            "W": StaticObject.WALL,
            "X": StaticObject.GOAL,
            "B": StaticObject.PLATE_PILE,
            "P": StaticObject.POT,
            "R": StaticObject.RECIPE_INDICATOR,
            "L": StaticObject.BUTTON_RECIPE_INDICATOR,
        }

        # Add ingredient piles
        for i in range(10):
            char_to_static[str(i)] = StaticObject.INGREDIENT_PILE_BASE + i

        agent_positions = []
        num_ingredients = 0
        includes_recipe_indicator = False

        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                # Handle deprecated 'O' as '0'
                if char == "O":
                    char = "0"

                # Record agent spawn points
                if char == "A":
                    agent_positions.append((c, r))
                    # Agent spawns on empty floor
                    static_objects[r, c] = StaticObject.EMPTY
                else:
                    obj = char_to_static.get(char, StaticObject.EMPTY)
                    static_objects[r, c] = obj

                    if StaticObject.is_ingredient_pile(obj):
                        ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                        num_ingredients = max(num_ingredients, ingredient_idx + 1)

                    if obj == StaticObject.RECIPE_INDICATOR:
                        includes_recipe_indicator = True

        # Validation
        if possible_recipes is not None:
            if not isinstance(possible_recipes, list):
                raise ValueError("possible_recipes must be a list")
            if not all(isinstance(r, list) for r in possible_recipes):
                raise ValueError("possible_recipes must be a list of lists")
            if not all(len(r) == 3 for r in possible_recipes):
                raise ValueError("All recipes must be of length 3")
        elif not includes_recipe_indicator:
            raise ValueError(
                "Layout does not include a recipe indicator, a fixed recipe must be provided"
            )

        if swap_agents:
            agent_positions = agent_positions[::-1]

        # Default to 1 ingredient if none found
        if num_ingredients == 0:
            num_ingredients = 1

        # Generate all recipes if not provided and has indicator
        if possible_recipes is None:
            possible_recipes = Layout._get_all_possible_recipes(num_ingredients)

        return Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            possible_recipes=possible_recipes,
        )


# ============================================================================
# Pre-built layouts registry
# ============================================================================

BUILTIN_LAYOUTS = {
    # Overcooked-AI layouts (matching JaxMARL)
    "cramped_room": Layout.from_string(
        cramped_room, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),
    "asymm_advantages": Layout.from_string(
        asymm_advantages, possible_recipes=[[0, 0, 0]]
    ),
    "coord_ring": Layout.from_string(
        coord_ring, possible_recipes=[[0, 0, 0]]
    ),
    "forced_coord": Layout.from_string(
        forced_coord, possible_recipes=[[0, 0, 0]]
    ),
    "counter_circuit": Layout.from_string(
        counter_circuit, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),
}

# Custom layouts
CUSTOM_LAYOUTS = {
    "cramped_room_mixed": Layout.from_string(
        cramped_room_mixed, possible_recipes=[[0, 0, 1]]  # 2 onions + 1 tomato
    ),
}


def get_layout(name: str) -> Layout:
    """Get a layout by name.

    Args:
        name: Layout name (builtin or custom)

    Returns:
        Layout object

    Raises:
        ValueError: If layout not found
    """
    if name in BUILTIN_LAYOUTS:
        return BUILTIN_LAYOUTS[name]
    if name in CUSTOM_LAYOUTS:
        return CUSTOM_LAYOUTS[name]
    raise ValueError(
        f"Unknown layout: {name}. "
        f"Available: {list(BUILTIN_LAYOUTS.keys()) + list(CUSTOM_LAYOUTS.keys())}"
    )
