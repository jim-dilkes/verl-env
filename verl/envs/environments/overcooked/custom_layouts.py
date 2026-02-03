"""Custom Overcooked layouts for research experiments."""

from jaxmarl.environments.overcooked_v2.layouts import Layout

# Cramped room variant with mixed recipe: 2 onions + 1 tomato
# Left pile: tomato (ingredient 1)
# Right pile: onion (ingredient 0)
# Recipe: [0, 0, 1] = 2 onions + 1 tomato
cramped_room_mixed_recipe = """
WWPWW
1A A0
W   W
WBWXW
"""

def get_cramped_room_mixed():
    """Get cramped room layout with 2 onion + 1 tomato recipe.

    Returns:
        Layout: Cramped room with left=tomato, right=onion, recipe=[0,0,1]
    """
    return Layout.from_string(
        cramped_room_mixed_recipe,
        possible_recipes=[[0, 0, 1]],  # 2 onions + 1 tomato
    )


# Registry of custom layouts
CUSTOM_LAYOUTS = {
    "cramped_room_mixed": get_cramped_room_mixed(),
}


def get_custom_layout(name: str) -> Layout:
    """Get a custom layout by name.

    Args:
        name: Layout name from CUSTOM_LAYOUTS

    Returns:
        Layout object

    Raises:
        ValueError: If layout name not found
    """
    if name not in CUSTOM_LAYOUTS:
        raise ValueError(
            f"Unknown custom layout: {name}. "
            f"Available: {list(CUSTOM_LAYOUTS.keys())}"
        )
    return CUSTOM_LAYOUTS[name]
