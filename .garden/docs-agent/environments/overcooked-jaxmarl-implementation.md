# Overcooked V2 Implementation Details

## Branch: `feat/overcooked_environment`

## JaxMARL V2 Internals

### Grid Encoding (3-channel per cell)
- `grid[y, x, 0]` - Static object type (StaticObject enum)
- `grid[y, x, 1]` - Dynamic item encoding (what's on/in the object)
- `grid[y, x, 2]` - Timer (for pots)

### StaticObject Constants
```python
EMPTY = 0      # floor
WALL = 1
COUNTER = 2
ONION_PILE = 3  # Actually uses INGREDIENT_PILE_BASE + idx
GOAL = 4        # Serving counter
POT = 5
RECIPE_INDICATOR = 6
BUTTON_RECIPE_INDICATOR = 7
PLATE_PILE = 9
INGREDIENT_PILE_BASE = 10  # ingredient idx = static - 10
```

### Dynamic Item Encoding (inventory, pot contents, counter items)
```
bit 0     = has plate
bit 1     = is cooked
bits 2-3  = ingredient 0 count (0-3)
bits 4-5  = ingredient 1 count (0-3)
bits 6-7  = ingredient 2 count (0-3)
```

Decoding:
```python
def _decode_item(val):
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
```

### Ingredients
- Index 0: onion
- Index 1: tomato
- Index 2: lettuce

### Actions
```python
ACTION_TO_IDX = {
    "right": 0,    # Direction.RIGHT
    "down": 1,     # Direction.DOWN
    "left": 2,     # Direction.LEFT
    "up": 3,       # Direction.UP
    "stay": 4,     # Actions.STAY
    "interact": 5  # Actions.INTERACT
}
```

Movement changes both position AND facing direction simultaneously.

### Direction Encoding
```python
DIRECTION_NAMES = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}
```

## Gameplay Loop

1. Pick up ingredient from pile (face pile + interact)
2. Place ingredient in pot (face pot + interact) - repeat 3x
3. Wait for cooking (timer counts down automatically)
4. Pick up plate from dish pile
5. Pick up soup from pot (face pot with plate + interact)
6. Deliver to serving counter (face counter + interact)

Each successful delivery = +20 reward (base)

## Shaped Rewards

When `shaped_reward=True`, adds intermediate rewards for:
- Picking up ingredients
- Adding to pot
- Picking up cooked soup
- etc.

Accessed via `info["shaped_reward"][agent_name]`

## JAX CPU Backend

Set before any JAX imports:
```python
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
```

Required because GPU JAX causes CUDA context issues with verl's async parallel workers.

## Agent State Access

```python
agents = state.agents
pos_x = int(agents.pos.x[agent_idx])
pos_y = int(agents.pos.y[agent_idx])
direction = int(agents.dir[agent_idx])
inventory = int(agents.inventory[agent_idx])
```

Agent indices: 0 = agent_0, 1 = agent_1

## LLMAgentsWrapper Interface

Required methods:
- `language_action_space` - dict of action_name -> description
- `extract_action(text)` - parse LLM output to action index
- `restructure_obs(obs)` - return `{text: {long_term_context, short_term_context}, state}`

Action extraction uses XML tags: `<action>interact</action>`

## Config Structure

```python
config.envs.overcooked_kwargs = {
    "layout": "cramped_room",
    "max_steps": 200,
    "partner_policy": "noop",
    "shaped_reward": True,
    "print_visualization": True,
    "print_coordinates": True,
}
```

## Available Layouts

From JaxMARL:
- cramped_room (5x4, simple)
- asymmetric_advantages
- coordination_ring
- forced_coordination
- counter_circuit
- Many more via `overcooked_v2_layouts.keys()`

## Testing Commands

```bash
# Interactive play
python -m verl.envs.environments.overcooked.interactive_play

# List layouts
python -m verl.envs.environments.overcooked.interactive_play --list-layouts

# Specific layout with random partner
python -m verl.envs.environments.overcooked.interactive_play --layout forced_coordination --partner random

# Coordinates only (no grid)
python -m verl.envs.environments.overcooked.interactive_play --no-grid
```

## Known Issues / Gotchas

1. Grid is `[y, x]` indexed (row, column), but positions are `(x, y)` (column, row)
2. Movement action changes facing direction even if blocked by wall
3. Interact fails silently if facing wrong direction or invalid target
4. Pot requires exactly 3 ingredients to start cooking
5. Must hold empty plate to pick up cooked soup

## Next Steps

- [ ] Experiment config for training
- [ ] Captioner prompts
- [ ] Cluster test run
- [ ] Multi-agent mode investigation
