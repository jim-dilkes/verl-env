# Overcooked-v2 Settings & Mechanics (JaxMARL)

## Core Settings (from settings.py)

```python
DELIVERY_REWARD = 20              # Base reward for correct soup delivery
POT_COOK_TIME = 20                # Ticks to cook (configurable via pot_cook_time)
INDICATOR_ACTIVATION_TIME = 10    # Ticks the recipe indicator stays active
INDICATOR_ACTIVATION_COST = 5     # Reward penalty for pressing indicator button

SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 3,        # Per ingredient added to pot
    "POT_START_COOKING": 5,       # When pot starts cooking (if manual)
    "DISH_PICKUP": 5,             # Picking up cooked soup
    "PLATE_PICKUP": 3,            # Picking up plate when useful
}
```

## Ingredients System

**Fixed constraint:** Exactly 3 ingredients per recipe (hardcoded in MAX_INGREDIENTS=3).

**Available ingredients:** Layout-dependent (0-9 ingredient types possible)
- Ingredient 0: typically onion (legacy 'O' symbol)
- Ingredient 1: typically tomato
- Ingredient 2: typically lettuce
- Ingredients 3-9: custom layouts can define more

**Encoding:** Each ingredient uses 2 bits (0-3 count per ingredient type)
- Bit layout in dynamic objects: `[plate:1bit][cooked:1bit][ing0:2bits][ing1:2bits][ing2:2bits]`
- Example: 3 onions = `0b001100` (bits 2-3 = 3), 2 tomatoes + 1 onion = `0b100100`

## Layout Symbols

Layouts defined as ASCII grids:
- `W` - Wall
- `A` - Agent spawn position
- `X` - Goal/serving counter (delivery location)
- `B` - Bowl/plate pile
- `P` - Pot
- `R` - Recipe indicator (passive, always shows recipe)
- `L` - Button recipe indicator (activated via interact, costs -5 reward, shows recipe for 10 ticks)
- `0-9` - Ingredient pile (ingredient index)
- ` ` (space) - Empty floor
- `O` - Legacy onion pile (deprecated, now use `0`)

## Recipe System

**Recipe definition:** List of 3 ingredient indices, e.g., `[0,0,0]` = 3 onions, `[0,1,1]` = 1 onion + 2 tomatoes

**Recipe sampling:**
- `possible_recipes` in layout defines allowed recipes (defaults to all combinations of available ingredients)
- Sampled on reset (or on delivery if `sample_recipe_on_delivery=True`)
- Examples from layouts:
  - Single ingredient type → only one recipe (e.g., `[0,0,0]`)
  - Two ingredient types → 4 possible recipes: `[0,0,0]`, `[0,0,1]`, `[0,1,1]`, `[1,1,1]`

**Recipe indicators:**
- `RECIPE_INDICATOR` (R): Passive, always displays current recipe on grid
- `BUTTON_RECIPE_INDICATOR` (L): Interactive button
  - Press via interact (must have empty inventory and empty cell)
  - Costs -5 reward
  - Activates for 10 ticks (countdown in grid extra channel)
  - Useful when recipe changes or is unknown

## Initialization Parameters

```python
OvercookedV2(
    layout="cramped_room",                    # Layout name or Layout object
    max_steps=400,                            # Episode length
    observation_type=ObservationType.DEFAULT, # DEFAULT or FEATURIZED
    agent_view_size=None,                     # Partial observability (None = full grid)
    random_reset=False,                       # Randomize agent pos, inventory, pots on reset
    random_agent_positions=False,             # Randomize only agent positions
    start_cooking_interaction=False,          # True = manual pot start, False = auto-cook when 3 ingredients
    negative_rewards=False,                   # Penalize wrong deliveries/ingredient placements
    sample_recipe_on_delivery=False,          # Resample recipe after each delivery
    indicate_successful_delivery=False,       # Add delivery success to observation
    op_ingredient_permutations=None,          # Permute ingredient indices per agent (for diversity)
    initial_state_buffer=None,                # Pre-sampled states for reset
    force_path_planning=False,                # Enable path planning (auto-enabled for FEATURIZED obs)
)
```

## Gameplay Mechanics

**Basic loop (3-onion soup example):**
1. Face ingredient pile (0), interact → pick up onion
2. Face pot, interact → drop onion into pot (repeat 3x)
3. Wait 20 ticks for cooking (or interact if `start_cooking_interaction=True`)
4. Pick up plate from pile (B)
5. Face cooked pot with plate, interact → pick up soup
6. Face goal counter (X), interact → deliver (+20 reward if correct recipe)

**Pot states:**
- Idle: Empty or partially full (<3 ingredients), not cooking
- Cooking: 3 ingredients, timer > 0
- Cooked: 3 ingredients, timer = 0, has COOKED flag

**Cooking triggers:**
- Auto (default): Pot starts cooking immediately when 3rd ingredient added
- Manual (`start_cooking_interaction=True`): Must interact with full pot to start

**Interaction rules:**
- Pickup: Empty inventory + facing pile/cooked pot with plate
- Drop: Holding ingredient + facing idle pot (not full)
- Delivery: Holding cooked soup + facing goal counter
- Button activation: Empty inventory + facing button indicator with empty cell

**Shaped reward conditions:**
- Ingredient placement: Only rewards if ingredient is part of current recipe
- Plate pickup: Only rewards if #plates_held < #nonempty_pots AND no plates on counters
- Dish pickup: Only rewards if picked dish matches current recipe
- Negative rewards (if enabled): Wrong ingredients in pot, wrong soup delivered

## Layout Categories

**Standard (simple):**
- `cramped_room` (5x4): 1 pot, 2 onion piles, basic
- `cramped_room_v2`: Adds recipe indicator
- `two_rooms`: Separated rooms, coordination required
- `long_room`: Extended horizontal layout

**Asymmetric:**
- `asymm_advantages`: Unequal positioning
- `asymm_advantages_recipes_{center,left,right}`: Recipe indicator placement variants

**Coordination-heavy:**
- `coord_ring`: Ring layout
- `forced_coord`: Narrow passages
- `counter_circuit`: Counter circuit design
- `fun_coordination`, `more_fun_coordination`: Multiple ingredient types
- `fun_symmetries{,1,_plates}`: Symmetric layouts with multiple ingredients

**Multi-ingredient (2+ ingredient types):**
- Layouts with `0`, `1`, `2`, `3` symbols
- Enable complex recipes like `[0,1,2]` (one of each)
- Examples: `fun_coordination`, `grounded_coord_ring`, `demo_cook_wide`

**Test/Demo:**
- `demo_cook_{simple,wide}`: Demo scenarios
- `test_time_{simple,wide}`: Protocol formation testing
- `grounded_coord_{simple,ring}`: Grounded coordination experiments

## Partner Policies (via wrapper)

Set in `OvercookedGymWrapper(partner_policy=...)`:
- `"noop"`: Partner stays still (default)
- `"random"`: Random actions
- `"none"`: Solo mode - partner moved off-map to (-1, -1), excluded from observations

**Solo mode implementation (bugfix):**
- Partner agent moved to position (-1, -1) after reset
- Partner excluded from observations and render
- Partner no longer blocks their spawn tile
- Allows full freedom of movement in solo play

## Grid Encoding (3 channels per cell)

```
grid[y, x, 0] = StaticObject  # POT, WALL, INGREDIENT_PILE_BASE+idx, etc.
grid[y, x, 1] = DynamicObject # Ingredient/plate encoding (bit-packed)
grid[y, x, 2] = Extra info    # Timer (pot cooking, indicator countdown)
```

**Timer channel:**
- Pot: Cooking countdown (20 → 0)
- Button indicator: Activation countdown (10 → 0)
- Other: Unused (0)

## Configurable Settings Summary

| Setting | Location | Default | Notes |
|---------|----------|---------|-------|
| Layout | init param | cramped_room | 23 built-in layouts |
| Max steps | init param | 400 | Episode length |
| Pot cook time | wrapper kwarg | 20 | Ticks to cook |
| Shaped rewards | wrapper kwarg | True | Intermediate rewards |
| Partner policy | wrapper kwarg | noop | noop/random/none |
| Manual cook | init param | False | Require interact to start pot |
| Recipe sampling | init param | on_reset | Can enable on_delivery |
| Negative rewards | init param | False | Penalize mistakes |
| Ingredients per recipe | **FIXED** | **3** | Cannot change |

## Multi-Ingredient Recipes

Layouts with multiple ingredient types (e.g., `0`, `1`, `2`) generate recipe sets automatically:

**2 ingredients (0,1):** 4 recipes
- `[0,0,0]`, `[0,0,1]`, `[0,1,1]`, `[1,1,1]`

**3 ingredients (0,1,2):** 10 recipes
- All combinations of 3 items from {0,1,2} with replacement
- Examples: `[0,0,0]`, `[0,0,1]`, `[0,1,2]`, `[2,2,2]`, etc.

Custom `possible_recipes` can override to specific subset.

## Custom Layouts

Custom layouts defined in `verl/envs/environments/overcooked/custom_layouts.py`:

**cramped_room_mixed:** 2 onions + 1 tomato recipe
```
Layout: 5x4
Left pile: tomato (ingredient 1)
Right pile: onion (ingredient 0)
Recipe: [0, 0, 1] (2 onions + 1 tomato)
```

**Usage in training config:**
```yaml
envs.overcooked_kwargs.layout_name="cramped_room_mixed"
```

**Usage in evaluation:**
```bash
# Single config file - greedy + mixed recipe
evaluation=overcooked_evals_mixed

# Combined standard + multi-action - both cramped_room and mixed recipe
evaluation=overcooked_evals_combined_mixed
```

**Evaluation config files:**
- `verl/trainer/config/evaluation/overcooked_evals_mixed.yaml` - Replaces asymm_advantages with cramped_room_mixed
- `verl/trainer/config/evaluation/overcooked_evals_combined_mixed.yaml` - Combined single/multi-action with mixed recipe

**Test scripts:**
```bash
# Automated test
python -m verl.envs.environments.overcooked.test_custom_layout

# Interactive play
python -m verl.envs.environments.overcooked.interactive_play --layout cramped_room_mixed

# List all layouts (built-in + custom)
python -m verl.envs.environments.overcooked.interactive_play --list-layouts
```

**Creating new custom layouts:**
1. Add layout string to `custom_layouts.py`
2. Define `possible_recipes` (list of 3-ingredient lists)
3. Register in `CUSTOM_LAYOUTS` dict
4. Use via config: `envs.overcooked_kwargs.layout_name="your_layout_name"`

## Common Gotchas

1. **Recipe always 3 ingredients** - No 1-ingredient or 5-ingredient soups
2. **Direction + interact** - Must face target object
3. **Plate required** - Can't pick up cooked soup without plate
4. **Auto-cook** - Pot starts cooking automatically unless `start_cooking_interaction=True`
5. **Button cost** - Recipe indicator button costs -5 reward
6. **Shaped rewards disable-able** - Set `shaped_reward=False` in wrapper for sparse rewards only
