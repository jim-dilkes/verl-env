# Overcooked Pure Python Implementation Plan

## Why This Exists

The current JaxMARL-based Overcooked implementation causes:
- **Spawn multiprocessing requirement**: fork() deadlocks with JAX's threading model
- **JAX CUDA context conflicts**: JAX pre-allocates GPU memory, conflicts with vLLM/PyTorch
- **Heavy dependency**: Requires JaxMARL submodule with complex build
- **Blocks PRIME-RL/verifiers porting**: These stacks expect pure Python environments

Since LLM inference is the bottleneck (not env stepping), pure Python/NumPy is acceptable.

---

## Architecture Overview

```
overcooked_pure/
├── __init__.py           # Constants: ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION, DIRECTION_NAMES
├── game_engine.py        # Core game logic (pure Python/NumPy)
├── layouts.py            # Layout parsing and built-in layouts
├── gym_wrapper.py        # Gymnasium wrapper (replaces jaxmarl_wrapper.py)
├── base.py               # LLMAgentsWrapper (copy from current - no changes)
├── overcooked_env.py     # Factory function
├── README.md             # Documentation
├── PLAN.md               # This file
└── tests/
    ├── __init__.py
    ├── test_behavioral_parity.py   # Compare JaxMARL vs pure Python
    └── test_standalone.py          # Tests without JaxMARL dependency
```

---

## File Structure & Responsibilities

### `__init__.py`
Constants matching current implementation exactly:
```python
ACTIONS = {...}           # Action name -> description
ACTION_TO_IDX = {...}     # Action name -> int (0-5)
IDX_TO_ACTION = {...}     # int -> action name
DIRECTION_NAMES = {...}   # Direction int -> name string
```

### `game_engine.py`
Pure Python/NumPy game state and logic:

**Classes:**
- `Position`: (x, y) coordinate with move methods
- `Agent`: pos, direction, inventory
- `GameState`: agents, grid, time, recipe, terminal
- `OvercookedEngine`: The core game logic class

**Grid Encoding (3-channel NumPy array [height, width, 3]):**
- Channel 0: Static objects (int enum)
- Channel 1: Dynamic items (bit-packed int)
- Channel 2: Timer/extra info

**Static Object Types:**
```python
EMPTY = 0
WALL = 1
COUNTER = 2  # Note: In JaxMARL, WALL is used for counters
GOAL = 4     # Serving counter
POT = 5
RECIPE_INDICATOR = 6
BUTTON_RECIPE_INDICATOR = 7
PLATE_PILE = 9
INGREDIENT_PILE_BASE = 10  # +idx for each ingredient type
```

**Dynamic Item Encoding (bit-packed):**
```
bit 0: has plate (1 = plate)
bit 1: is cooked (1 = cooked)
bits 2-3: ingredient 0 count (0-3)
bits 4-5: ingredient 1 count (0-3)
bits 6-7: ingredient 2 count (0-3)
```

**Direction Encoding:**
```python
UP = 0, DOWN = 1, RIGHT = 2, LEFT = 3
```

**Action Encoding:**
```python
RIGHT = 0, DOWN = 1, LEFT = 2, UP = 3, STAY = 4, INTERACT = 5
```

### `layouts.py`
Layout representation and parsing:

**Layout Class:**
- `agent_positions`: List of (x, y) spawn points
- `static_objects`: NumPy array of static object types
- `height`, `width`: Grid dimensions
- `num_ingredients`: Number of ingredient types
- `possible_recipes`: List of valid recipes [[0,0,0], [0,0,1], ...]

**Layout String Format:**
```
W = wall
A = agent spawn
0-9 = ingredient pile (index)
P = pot
X = serving counter (GOAL)
B = plate (bowl) pile
R = recipe indicator
L = button recipe indicator
' ' = empty floor
```

**Built-in Layouts:**
- cramped_room (5x4): Standard test layout
- asymm_advantages (9x5)
- coord_ring (5x5)
- forced_coord (5x5)
- counter_circuit (8x5)

**Custom Layouts:**
- cramped_room_mixed: 2 onion + 1 tomato recipe variant

### `gym_wrapper.py`
Gymnasium-compatible wrapper:

**OvercookedGymWrapper Class:**
- Same constructor signature as current jaxmarl_wrapper.py
- `reset(seed=None)` → (obs_array, info)
- `step(action)` → (obs_array, reward, terminated, truncated, info)
- `render()` → text string (ASCII grid + coordinates)
- `get_state_info()` → dict with agents, grid, time, etc.

**Partner Policies:**
- `"noop"`: Partner stays in place (action=4)
- `"random"`: Partner takes random action
- `"none"`: Solo mode - partner moved off-map

**Rendering:**
Must produce **identical** text output to JaxMARL version for same state.

### `base.py`
Copy of current `base.py` (OvercookedLLMAgentsWrapper) - no JAX dependencies.

### `overcooked_env.py`
Factory function `make_overcooked_env()`:
- Same signature as current
- Imports from pure modules instead of JaxMARL

---

## Core Game Mechanics

### Movement
- Actions 0-3 (right/down/left/up) change **both** position **and** facing direction
- Action 4 (stay) does nothing
- Movement blocked by: walls, counters, other agents, out-of-bounds
- Direction changes even if movement is blocked

### Collision Resolution
1. Calculate intended new positions for all agents
2. Resolve collisions: if two agents would occupy same cell, both stay in original positions
3. Prevent swapping: if agent A moves to B's position and B moves to A's position, both stay

### Interact Action (action=5)
The most complex action - depends on what agent faces and holds:

1. **Facing ingredient pile + hands empty** → Pick up ingredient
2. **Facing pot + holding ingredient + pot not full** → Add ingredient to pot
3. **Facing pot + pot is cooked + holding empty plate** → Pick up soup on plate
4. **Facing plate pile + hands empty** → Pick up empty plate
5. **Facing serving counter + holding completed soup** → Deliver (+20 reward)
6. **Facing counter + holding item** → Place item on counter
7. **Facing counter + counter has item + hands empty** → Pick up from counter

### Cooking
- Pot can hold up to 3 ingredients
- When 3 ingredients added, timer starts automatically (no interaction needed)
- Timer decrements each step
- When timer reaches 0, soup becomes "cooked"
- Default cook time: 20 ticks

### Rewards
**Base reward:**
- Delivery of correct recipe: +20

**Shaped rewards (when enabled):**
- Pick up ingredient from pile: +3 (if useful)
- Add ingredient to pot: +3 (if matches recipe)
- Pick up cooked soup: +5 (if matches recipe)
- Pick up plate: +3 (if there's a cooking/ready pot)
- Start cooking: +5 (if matches recipe, only with start_cooking_interaction=True)

### Recipe Matching
- Recipe is stored as bit-packed integer (same encoding as ingredients)
- A soup matches if its ingredients match the current recipe
- Recipe is sampled on reset from `possible_recipes`

---

## Behavioral Parity Test Plan

### Test Philosophy
The pure Python version must be **behaviorally identical** to JaxMARL. This means:
- Same state transitions for same action sequences
- Same rewards for same sequences
- Same text rendering for same states
- Same random behavior given same seed

### Test Categories

#### 1. Deterministic Sequence Tests
```python
def test_deterministic_sequence():
    """Fixed seed, play N actions, compare step-by-step."""
    actions = [0, 0, 5, 3, 3, 5, ...]  # Right, Right, Interact, ...
    
    jax_env = JaxMARLWrapper(layout="cramped_room", seed=42)
    pure_env = PureWrapper(layout="cramped_room", seed=42)
    
    jax_obs, _ = jax_env.reset()
    pure_obs, _ = pure_env.reset()
    
    for action in actions:
        jax_obs, jax_r, _, _, _ = jax_env.step(action)
        pure_obs, pure_r, _, _, _ = pure_env.step(action)
        
        # Compare agent positions
        assert jax_info['agents'] == pure_info['agents']
        
        # Compare grid state
        assert np.allclose(jax_info['grid'], pure_info['grid'])
        
        # Compare rewards
        assert abs(jax_r - pure_r) < 1e-6
        
        # Compare render output (exact string match)
        assert jax_env.render() == pure_env.render()
```

#### 2. Full Episode Test
```python
def test_complete_cooking_sequence():
    """Play through a complete soup delivery."""
    # Sequence: pick 3 onions, place in pot, wait 20 ticks, get plate, get soup, deliver
    # Compare final reward
```

#### 3. Movement Blocking Tests
```python
def test_wall_blocking():
    """Agent can't walk through walls."""

def test_agent_collision():
    """Two agents can't occupy same cell."""

def test_swap_prevention():
    """Agents can't swap positions."""

def test_out_of_bounds():
    """Movement clipped to grid bounds."""
```

#### 4. Interact Mechanics Tests
```python
def test_pickup_from_pile():
    """Pick up ingredient when hands empty."""

def test_place_in_pot():
    """Add ingredient to pot."""

def test_pot_cooking():
    """Pot starts cooking at 3 ingredients, timer decrements."""

def test_pickup_cooked_soup():
    """Pick up soup with plate when cooked."""

def test_delivery():
    """Deliver soup to serving counter."""

def test_counter_place_pickup():
    """Place and pick up from counters."""

def test_interact_empty_space():
    """Interact with empty floor does nothing."""

def test_interact_when_holding():
    """Can't pick up when already holding."""
```

#### 5. Shaped Reward Tests
```python
def test_shaped_reward_ingredient_pickup():
    """Shaped reward for picking useful ingredient."""

def test_shaped_reward_pot_placement():
    """Shaped reward for adding to pot."""

def test_shaped_reward_soup_pickup():
    """Shaped reward for picking up cooked soup."""

def test_shaped_reward_plate_pickup():
    """Shaped reward for plate when pot has stuff."""
```

#### 6. Solo Mode Tests
```python
def test_solo_mode_partner_offmap():
    """Partner at (-1, -1) in solo mode."""

def test_solo_mode_no_collision():
    """No collision with off-map partner."""

def test_solo_mode_render():
    """Solo mode render excludes partner."""
```

#### 7. Random Agent Positions Tests
```python
def test_random_positions_seed():
    """Same seed → same random positions."""

def test_random_positions_valid():
    """Random positions on valid floor tiles."""
```

#### 8. Edge Cases
```python
def test_interact_blocked():
    """Interact when facing out of bounds."""

def test_double_place_pot():
    """Can't add to full pot."""

def test_pickup_empty_pot():
    """Can't pick up from empty pot."""

def test_deliver_wrong_recipe():
    """Deliver wrong recipe (if negative_rewards=False, reward=0)."""

def test_max_steps_termination():
    """Episode ends at max_steps."""
```

### Test Infrastructure

```python
class ParityTestBase:
    """Base class with setup for both environments."""
    
    @classmethod
    def setUpClass(cls):
        # Check if JaxMARL is available
        cls.jaxmarl_available = check_jaxmarl()
    
    def get_both_envs(self, **kwargs):
        """Create matched JaxMARL and Pure Python envs."""
        pure_env = PureWrapper(**kwargs)
        if self.jaxmarl_available:
            jax_env = JaxMARLWrapper(**kwargs)
            return jax_env, pure_env
        return None, pure_env
```

---

## Implementation Order

1. **`__init__.py`** - Constants (5 min)
2. **`layouts.py`** - Layout parsing (30 min)
3. **`game_engine.py`** - Core logic (2-3 hours)
   - Position, Agent, GameState classes
   - Movement with collision resolution
   - Interact logic
   - Cooking timer
   - Rewards
4. **`gym_wrapper.py`** - Wrapper (1 hour)
   - State management
   - Partner policies
   - Rendering (exact text match)
5. **`base.py`** - Copy from current (5 min)
6. **`overcooked_env.py`** - Factory (10 min)
7. **Tests** (1-2 hours)
   - `test_standalone.py` first (no JaxMARL needed)
   - `test_behavioral_parity.py` (requires JaxMARL)

---

## Critical Implementation Details

### Item Encoding
```python
def encode_item(plate=False, cooked=False, ingredients=None):
    """Encode item as integer."""
    val = 0
    if plate:
        val |= 1
    if cooked:
        val |= 2
    for idx, count in ingredients.items():
        val |= (count & 0x3) << (2 + 2 * idx)
    return val

def decode_item(val):
    """Decode item from integer."""
    if val == 0:
        return None
    plate = bool(val & 1)
    cooked = bool(val & 2)
    ingredients = {}
    for i in range(3):
        count = (val >> (2 + 2 * i)) & 0x3
        if count > 0:
            ingredients[i] = count
    return {"plate": plate, "cooked": cooked, "ingredients": ingredients}
```

### Direction Vectors
```python
DIR_TO_VEC = {
    0: (0, -1),  # UP
    1: (0, 1),   # DOWN
    2: (1, 0),   # RIGHT
    3: (-1, 0),  # LEFT
}
```

### Action to Direction Mapping
```python
ACTION_TO_DIRECTION = {
    0: 2,   # right → RIGHT
    1: 1,   # down → DOWN
    2: 3,   # left → LEFT
    3: 0,   # up → UP
    4: -1,  # stay → no direction change
    5: -1,  # interact → no direction change
}
```

### Recipe Encoding
A recipe `[0, 0, 1]` (2 onions + 1 tomato) encodes as:
- ingredient(0) = 4 (0b0100)
- ingredient(0) = 4 (0b0100)
- ingredient(1) = 16 (0b10000)
- Sum = 4 + 4 + 16 = 24

But actually, ingredients are counted, not summed:
- 2x onion = count=2 at bits 2-3 = 0b1000 = 8
- 1x tomato = count=1 at bits 4-5 = 0b010000 = 16
- Total = 8 + 16 = 24

Wait, looking at the code more carefully:
```python
def ingredient(idx):
    return DynamicObject.BASE_INGREDIENT << 2 * idx
```
BASE_INGREDIENT = 4 (1 << 2)

So:
- ingredient(0) = 4 << 0 = 4
- ingredient(1) = 4 << 2 = 16
- ingredient(2) = 4 << 4 = 64

A recipe like [0, 0, 0] (3 onions) = 4 + 4 + 4 = 12

Actually, looking at `get_recipe_encoding`:
```python
def get_recipe_encoding(recipe):
    ingredients = jax.vmap(DynamicObject.ingredient)(recipe)
    return jnp.sum(ingredients)
```

So it's adding the individual ingredient values. For [0, 0, 0]:
- 4 + 4 + 4 = 12

Let me check bit encoding:
- 12 = 0b1100
- bits 2-3 = (12 >> 2) & 3 = 3 ✓ (3 onions)
- bits 4-5 = (12 >> 4) & 3 = 0 ✓ (0 tomatoes)

For [0, 0, 1]:
- 4 + 4 + 16 = 24 = 0b11000
- bits 2-3 = (24 >> 2) & 3 = 2 ✓ (2 onions)
- bits 4-5 = (24 >> 4) & 3 = 1 ✓ (1 tomato)

Perfect, the encoding is consistent.

---

## Verification Strategy

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test complete episodes
3. **Parity tests**: Compare against JaxMARL step-by-step
4. **Render tests**: Exact string matching for text output
5. **Seed tests**: Same seed produces same results

---

## Risk Areas

1. **Collision resolution order**: JaxMARL uses JAX's scan; Python may process differently
2. **Floating point rewards**: Ensure identical reward calculations
3. **Random number generation**: Different RNG between JAX and NumPy - need careful seed handling
4. **Edge cases in interact**: Many branches, easy to miss one
5. **Timer mechanics**: Off-by-one errors possible

---

## Success Criteria

1. All parity tests pass when JaxMARL is available
2. All standalone tests pass without JaxMARL
3. Full episode produces identical total reward
4. Render output is character-for-character identical
5. No JAX imports anywhere in `overcooked_pure/`
