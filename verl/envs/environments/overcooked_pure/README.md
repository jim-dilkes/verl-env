# Overcooked Pure Python Implementation

A JAX-free reimplementation of the Overcooked environment for RL training with LLM agents.

## Why This Exists

The original JaxMARL-based Overcooked implementation causes several issues:

1. **Spawn multiprocessing requirement**: JAX's threading model deadlocks with `fork()` on Linux
2. **CUDA context conflicts**: JAX pre-allocates GPU memory, conflicting with vLLM/PyTorch
3. **Heavy dependency**: Requires JaxMARL submodule with complex build
4. **Blocks PRIME-RL/verifiers porting**: These stacks expect pure Python environments

Since LLM inference is the bottleneck (not environment stepping), pure Python/NumPy is acceptable.

## Installation

No additional dependencies required beyond NumPy and Gymnasium. This module is self-contained.

```bash
# Already part of verl-env package
pip install -e .
```

## Usage

### Direct Usage

```python
from verl.envs.environments.overcooked_pure import (
    OvercookedGymWrapper,
    OvercookedLLMAgentsWrapper,
    ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION
)

# Create base environment
env = OvercookedGymWrapper(
    layout="cramped_room",
    max_steps=200,
    partner_policy="noop",  # or "random" or "none" (solo mode)
    shaped_reward=True,
    pot_cook_time=20,
)

# Reset and step
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(5)  # interact

# Get text rendering
print(env.render())
```

### With LLM Wrapper

```python
from verl.envs.environments.overcooked_pure import (
    OvercookedGymWrapper,
    OvercookedLLMAgentsWrapper,
)

base_env = OvercookedGymWrapper(layout="cramped_room")
env = OvercookedLLMAgentsWrapper(base_env)

obs, info = env.reset()
# obs["text"]["long_term_context"] contains the grid visualization
# obs["text"]["short_term_context"] contains the last event

# Step with text action
obs, reward, terminated, truncated, info = env.step("interact")
```

### Via Factory Function

```python
from verl.envs.environments.overcooked_pure import make_overcooked_env

# Assuming you have a config object
env = make_overcooked_env("overcooked", "default", config)
```

## Available Layouts

### Built-in (from JaxMARL)

- `cramped_room` (5×4): Standard test layout
- `asymm_advantages` (9×5): Asymmetric advantages
- `coord_ring` (5×5): Coordination ring
- `forced_coord` (5×5): Forced coordination
- `counter_circuit` (8×5): Counter circuit

### Custom

- `cramped_room_mixed`: Cramped room with mixed recipe (2 onion + 1 tomato)

## Actions

| Action | Index | Description |
|--------|-------|-------------|
| right  | 0     | Move right (and face right) |
| down   | 1     | Move down (and face down) |
| left   | 2     | Move left (and face left) |
| up     | 3     | Move up (and face up) |
| stay   | 4     | Stay in place |
| interact | 5   | Interact with object in front |

## Game Mechanics

### Cooking Process

1. Pick up ingredients from ingredient piles
2. Place 3 ingredients in a pot
3. Wait for cooking (default: 20 ticks)
4. Pick up a plate from the plate pile
5. Pick up cooked soup from pot (with plate)
6. Deliver to serving counter (+20 reward)

### Partner Policies

- `"noop"`: Partner stays in place (default)
- `"random"`: Partner takes random actions
- `"none"`: Solo mode - partner hidden and moved off-map

### Rewards

- Delivery: +20 (base reward)
- Shaped rewards (optional):
  - Pick up useful ingredient: +3
  - Add ingredient to pot: +3
  - Pick up cooked soup: +5
  - Pick up plate (when useful): +3

## Running Tests

```bash
# Run standalone tests (no JaxMARL needed)
python -m pytest verl/envs/environments/overcooked_pure/tests/test_standalone.py -v

# Run behavioral parity tests (requires JaxMARL)
python -m pytest verl/envs/environments/overcooked_pure/tests/test_behavioral_parity.py -v
```

## Behavioral Parity

This implementation maintains behavioral parity with JaxMARL's OvercookedV2:

- Same state transitions for identical action sequences
- Same rewards for same sequences
- Same text rendering output
- Same random behavior given same seed

See `tests/test_behavioral_parity.py` for detailed comparison tests.

## File Structure

```
overcooked_pure/
├── __init__.py           # Constants and enums
├── game_engine.py        # Core game logic (pure Python/NumPy)
├── layouts.py            # Layout parsing and built-in layouts
├── gym_wrapper.py        # Gymnasium wrapper
├── base.py               # LLM agent wrapper
├── overcooked_env.py     # Factory function
├── README.md             # This file
├── PLAN.md               # Implementation plan
└── tests/
    ├── __init__.py
    ├── test_standalone.py        # Tests without JaxMARL
    └── test_behavioral_parity.py # Parity tests with JaxMARL
```

## Differences from JaxMARL

- No JAX dependencies
- Uses NumPy arrays instead of JAX arrays
- Uses Python's `random` instead of JAX's PRNG
- Single-threaded (acceptable since LLM inference is the bottleneck)
