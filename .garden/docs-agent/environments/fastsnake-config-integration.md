# FastSnake Environment

## Location
`verl/envs/environments/FastSnake/` (cloned from github.com/jim-dilkes/FastSnake)

## Core Files
- `src/env.py` - FastSnakeEnv gym environment
- `src/core.py` - FastSnake game logic

## Configuration Options
```python
FastSnakeEnv(
    width=10, height=10,
    max_rounds=100,
    num_external_snakes=1,      # Agent-controlled
    num_random_snakes=1,        # NPC with random policy
    death_reward=-2,
    step_reward=-0.01,
    num_apples=5, apple_reward=1,
    num_bananas=0, banana_reward=10,
    num_fires=0, fire_reward=-1,
    hill_direction=None,        # 'up'|'down'|'left'|'right' for rolling apples
    destroy_at_bottom=False,
    include_absent_objects=None,
    print_visualization=True,
    print_coordinates=True,
    print_axes=False
)
```

## Actions
4 discrete: up (0), down (1), left (2), right (3)

## Observation
- Channel-based tensor: head, bodies, apples, bananas, fires
- Text rendering via `game_state_text()`:
  - Board size + coordinate system
  - Object positions (apples, bananas, fires)
  - Snake positions (yours + enemies)
  - ASCII visualization

## Seeding
- Separate RNGs for: snake actions, apple placement, banana placement, fire placement
- Deterministic when seed provided

## Integration Status
- Needs LLMAgentsWrapper integration (like other envs)
- Wire up to make_env() in __init__.py
