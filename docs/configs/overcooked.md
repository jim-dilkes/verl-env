# Overcooked Environment Configuration

Overcooked is a cooperative cooking game environment based on [JaxMARL Overcooked V2](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked_v2).

## Basic Usage

```bash
envs.env_name=overcooked \
envs.overcooked_kwargs.layout_name=cramped_room \
```

## Config Options

All options are set via `envs.overcooked_kwargs.<option>`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `layout_name` | str | `"cramped_room"` | Kitchen layout |
| `horizon` | int | `200` | Max steps per episode |
| `partner_policy` | str | `"noop"` | Partner agent behavior |
| `shaped_reward` | bool | `True` | Include intermediate shaping rewards |
| `seed` | int | `0` | Random seed |
| `print_visualization` | bool | `True` | Include ASCII grid in observations |
| `print_coordinates` | bool | `True` | Include coordinate descriptions in observations |
| `pot_cook_time` | int | `20` | Cooking duration in ticks |

## Available Layouts

| Layout | Description |
|--------|-------------|
| `cramped_room` | Compact 5x4 kitchen, basic coordination |
| `asymmetric_advantages` | Unequal agent positioning |
| `coordination_ring` | Ring-shaped, requires synchronization |
| `forced_coordination` | Mandates cooperation to complete tasks |
| `counter_circuit` | Counter-based setup |

List all layouts:
```bash
python -m verl.envs.environments.overcooked.interactive_play --list-layouts
```

## Partner Policies

| Policy | Behavior |
|--------|----------|
| `"noop"` | Partner always stays in place (default) |
| `"random"` | Partner takes random actions |
| `"none"` | Solo mode - partner hidden from observations |

### Solo Mode Notes

When `partner_policy="none"`:
- Partner is hidden from text observations and ASCII render
- Instruction prompt adapts ("You are playing Overcooked solo...")
- **Caveat**: Partner agent still exists physically in the environment and may block movement at their start position

## Game Rules

### Actions (6 discrete)

| Action | Description |
|--------|-------------|
| `right` | Move/face right |
| `down` | Move/face down |
| `left` | Move/face left |
| `up` | Move/face up |
| `stay` | Wait in place |
| `interact` | Pick up, place, or use object in front |

### Cooking Process

1. Pick up ingredients from ingredient piles (interact while facing)
2. Place 3 ingredients in a pot (interact while facing pot)
3. Wait for cooking to complete (`pot_cook_time` ticks)
4. Pick up a dish from dish pile
5. Pick up cooked soup from pot (with dish in hand)
6. Deliver to serving counter (interact)

### Rewards

- **Delivery reward**: +20 for delivering a completed soup
- **Shaped rewards** (if enabled): Small rewards for intermediate progress (placing ingredients, picking up dishes, etc.)
- **Format penalty**: Configurable penalty for invalid LLM action format (default: 0.1)

## Example Configurations

### Solo Mode with Fast Cooking

```bash
envs.env_name=overcooked \
envs.overcooked_kwargs.layout_name=cramped_room \
envs.overcooked_kwargs.partner_policy=none \
envs.overcooked_kwargs.pot_cook_time=5 \
envs.overcooked_kwargs.horizon=100 \
```

### Cooperative with Random Partner

```bash
envs.env_name=overcooked \
envs.overcooked_kwargs.layout_name=forced_coordination \
envs.overcooked_kwargs.partner_policy=random \
envs.overcooked_kwargs.shaped_reward=True \
```

### Coordinates Only (No ASCII Grid)

```bash
envs.env_name=overcooked \
envs.overcooked_kwargs.layout_name=cramped_room \
envs.overcooked_kwargs.print_visualization=False \
envs.overcooked_kwargs.print_coordinates=True \
```

## Observation Format

The LLM receives text observations with two components:

### Long-term Context (State Description)

```
Layout: cramped_room (5x4)
Coordinates: (x, y) where x=column, y=row from top-left (0,0)

You: pos=(3, 1), facing UP, holding: NOTHING
Partner: pos=(1, 1), facing UP, holding: NOTHING

Pot at (2, 0): 2x onion (2/3 ingredients)

Serving counter: (3, 3)
Dish pile: (1, 3)
Ingredient piles: (0, 1, 'onion'), (4, 1, 'onion')

Kitchen (cramped_room, 5x4):
Legend: #=wall .=floor P=pot S=serve D=dish 0-9=ingredients @=you X=partner
# # P # #
0 X . @ 0
# . . . #
# D # S #

Step: 5/200
Last: Took action 'interact'
```

### Short-term Context (Last Event)

```
Took action 'interact' and received reward 3.0!
```

## Limitations

- **3 ingredients per recipe**: Hardcoded in JaxMARL, cannot be changed
- **Recipe types**: Determined by layout, cannot be customized at runtime
- **Solo mode blocking**: Hidden partner still physically occupies space

## Interactive Testing

```bash
# Basic play
python -m verl.envs.environments.overcooked.interactive_play

# Solo mode with fast cooking
python -m verl.envs.environments.overcooked.interactive_play --partner none --cook-time 5

# Different layout
python -m verl.envs.environments.overcooked.interactive_play --layout forced_coordination

# Controls: W/A/S/D = move, E = interact, Space = stay, Q = quit
```
