# Environment Wrapper API Requirements

## Overview

All environment wrappers in `verl/envs/environments/` must follow a standardized API to ensure compatibility with VecEnv, captioners, and the training pipeline.

## Required Properties

These must be **properties** (using `@property` decorator) or **instance attributes**, NOT methods:

| Property | Type | Description |
|----------|------|-------------|
| `language_action_space` | `list[str]` | List of valid text actions |
| `default_action` | `str` | Fallback action when parsing fails |
| `actions` | `list[str]` | Alias for `language_action_space` |
| `max_steps` | `int` | Maximum steps per episode |

**Critical**: Using `@classmethod` or bare `def` (without `@property`) causes `env.language_action_space` to return a bound method object instead of the list, breaking `random.choice()` and `in` checks.

```python
# WRONG - returns method object
@classmethod
def language_action_space(cls):
    return list(ACTIONS.keys())

# WRONG - returns method object
def language_action_space(self):
    return list(ACTIONS.keys())

# CORRECT - returns list
@property
def language_action_space(self):
    return list(ACTIONS.keys())

# ALSO CORRECT - instance attribute
def __init__(self, ...):
    self.language_action_space = list(ACTIONS.keys())
```

## Required Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `step` | `(action: str, is_valid: bool) -> tuple` | Execute action, return (obs, reward, terminated, truncated, info) |
| `reset` | `(**kwargs) -> tuple` | Reset env, return (obs, info) |
| `extract_action` | `(action: str) -> tuple` | Parse LLM output, return (full, extracted, valid, is_valid, metrics) |
| `get_instruction_prompt` | `() -> str` | Return system prompt for LLM |
| `get_text_action` | `(action: int) -> str` | Convert action index to text |
| `restructure_obs` | `(obs) -> dict` | Convert raw obs to standard format |

## EnvWrapper Forwarding

`EnvWrapper` (in `env_wrapper.py`) wraps all environments and forwards these properties/methods to the inner env. When adding new required properties, also add forwarding in EnvWrapper:

```python
@property
def language_action_space(self):
    return self.env.language_action_space
```

## Observation Format

`restructure_obs()` must return:
```python
{
    'text': {
        'long_term_context': str,  # Persistent state description
        'short_term_context': str   # Recent events/changes
    },
    'state': Any,  # Raw state for debugging
    'image': Optional[np.ndarray]  # If VLM mode
}
```

## Metrics Convention

`extract_action()` returns metrics dict with keys prefixed by `behavior/`:
- `behavior/valid_action_ratio`: 1.0 if action parsed successfully, 0.0 otherwise
- `behavior/epsilon_explored`: 1.0 if epsilon-greedy triggered, 0.0 otherwise

## Testing New Environments

Before integrating a new environment:
1. Verify `type(env.language_action_space)` is `list`, not `method`
2. Verify `env.language_action_space[0]` returns a string action
3. Run `random.choice(env.language_action_space)` succeeds
