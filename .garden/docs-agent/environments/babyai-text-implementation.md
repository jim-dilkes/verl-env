# BabyAI Text Wrapper Implementation

## Architecture

```
make_babyai_env() → EnvWrapper(
    BabyAILLMAgentsWrapper(
        BabyAITextCleanLangWrapper(
            gym.make("BabyAI-GoToRedBall-v0")
        )
    )
)
```

Two-wrapper chain:
- **CleanLangWrapper**: Converts obs to text format, stores mission, handles action mapping
- **LLMAgentsWrapper**: Parses LLM output, applies rewards/penalties, provides interface

## Files

- `verl/envs/environments/babyai_text/llm_agents_wrapper.py` - main LLM interface
- `verl/envs/environments/babyai_text/clean_lang_wrapper.py` - obs-to-text conversion
- `verl/envs/environments/babyai_text/babyai_env.py` - factory function
- `verl/trainer/config/prompt/babyai.yaml` - standard config
- `verl/trainer/config/prompt/babyai_multi_action.yaml` - multi-action config

## Actions

```python
BABYAI_ACTION_SPACE = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]
```

Default action: `"go forward"`

Action descriptions (for prompts):
```python
ACTIONS = {
    "turn left": "Rotate 90 degrees counter-clockwise",
    "turn right": "Rotate 90 degrees clockwise",
    "go forward": "Move one tile in facing direction",
    "pick up": "Pick up object directly in front",
    "drop": "Drop held object in front",
    "toggle": "Interact with object in front (open doors, etc.)",
}
```

## Mission Retrieval

Fallback order in `get_instruction_prompt()`:
1. Explicit `mission` kwarg if provided
2. `self._last_obs.get("mission")` from cached obs
3. `self.env._mission` from CleanLangWrapper
4. Default: `"complete the task"`

## Multi-Action Reasoning

When `multi_action_reasoning=True`:
- Prompt asks model to reason about each action in `<action name="...">` tags
- Final decision goes in `<decision>` tag
- `extract_action_instance()` parses `<decision>` tag

When `multi_action_reasoning=False` (default):
- Model outputs `<action>go forward</action>`
- `extract_action()` parses `<action>` tag

## Config Parameters

```yaml
# verl/trainer/config/prompt/babyai.yaml
prompt:
  multi_action_reasoning: false
  instruction_prompt: null  # Uses default if null
```

Override instruction prompt entirely via config:
```yaml
prompt:
  instruction_prompt: "Custom prompt with {mission} placeholder"
```

## Observation Format

CleanLangWrapper provides:
```python
{
    "text": {
        "long_term_context": str,   # Grid description
        "short_term_context": str,  # Recent event
    },
    "mission": str,  # Task goal
    "state": Any,    # Raw obs
}
```

## Metrics

`extract_action_instance()` returns metrics dict:
- `behavior/valid_action_ratio`: 1.0 if valid, 0.0 otherwise
- `behavior/plan_length`: Count of planning words detected
- `behavior/backtrack_length`: Count of backtrack words detected

## Testing

Interactive play:
```bash
python -m verl.envs.environments.babyai_text.interactive_play
python -m verl.envs.environments.babyai_text.interactive_play --task BabyAI-UnlockPickup-v0
```

Contract tests:
```bash
pytest tests/envs/test_babyai_wrapper.py -v
```

## Common Issues

1. **Mission not found**: Ensure CleanLangWrapper is in chain before LLMAgentsWrapper
2. **Circular import**: ACTIONS dict lives in llm_agents_wrapper.py, not __init__.py
3. **Action normalization**: Common variants handled (turnleft→turn left, goforward→go forward)
