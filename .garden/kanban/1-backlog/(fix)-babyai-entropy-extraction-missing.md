# Fix: BabyAI missing from get_action_extraction_fn

**Type:** fix
**Priority:** medium

## Problem
BabyAI is not registered in `get_action_extraction_fn()` in `verl/envs/environments/__init__.py`. This causes entropy probing in the evaluator to crash:
```
ValueError("Not accessible action extraction function for babyai")
```

## Scope
- [ ] Add BabyAI case to `get_action_extraction_fn` (lines 69-109)
- [ ] Handle both standard mode (`<action>` tag) and multi-action mode (`<decision>` tag)
- [ ] Use `BabyAILLMAgentsWrapper.extract_action` for standard mode
- [ ] Create multi-action extraction similar to FastSnake/Overcooked pattern

## Reference
BabyAI valid actions (from `llm_agents_wrapper.py`):
```python
BABYAI_ACTION_SPACE  # imported from clean_lang_wrapper
# Actions: "turn left", "turn right", "go forward", "pick up", "drop", "toggle"
```

Default action: `"go forward"`

## Code Location
`verl/envs/environments/__init__.py:69-109`

## Cleanup (optional)
- Remove duplicate `ACTIONS` dict from `babyai_text/__init__.py` (already in `llm_agents_wrapper.py`)
- Remove unused `get_instruction_prompt` function from `babyai_text/__init__.py` (lines 14-32)
