# Fix: BabyAI missing from get_action_extraction_fn

**Type:** fix
**Branch:** fix/babyai-entropy-extraction
**Created:** 2026-01-16
**Started:** 2026-01-16
**Completed:** —

## Problem
BabyAI is not registered in `get_action_extraction_fn()` in `verl/envs/environments/__init__.py`. This causes entropy probing in the evaluator to crash:
```
ValueError("Not accessible action extraction function for babyai")
```

## Scope
- [x] Add BabyAI case to `get_action_extraction_fn` (lines 69-109)
- [x] Handle both standard mode (`<action>` tag) and multi-action mode (`<decision>` tag)
- [x] Use `BabyAILLMAgentsWrapper.extract_action` for standard mode
- [x] Create multi-action extraction similar to FastSnake/Overcooked pattern

## Reference
BabyAI valid actions (from `llm_agents_wrapper.py`):
```python
BABYAI_ACTION_SPACE  # imported from clean_lang_wrapper
# Actions: "turn left", "turn right", "go forward", "pick up", "drop", "toggle"
```

Default action: `"go forward"`

## Code Location
`verl/envs/environments/__init__.py:69-109`

## Cleanup
- [x] Remove duplicate `ACTIONS` dict from `babyai_text/__init__.py` (already in `llm_agents_wrapper.py`)
- [x] Remove unused `get_instruction_prompt` function from `babyai_text/__init__.py` (lines 14-32)

## Working Notes
### 2026-01-16 - Feature Started
Branch: `babyai-review-fix` (existing), renaming to `fix/babyai-entropy-extraction`

**Implementation approach:**
- Add `elif env_name == "babyai":` block in `get_action_extraction_fn`
- Import `BabyAILLMAgentsWrapper` from `babyai_text.llm_agents_wrapper`
- Standard mode: return `BabyAILLMAgentsWrapper.extract_action`
- Multi-action mode: create inline function using `extract_decision_from_xml` + `BABYAI_ACTION_SPACE` validation

**BabyAI action space:** "turn left", "turn right", "go forward", "pick up", "drop", "toggle"
**Default action:** "go forward"

### 2026-01-16 - Context from Docs

**From environments/babyai-text-implementation.md:**
- Wrapper chain: `make_babyai_env()` → `EnvWrapper(BabyAILLMAgentsWrapper(BabyAITextCleanLangWrapper(...)))`
- Main LLM interface: `verl/envs/environments/babyai_text/llm_agents_wrapper.py`
- Multi-action mode: `extract_action_instance()` parses `<decision>` tag, standard uses `<action>` tag
- Action normalization: variants like "turnleft" → "turn left" handled
- Common issue: ACTIONS dict should live in llm_agents_wrapper.py, not __init__.py

**From environments/wrapper-interface-api.md:**
- `get_action_extraction_fn(env_name)` provides standalone action parsing for evaluator
- Multi-action mode uses `<decision>` tag after reasoning block
- Pattern for new envs: import wrapper class, return `extract_action` for standard mode

**From environments/wrapper-api-requirements.md:**
- `extract_action()` must return 5 values: (full, extracted, valid, is_valid, metrics)
- Metrics dict must include `behavior/valid_action_ratio`

### 2026-01-16 - Implementation Complete

**Files modified:**
- `verl/envs/environments/__init__.py` - Added BabyAI case to `get_action_extraction_fn` (lines 108-122)
- `verl/envs/environments/babyai_text/__init__.py` - Removed duplicate ACTIONS dict and unused get_instruction_prompt

**Implementation details:**
- Standard mode: returns `BabyAILLMAgentsWrapper.extract_action` classmethod directly
- Multi-action mode: inline function uses `extract_decision_from_xml` + `_normalize_action` for variant handling
- Action validation against `BABYAI_ACTION_SPACE`, default to "go forward"

**Wrapper review:** BabyAI wrapper is fully compliant with training/eval API. All required properties and methods implemented.

### 2026-01-16 - Refactor: Single Source of Truth

**Changes to `llm_agents_wrapper.py`:**
- Added class constants: `DEFAULT_ACTION = "go forward"`, `ACTION_SPACE = BABYAI_ACTION_SPACE`
- Renamed `_normalize_action` → `normalize_action` (public API for evaluator)
- Updated `default_action` property to use `self.DEFAULT_ACTION`
- Updated `extract_action` to use `cls.ACTION_SPACE` and `cls.DEFAULT_ACTION`

**Changes to `__init__.py`:**
- Removed separate `BABYAI_ACTION_SPACE` import
- Now uses `BabyAILLMAgentsWrapper.ACTION_SPACE` and `.DEFAULT_ACTION`
- Uses public `normalize_action()` instead of private `_normalize_action()`

**Benefit:** Evaluator and wrapper now share single source of truth for action space and default action.
