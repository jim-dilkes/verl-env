# Update BabyAI-Text Wrapper to Current Interface

**Type:** feat
**Branch:** feat/babyai-wrapper
**Created:** 2026-01-15 15:49
**Started:** 2026-01-16
**Completed:** —

## Goal
Bring `babyai_text/` wrapper up to parity with Overcooked's LLMAgentsWrapper pattern, including multi-action reasoning support.

## Scope
- [x] Add properties: `default_action`, `actions`, `max_steps` (explicit, not via __getattr__)
- [x] Store `language_action_space` locally in __init__ as `list[str]` (not delegation)
- [x] Add `_default_instruction_prompt()` with `{mission}` placeholder
- [x] Add `get_instruction_prompt(*, mission=None)` - kwarg-only, fetches from env if not provided
- [x] Support config-driven `instruction_prompt` override
- [x] Add `multi_action_reasoning` config flag + attr on wrapper
- [x] Add `extract_action_instance()` with multi-action support
- [x] Add `restructure_obs()` that validates required keys (not blind passthrough)
- [x] Add `get_stats()` delegation
- [x] Fix `get_text_action()` in clean_lang_wrapper.py to handle str|int|enum input
- [x] Update `babyai_env.py` factory to pass new config params
- [x] Update `env_wrapper.py` dispatch (pass mission as kwarg)
- [x] Register config params in `verl/trainer/config/prompt/babyai.yaml`
- [x] Add pytest `tests/envs/test_babyai_wrapper.py` that asserts contract

## Definition of Done
- [x] `env.language_action_space` returns `list[str]`, `random.choice()` works, `in` checks work
- [x] `extract_action()` returns exactly 5 values: `(full, extracted, valid, is_valid, metrics)`
- [x] `metrics` includes at least `behavior/valid_action_ratio`
- [x] `get_instruction_prompt(mission=None)` works when called both ways:
  - `env.get_instruction_prompt()` (fetches mission from env)
  - `env.get_instruction_prompt(mission="...")` (uses provided mission)
- [x] `restructure_obs()` output has `text.long_term_context` and `text.short_term_context` keys
- [x] Multi-action mode: `multi_action_reasoning=True` → parses `<decision>` tag
- [x] Standard mode: `multi_action_reasoning=False` → parses `<action>` tag
- [x] Pytest passes locally without GPU/cluster deps (37 tests pass)

## Out of Scope
- Task-specific instruction prompts
- Changes to `babaisai/` wrapper
- New BabyAI tasks or environments

## Key Decisions

### Action Space Representation
- `language_action_space`: `list[str]` (NOT dict) - e.g., `["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]`
- `actions`: alias property returning same `list[str]`
- Action descriptions kept in separate `ACTIONS` dict for prompt generation only

### Multi-Action Parsing Spec
Standard mode (`multi_action_reasoning=False`):
```
Model outputs: <action>go forward</action>
Parsed via: extract_action_from_xml_tag(text, "action")
```

Multi-action mode (`multi_action_reasoning=True`):
```
Model outputs:
<actions>
<action name="turn left">reasoning...</action>
<action name="turn right">reasoning...</action>
...
</actions>
<decision>go forward</decision>

Parsed via: extract_decision_from_xml(text) → extracts from <decision> tag
Intermediate <action> tags are for reasoning only, ignored for execution
```

### Mission Retrieval Contract
Fallback order (in `get_instruction_prompt`):
1. Explicit `mission` kwarg if provided
2. `self.env._obs.get("mission")` if available (from last reset/step)
3. `getattr(self.env, "_mission", None)` (CleanLangWrapper stores here)
4. Default: `"complete the task"`

### Instruction Prompt
- Template with `{mission}` placeholder, filled at runtime
- Config can override entire prompt via `instruction_prompt` kwarg
- Self-contained in wrapper class (not separate module function)

## Working Notes
<!-- Session handoff and working context goes here -->
### 2026-01-16 - Feature Started
Interview complete. Target: bring babyai_text wrapper to Overcooked-level interface.

Key files:
- `verl/envs/environments/babyai_text/llm_agents_wrapper.py` - main wrapper to update
- `verl/envs/environments/babyai_text/babyai_env.py` - factory function
- `verl/envs/environments/env_wrapper.py` - instruction prompt dispatch

Reference: `verl/envs/environments/overcooked/base.py` (target pattern)

### 2026-01-16 - Context from Docs

**From wrapper-api-requirements.md:**
- Required properties: `language_action_space` (list), `default_action` (str), `actions` (list), `max_steps` (int)
- CRITICAL: Must use `@property` decorator or instance attribute, NOT method
- `extract_action()` returns 5 values: (full, extracted, valid, is_valid, metrics)
- Metrics must include `behavior/valid_action_ratio`

**From wrapper-interface-api.md:**
- Obs format: `{text: {long_term_context, short_term_context}, state}`
- VecEnv calls `extract_action()` then `step(executed_action, is_valid)`
- Required methods: `reset`, `step`, `extract_action`, `get_instruction_prompt`, `get_text_action`, `get_stats`
- To add env: update `env_wrapper._process_observation()` and `get_instruction_prompt()` switches

**From overcooked-jaxmarl-implementation.md (reference pattern):**
- Multi-action uses `<decision>` tag; standard uses `<action>` tag
- `extract_action_instance()` method for training (uses instance config)
- `extract_action()` classmethod for evaluator (no multi-action)
- Epsilon-greedy handled centrally in vec_env.py

### 2026-01-16 - Plan Review Findings

**Architecture insight:** BabyAI has TWO wrappers in chain:
- `CleanLangWrapper` - converts obs to text, handles action mapping
- `LLMAgentsWrapper` - handles LLM output parsing, rewards

Key implications:
- `restructure_obs()` is passthrough (CleanLangWrapper already formats)
- `language_action_space` exists via `__getattr__` delegation - store locally for explicitness
- Mission stored in `CleanLangWrapper._mission` - access via `self.env._mission`

**Instruction prompt strategy:**
- `_default_instruction_prompt()` returns template with `{mission}` placeholder
- `get_instruction_prompt(mission=None)` fills placeholder from env if not provided
- Config can override entire prompt via `instruction_prompt` kwarg

**Backward compat:**
- `get_instruction_prompt(*, mission=None)` - kwarg-only to avoid positional conflicts
- env_wrapper passes `mission=instructions` - compatible with new signature

### 2026-01-16 - Feedback Review Verification

**Verified against code:**
- CleanLangWrapper obs format (lines 53, 63): ✓ provides `text.long_term_context` + `text.short_term_context`
- Mission access: ✓ in `obs["mission"]` (line 48) AND `_mission` attr - use obs as primary
- language_action_space (line 17): ✓ `BABYAI_ACTION_SPACE[:]` is copied list, invariant after init
- get_text_action (line 34): ✗ `action.value` expects enum - needs fix for str/int
- Invalid action penalty: ✓ already in LLMAgentsWrapper.step() with format_penalty
- Vec_env multi-action (lines 347-358): ✓ checks `multi_action_reasoning` attr → requires `extract_action_instance()`

**Wrapper chain confirmed:**
```
make_babyai_env() → EnvWrapper(
    BabyAILLMAgentsWrapper(
        BabyAITextCleanLangWrapper(
            gym.make(task)
        )
    )
)
```

CleanLangWrapper responsibilities:
- Converts obs to `text.long_term_context` + `text.short_term_context`
- Stores mission in `_mission` and `obs["mission"]`
- Provides `language_action_space`, `default_action`, `max_steps`
- Handles `step()` action string → int conversion

LLMAgentsWrapper responsibilities:
- Parses LLM output via `extract_action()`
- Applies `format_penalty` for invalid actions
- Applies `binary_reward` transformation

### 2026-01-16 - Implementation Complete

**Commits:**
1. `fix: get_text_action handles str/int/enum input` (84a25050)
2. `feat: refactor BabyAI LLMAgentsWrapper to current interface` (295ee4dc)
3. `feat: pass multi_action_reasoning to BabyAI wrapper` (b1282df4)
4. `refactor: use self-contained get_instruction_prompt for babyai` (d6beb2bc)
5. `chore: register babyai prompt config params` (bf213a26)
6. `test: add BabyAI wrapper contract tests` (a82e2494)

**Tests:** 37 tests pass in `tests/envs/test_babyai_wrapper.py`

**Files modified:**
- `verl/envs/environments/babyai_text/clean_lang_wrapper.py` - get_text_action fix
- `verl/envs/environments/babyai_text/llm_agents_wrapper.py` - complete refactor
- `verl/envs/environments/babyai_text/babyai_env.py` - factory config passing
- `verl/envs/environments/env_wrapper.py` - dispatch simplification
- `verl/trainer/config/prompt/babyai.yaml` - new config
- `verl/trainer/config/prompt/babyai_multi_action.yaml` - new config
- `tests/envs/test_babyai_wrapper.py` - new tests

**Note:** Moved `ACTIONS` dict into `llm_agents_wrapper.py` to avoid circular import with `__init__.py`.

## Implementation Plan

### Phase 1: Fix CleanLangWrapper (no breaking changes)
**File:** `verl/envs/environments/babyai_text/clean_lang_wrapper.py`

1. Fix `get_text_action()` to handle multiple input types:
```python
def get_text_action(self, action):
    if isinstance(action, str):
        return action
    if isinstance(action, int):
        return self.language_action_space[action]
    return self.language_action_space[action.value]  # enum
```

**Commit:** `fix: get_text_action handles str/int/enum input`

---

### Phase 2: Refactor LLMAgentsWrapper (main work)
**File:** `verl/envs/environments/babyai_text/llm_agents_wrapper.py`

2a. Update `__init__` - store local state:
```python
def __init__(self, env, vlm=False, **kwargs):
    super().__init__(env)
    self.format_penalty = kwargs.get("format_penalty", 0.0)
    self.binary_reward = kwargs.get("binary_reward", False)
    self.multi_action_reasoning = kwargs.get("multi_action_reasoning", False)

    # Store locally (not via __getattr__ delegation)
    self.language_action_space = list(env.language_action_space)
    self._last_obs = None  # For mission retrieval

    # Instruction prompt: config override or default
    self.instruction_prompt = kwargs.get("instruction_prompt", None)
    if self.instruction_prompt is None:
        self.instruction_prompt = self._default_instruction_prompt()
```

2b. Add required properties:
```python
@property
def default_action(self):
    return "go forward"

@property
def actions(self):
    return self.language_action_space

@property
def max_steps(self):
    return getattr(self.env, "max_steps", 100)
```

2c. Add `_default_instruction_prompt()`:
```python
def _default_instruction_prompt(self):
    from verl.envs.environments.babyai_text import ACTIONS
    action_strings = ",\n".join(
        f'"{action}": {description}' for action, description in ACTIONS.items()
    )

    if self.multi_action_reasoning:
        return f"""[Instructions]
You are an agent playing a navigation game. Your maximum response length: 300 words.

[Available Actions]
{action_strings}

[Response Format]
Reason about EACH action, then make a decision.

<actions>
<action name="turn left">Your reasoning...</action>
<action name="turn right">Your reasoning...</action>
<action name="go forward">Your reasoning...</action>
<action name="pick up">Your reasoning...</action>
<action name="drop">Your reasoning...</action>
<action name="toggle">Your reasoning...</action>
</actions>

<decision>your_chosen_action</decision>

[Rules]
- Your goal is to: {{mission}}
- You cannot "go forward" if blocked by wall/object
- Use 'toggle' to interact with objects in front of you"""
    else:
        return f"""[Instructions]
You are an agent playing a navigation game. Your maximum response length: 200 words.

[Available Actions]
{action_strings}

[Rules]
- Your goal is to: {{mission}}
- You cannot "go forward" if blocked by wall/object
- Use 'toggle' to interact with objects in front of you

Output your action in: <action>your_action</action>"""
```

2d. Add `get_instruction_prompt()`:
```python
def get_instruction_prompt(self, *, mission=None):
    if mission is None:
        # Fallback chain
        if self._last_obs is not None and "mission" in self._last_obs:
            mission = self._last_obs["mission"]
        else:
            mission = getattr(self.env, "_mission", None)
        if mission is None:
            mission = "complete the task"

    return self.instruction_prompt.format(mission=mission)
```

2e. Add `restructure_obs()` with validation:
```python
def restructure_obs(self, obs):
    # Validate required keys (CleanLangWrapper provides these)
    if "text" not in obs:
        raise ValueError("Obs missing 'text' key - check wrapper chain")
    text = obs["text"]
    if "long_term_context" not in text or "short_term_context" not in text:
        raise ValueError("Obs['text'] missing required keys")

    self._last_obs = obs  # Cache for mission retrieval
    return obs
```

2f. Override `reset()` and `step()` to use `restructure_obs`:
```python
def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    obs = self.restructure_obs(obs)
    return obs, info

def step(self, action, is_valid=True):
    obs, reward, terminated, truncated, info = self.env.step(action)
    if not is_valid:
        reward = -self.format_penalty
    if self.binary_reward:
        reward = 1.0 if reward > 0 else reward
    obs = self.restructure_obs(obs)
    return obs, float(reward), terminated, truncated, info
```

2g. Add `extract_decision_from_xml()` static method:
```python
@staticmethod
def extract_decision_from_xml(text: str) -> str:
    try:
        return text.split("<decision>")[1].split("</decision>")[0].strip().lower()
    except (IndexError, AttributeError):
        return None
```

2h. Convert `extract_action()` to `@classmethod` (evaluator compat):
```python
@classmethod
def extract_action(cls, action):
    """Parse LLM output (classmethod for evaluator).
    Standard mode only - no multi-action support.
    """
    full_action = str(action)
    extracted = cls.extract_action_from_xml_tag(full_action)

    # Normalize action string
    if extracted:
        extracted = extracted.lower().replace("_", " ")
        # Handle common variants
        variants = {"turnleft": "turn left", "turnright": "turn right",
                   "goforward": "go forward", "pickup": "pick up"}
        extracted = variants.get(extracted, extracted)

    from verl.envs.environments.babyai_text.clean_lang_wrapper import BABYAI_ACTION_SPACE
    is_valid = extracted in BABYAI_ACTION_SPACE
    valid_action = extracted if is_valid else "go forward"

    metrics = {"behavior/valid_action_ratio": 1.0 if is_valid else 0.0}
    return full_action, extracted, valid_action, is_valid, metrics
```

2i. Add `extract_action_instance()`:
```python
def extract_action_instance(self, action):
    """Parse LLM output (instance method for training).
    Supports multi-action reasoning mode.
    """
    full_action = str(action)

    if self.multi_action_reasoning:
        extracted = self.extract_decision_from_xml(full_action)
    else:
        extracted = self.extract_action_from_xml_tag(full_action)

    # Normalize
    if extracted:
        extracted = extracted.lower().replace("_", " ")
        variants = {"turnleft": "turn left", "turnright": "turn right",
                   "goforward": "go forward", "pickup": "pick up"}
        extracted = variants.get(extracted, extracted)

    is_valid = extracted in self.language_action_space
    valid_action = extracted if is_valid else self.default_action

    metrics = {"behavior/valid_action_ratio": 1.0 if is_valid else 0.0}
    return full_action, extracted, valid_action, is_valid, metrics
```

2j. Add `get_stats()`:
```python
def get_stats(self):
    inner_stats = self.env.get_stats() if hasattr(self.env, "get_stats") else {}
    return inner_stats
```

2k. Add `get_text_action()` forwarding:
```python
def get_text_action(self, action):
    return self.env.get_text_action(action)
```

2l. Remove `__getattr__` method (explicit > implicit).

**Commit:** `feat: refactor BabyAI LLMAgentsWrapper to current interface`

---

### Phase 3: Update factory function
**File:** `verl/envs/environments/babyai_text/babyai_env.py`

3. Pass new config params:
```python
def make_babyai_env(env_name, task, config, render_mode: Optional[str] = None):
    # ... existing env creation ...

    env_kwargs = dict(config.envs) if hasattr(config, "envs") else {}

    # Check for multi-action reasoning mode
    multi_action_reasoning = False
    if hasattr(config, "prompt") and hasattr(config.prompt, "prompt"):
        multi_action_reasoning = getattr(config.prompt.prompt, "multi_action_reasoning", False)
    env_kwargs["multi_action_reasoning"] = multi_action_reasoning

    # Config instruction override
    if hasattr(config, "prompt") and hasattr(config.prompt, "prompt"):
        instruction_prompt = getattr(config.prompt.prompt, "instruction_prompt", None)
        if instruction_prompt is not None:
            env_kwargs["instruction_prompt"] = instruction_prompt

    env = BabyAILLMAgentsWrapper(env, **env_kwargs)
    return env
```

**Commit:** `feat: pass multi_action_reasoning to BabyAI wrapper`

---

### Phase 4: Update env_wrapper dispatch
**File:** `verl/envs/environments/env_wrapper.py`

4. Change babyai instruction prompt dispatch:
```python
elif self.env_name == "babyai":
    return self.env.get_instruction_prompt(mission=instructions)
```

**Commit:** `refactor: use self-contained get_instruction_prompt for babyai`

---

### Phase 5: Register config params
**File:** `verl/trainer/config/prompt/babyai.yaml` (create if needed)

5. Add new params:
```yaml
prompt:
  multi_action_reasoning: false
  instruction_prompt: null  # Use default if null
```

**Commit:** `chore: register babyai prompt config params`

---

### Phase 6: Add pytest
**File:** `tests/envs/test_babyai_wrapper.py` (new)

6. Test contract:
```python
import random
import pytest

def test_language_action_space_is_list():
    env = make_test_env()
    assert isinstance(env.language_action_space, list)
    assert all(isinstance(a, str) for a in env.language_action_space)
    random.choice(env.language_action_space)  # Should not raise
    assert "go forward" in env.language_action_space

def test_extract_action_returns_5_values():
    env = make_test_env()
    result = env.extract_action("<action>go forward</action>")
    assert len(result) == 5
    full, extracted, valid, is_valid, metrics = result
    assert "behavior/valid_action_ratio" in metrics

def test_extract_action_instance_multi_action():
    env = make_test_env(multi_action_reasoning=True)
    result = env.extract_action_instance("<decision>turn left</decision>")
    _, _, valid, is_valid, _ = result
    assert is_valid
    assert valid == "turn left"

def test_get_instruction_prompt_both_signatures():
    env = make_test_env()
    env.reset()
    prompt1 = env.get_instruction_prompt()
    prompt2 = env.get_instruction_prompt(mission="test mission")
    assert "test mission" in prompt2

def test_restructure_obs_validates_keys():
    env = make_test_env()
    obs, _ = env.reset()
    assert "text" in obs
    assert "long_term_context" in obs["text"]
    assert "short_term_context" in obs["text"]

def test_step_reset_cycle():
    env = make_test_env()
    obs, info = env.reset()
    action = random.choice(env.language_action_space)
    obs, reward, term, trunc, info = env.step(action, is_valid=True)
    assert isinstance(reward, float)
```

**Commit:** `test: add BabyAI wrapper contract tests`

---

### Execution Order
1. Phase 1 (fix get_text_action) - no deps
2. Phase 2 (main refactor) - depends on Phase 1
3. Phase 3 (factory) - depends on Phase 2
4. Phase 4 (env_wrapper) - depends on Phase 2
5. Phase 5 (config) - no deps, can parallel with 3-4
6. Phase 6 (tests) - depends on all above

### Verification
After each phase:
- Run existing tests (if any)
- After Phase 6: `pytest tests/envs/test_babyai_wrapper.py -v`
- Final: manual test with `python -c "from verl.envs.environments.babyai_text import make_babyai_env; ..."`

## Original Notes
(feat) - set up BabyAI for current env interface
