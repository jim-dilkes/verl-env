# Environment Wrapper Interface

## Architecture Overview

```
make_env() -> EnvWrapper(LLMAgentsWrapper(BaseEnv))
                    |
                    v
              VecEnv worker -> Captioner -> LLM prompt
```

## Required Interface for LLMAgentsWrapper

Every environment must implement an inner wrapper (e.g., `OvercookedLLMAgentsWrapper`)
that the `EnvWrapper` delegates to. The wrapper must implement:

### Required Properties

```python
@property
def language_action_space(self) -> List[str]:
    """List of valid action strings the LLM can output."""
    return ["up", "down", "left", "right", "stay", "interact"]

@property
def default_action(self) -> str:
    """Fallback action when parsing fails."""
    return "stay"

@property
def max_steps(self) -> int:
    """Maximum steps per episode."""
    return self.env.max_steps
```

### Required Methods

```python
def reset(self, **kwargs) -> Tuple[dict, dict]:
    """Reset environment.

    Returns:
        obs: Observation dict (see format below)
        info: Info dict (can be empty)
    """

def step(self, action: str, is_valid: bool = True) -> Tuple[dict, float, bool, bool, dict]:
    """Execute action.

    Args:
        action: Action string from language_action_space
        is_valid: Whether the LLM output parsed successfully

    Returns:
        obs: Observation dict
        reward: Float reward (apply format_penalty if not is_valid)
        terminated: Episode done due to goal/failure
        truncated: Episode done due to max_steps
        info: Info dict
    """

def extract_action(self, llm_output: str) -> Tuple[str, str, str, bool, dict]:
    """Parse LLM output to extract action.

    CRITICAL: Must return exactly 5 values!

    Args:
        llm_output: Raw text from LLM

    Returns:
        full_action: Original LLM output (for logging/captioner)
        extracted_action: Parsed action before validation (may be None)
        valid_action: Action to execute (default_action if invalid)
        is_valid: Whether extraction succeeded
        metrics: Dict with at least {"behavior/valid_action_ratio": float}
    """

def get_instruction_prompt(self, instructions=None, info=None) -> str:
    """Return system prompt for LLM."""

def get_text_action(self, action) -> str:
    """Convert action index to string if needed."""

def get_stats(self) -> dict:
    """Return any custom stats for logging."""
    return {}
```

### Observation Format

The `restructure_obs` method must return:

```python
{
    'text': {
        'long_term_context': str,  # Full state description (grid, positions, etc.)
        'short_term_context': str, # Last event/feedback (brief)
    },
    'state': Any,  # Raw observation for debugging
    # Optional:
    'image': np.ndarray,  # For VLM environments
    'mission': str,       # For BabyAI (task-specific instruction)
}
```

## VecEnv Integration

The `vec_env.py` worker calls:

```python
# On step:
full_action, extracted_action, executed_action, is_valid, metrics = env.extract_action(action)
env_obs, reward, terminated, truncated, info = env.step(executed_action, is_valid)
info["action_was_valid"] = is_valid  # Added by vec_env
info["executed_action_text"] = executed_action  # Added by vec_env

# On reset:
env_obs, info = env.reset(seed=seed)
```

## Factory Functions (environments/__init__.py)

```python
make_env(env_name, task, config, render_mode=None) -> EnvWrapper
get_action_extraction_fn(env_name) -> Callable  # For standalone action parsing
```

To add a new environment:
1. Create `verl/envs/environments/<env_name>/` directory
2. Implement base gym env in `<env_name>_env.py` or similar
3. Implement LLMAgentsWrapper in `base.py`
4. Create factory function `make_<env_name>_env()`
5. Register in `environments/__init__.py`:
   - Add to `make_env()` switch
   - Add to `env_wrapper._process_observation()` switch
   - Add to `env_wrapper.get_instruction_prompt()` switch

## Action Extraction Patterns

Standard XML pattern (recommended):
```python
@staticmethod
def extract_action_from_xml_tag(text: str, tag: str = "action") -> str:
    try:
        return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip().lower()
    except (IndexError, AttributeError):
        return None
```

## Optional Info Dict Keys

The evaluator looks for these optional keys in info:
- `score`: Cumulative episode score (if not present, evaluator skips score tracking)

## Common Bugs to Avoid

1. **Wrong extract_action signature**: Must return 5 values, not 4
2. **Missing action_was_valid**: Now handled by vec_env.py (not env wrapper)
3. **Missing get_stats()**: Required method, can return empty dict
4. **Wrong obs format**: Must have `text.long_term_context` and `text.short_term_context`
