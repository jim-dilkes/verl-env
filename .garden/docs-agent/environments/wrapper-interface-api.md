# Environment Wrapper Interface

## Standard LLMAgentsWrapper Interface
All environments implement via `env_wrapper.py`:

```python
class EnvWrapper(gym.Wrapper):
    # Properties
    @property language_action_space  # List[str] of valid actions
    @property default_action         # Fallback when parsing fails
    @property max_steps              # Episode length

    # Core methods
    def reset(**kwargs) -> (obs, info)
    def step(action, is_valid=True) -> (obs, reward, terminated, truncated, info)

    # LLM interface
    def get_instruction_prompt(instructions=None, info=None) -> str
    def extract_action(llm_output) -> (full, valid, is_valid, metrics)
    def check_action_validity(action) -> bool
    def get_text_action(action) -> str
    def get_stats() -> dict
```

## Observation Format (all envs)
```python
{
    'text': {
        'long_term_context': "State + rules + available actions",
        'short_term_context': "Recent change or feedback"
    },
    'state': raw_observation  # For debugging
}
```

## Action Extraction Patterns
1. XML tags: `<action>left</action>` (FrozenLake, WebShop, BabyAI, Overcooked)
2. Keywords: `ACTION: go forward` (Crafter, CoT captioner)
3. Fuzzy matching against language_action_space

## Factory Functions (environments/__init__.py)
```python
make_env(env_name, task, config, render_mode=None) -> EnvWrapper
get_action_extraction_fn(env_name) -> Callable
```
