# Fix: Eval Config Prompt Override Ignored

**Type:** fix
**Branch:** fix/wrong-eval-prompt-ma
**Created:** 2026-01-16
**Started:** 2026-01-16
**Completed:** 2026-01-16

## Goal
Ensure evaluation configs use their own `instruction_prompt` and `multi_action_reasoning` fields instead of inheriting from the training config.

## Scope
- [x] Find where evaluator gets prompt config (multi_env_evaluator.py:283)
- [x] Fix prompt inheritance to use eval config's fields when specified
- [x] Standardize BabyAI to use `environment_instruction` like other envs
- [x] Re-enable BabyAI in make_env
- [ ] Verify on cluster (pending)

## Out of Scope
- Changing other eval config inheritance behavior
- Refactoring captioner/generation param handling

## Key Decisions
- Explicit override only: eval config's `instruction_prompt` and `multi_action_reasoning` should be used when present, regardless of training config
- This is a bug fix, not a behavioral change - the eval config already has these fields specified

## Working Notes
### 2026-01-16 - Feature Started
**Problem:** When running `snake_evals_combined.yaml`, the evaluator uses training prompt even though eval config specifies different prompts per eval scenario.

**Example:** During multi-action training, `Snake-20Step-Greedy` (single-action eval) should use the SA prompt specified in the eval config, but instead uses the MA training prompt.

### 2026-01-16 - Root Cause Found
**Root cause:** Config path mismatch between evaluator and env factory.

In `multi_env_evaluator.py:_create_env_config()`:
- Line 283: Sets `temp_config.envs.instruction_prompt` from eval config

In `fastsnake_env.py:make_fastsnake_env()`:
- Lines 32-35: Reads from `config.prompt.prompt.environment_instruction` (NOT `envs.instruction_prompt`)

**Fix needed:** In evaluator, set `temp_config.prompt.prompt.environment_instruction` instead of `temp_config.envs.instruction_prompt`

**Note:** `multi_action_reasoning` is already correctly set to `temp_config.prompt.prompt.multi_action_reasoning` (line 344).

### 2026-01-16 - Feature Complete
**Changes:**
- `multi_env_evaluator.py`: Write to `prompt.prompt.environment_instruction` instead of `envs.instruction_prompt`
- `babyai_env.py`: Read from `environment_instruction` for API consistency with FastSnake/Overcooked
- `__init__.py`: Re-enabled BabyAI in make_env

**Follow-up:** Created backlog card for BabyAI missing from `get_action_extraction_fn` (entropy probing will fail).
