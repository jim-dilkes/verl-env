# Config Instruction Override

## Status
- Created: 2026-01-12
- Started: 2026-01-12
- Completed:

## Scope
**In scope:**
- Fix fastsnake_env.py to use config `environment_instruction` even when `multi_action_reasoning=True`
- Add multi-action instruction to snake.yaml prompt config
- Add multi-action instruction to overcooked.yaml prompt config
- Create local test to verify config instruction takes precedence over env default
- Verify Overcooked env also respects config instruction

**Out of scope:**
- Evaluation config changes (already have `instruction_prompt` in eval configs)
- Training loop changes
- New prompt formats

## Goals
- [x] Fix fastsnake_env.py: config instruction always overrides when provided
- [x] Add multi_action_instruction to snake.yaml (copy from FastSnake wrapper default)
- [x] Add multi_action_instruction to overcooked.yaml
- [x] Add multi_action support to Overcooked wrapper (epsilon, decision parsing)
- [x] Create local feature test to verify override logic
- [x] Test both envs

## Acceptance Criteria
- When `environment_instruction` is set in prompt config, it's used regardless of `multi_action_reasoning` setting
- When `environment_instruction` is NOT set, wrapper generates appropriate default (standard or multi-action)
- Overcooked supports multi-action reasoning format
- Local test passes

## Test Cases
- FastSnake with multi_action_reasoning=true and custom environment_instruction → uses custom
- FastSnake with multi_action_reasoning=true and NO environment_instruction → uses wrapper's multi-action default
- FastSnake with multi_action_reasoning=false and custom environment_instruction → uses custom
- Overcooked with multi_action_reasoning=true → supports decision parsing

## Constraints
- Must maintain backwards compatibility (existing configs without environment_instruction should still work)
- Don't break evaluation configs

## Context
- Related files:
  - `verl/envs/environments/FastSnake/fastsnake_env.py` (fix override logic)
  - `verl/envs/environments/FastSnake/base.py` (wrapper with default prompts)
  - `verl/envs/environments/overcooked/overcooked_env.py` (needs multi-action support)
  - `verl/envs/environments/overcooked/base.py` (add multi-action support)
  - `verl/trainer/config/prompt/snake.yaml` (add instruction)
  - `verl/trainer/config/prompt/overcooked.yaml` (add instruction)

## Interview Notes
- Config instruction should ALWAYS override env default when provided
- If config doesn't specify instruction, fall back to env built-in default
- Instruction content same as env built-in (just moving to config)
- Both Snake and Overcooked need this
- Simple local test for verification