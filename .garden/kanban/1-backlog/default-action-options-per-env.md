# Add per-environment default-action options

## Problem
We currently hardcode fallback/default actions when parsing fails (e.g., Overcooked falls back to `stay`, FastSnake falls back to `up`). This is scattered across env wrappers and `get_action_extraction_fn`, and it’s easy to accidentally change behavior during refactors.

## Goal
Add an explicit, configurable **default action** per environment that is used when:
- the model output cannot be parsed (missing tags / malformed XML)
- the parsed action is invalid

## Proposal
- Extend env configs (Hydra/OmegaConf) with something like:
  - `envs.default_action: <string>` (global) and/or
  - `<env>_kwargs.default_action: <string>` (per environment)
- Update each env wrapper to use the configured default rather than a hardcoded one.
- Update `get_action_extraction_fn(env_name, ...)` to honor the same configured default (or accept an optional `default_action` parameter).

## Acceptance criteria
- Each environment has a single source of truth for its default fallback action.
- Behavior is covered by a small unit test per environment (parse invalid → fallback to configured default).

## Notes / touchpoints
- `OvercookedLLMAgentsWrapper.default_action` in [verl/envs/environments/overcooked/base.py](verl/envs/environments/overcooked/base.py)
- `FastSnakeLLMAgentsWrapper.default_action` / extraction in [verl/envs/environments/FastSnake/base.py](verl/envs/environments/FastSnake/base.py)
- `get_action_extraction_fn` in [verl/envs/environments/__init__.py](verl/envs/environments/__init__.py)
