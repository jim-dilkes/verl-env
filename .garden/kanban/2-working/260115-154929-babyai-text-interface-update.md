# Update BabyAI-Text Wrapper to Current Interface

**Type:** feat
**Branch:** feat/babyai-wrapper
**Created:** 2026-01-15 15:49
**Started:** 2026-01-16
**Completed:** —

## Goal
Bring `babyai_text/` wrapper up to parity with Overcooked's LLMAgentsWrapper pattern, including multi-action reasoning support.

## Scope
- [ ] Add `extract_action_instance()` method with multi-action reasoning support
- [ ] Add `multi_action_reasoning` config flag
- [ ] Move instruction prompt into wrapper class (`_default_instruction_prompt()`)
- [ ] Support config-driven `instruction_prompt` override
- [ ] Add `get_instruction_prompt()` as instance method (self-contained)
- [ ] Update `babyai_env.py` to pass `multi_action_reasoning` from config
- [ ] Update `env_wrapper.py` to use new self-contained pattern
- [ ] Test on login node

## Out of Scope
- Task-specific instruction prompts
- Changes to `babaisai/` wrapper
- New BabyAI tasks or environments

## Key Decisions
- **Single task-agnostic prompt**: One generic instruction covering all BabyAI task types (GoTo, Pickup, etc.)
- **Self-contained pattern**: Instruction prompt lives in wrapper class, not separate module function
- **Multi-action format**: Use `<decision>` tag for final action (matching Overcooked pattern)
- **Standard format**: Use `<action>` tag when not in multi-action mode

## Working Notes
<!-- Session handoff and working context goes here -->
### 2026-01-16 - Feature Started
Interview complete. Target: bring babyai_text wrapper to Overcooked-level interface.

Key files:
- `verl/envs/environments/babyai_text/llm_agents_wrapper.py` - main wrapper to update
- `verl/envs/environments/babyai_text/babyai_env.py` - factory function
- `verl/envs/environments/env_wrapper.py` - instruction prompt dispatch

Reference: `verl/envs/environments/overcooked/base.py` (target pattern)

## Original Notes
(feat) - set up BabyAI for current env interface
