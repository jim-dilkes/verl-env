# Bugfix: Multi-act Eval Prompt

## Status
- Created: 2026-01-12
- Started: 2026-01-12
- Completed:

## Scope
**In scope:**
- Fix evaluator to inherit captioner.type from training config by default
- Allow explicit override in eval config when desired
- Test multi-action training with correct eval prompts

**Out of scope:**
- Changing how captioners construct prompts
- New captioner types

## Goals
- [ ] Fix eval to use correct multi-act prompt
- [ ] Test with 25% epsilon (much higher than current)

## Problem Analysis
When training with `envs.captioner.type=multi_action` and `prompt.prompt.multi_action_reasoning=True`:
- Training uses `MultiActionCaptioner` which relies on `instruction_prompt` for format
- Evaluations in `snake_evals.yaml` explicitly set `captioner.type: naive`
- `NaiveCaptioner` appends its own response template, conflicting with multi-action format

### Root Cause
Eval YAML hardcodes `captioner.type: naive` for all environments (lines 36-38, 80-82, 126-128, etc. in `snake_evals.yaml`). The evaluator's `_create_env_config()` merges eval config on top of training config, so the explicit `type: naive` overrides.

### Solution
Two-part fix:
1. **Eval YAMLs**: Remove `type: naive` from captioner blocks (or omit `type` entirely) so they inherit from training
2. **Optional explicit override**: Eval configs can still set `captioner.type: <value>` when intentionally testing different format

## Acceptance Criteria
- [ ] Training with `multi_action` captioner uses same captioner in evals (when eval doesn't override)
- [ ] Eval configs can still explicitly set different captioner type when desired
- [ ] Logged eval generations show correct multi-action format

## Test Cases
1. Run `test_login_node_multi_action.sh` - verify eval uses multi_action format
2. Create eval with explicit `captioner.type: naive` - verify it uses naive format (override works)
3. Compare logged generations before/after fix

## Implementation Plan
1. Remove `type: naive` from `snake_evals.yaml` captioner blocks (or create new `snake_evals_inherit.yaml`)
2. Verify merge logic in `_create_env_config()` correctly inherits when `type` not specified
3. Run test to confirm fix

## Constraints
- Must not break existing experiments that intentionally use different eval captioner
- Backward compatible - explicit type in eval config still works

## Context
- Related: context-driven-action-selection (completed)
- Files:
  - `verl/trainer/ppo/multi_env_evaluator.py` - `_create_env_config()` merge logic
  - `verl/trainer/config/evaluation/snake_evals.yaml` - eval configs
  - `verl/envs/captioners/__init__.py` - `make_captioner()`

## Original Notes
Multi-act context mediated exploration is almost correct, however the evaluations don't actually use the correct prompt.
