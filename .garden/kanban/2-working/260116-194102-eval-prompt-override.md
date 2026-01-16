# Fix: Eval Config Prompt Override Ignored

**Type:** fix
**Branch:** fix/wrong-eval-prompt-ma
**Created:** 2026-01-16
**Started:** 2026-01-16
**Completed:** —

## Goal
Ensure evaluation configs use their own `instruction_prompt` and `multi_action_reasoning` fields instead of inheriting from the training config.

## Scope
- [ ] Find where evaluator gets prompt config (likely multi_env_evaluator.py)
- [ ] Fix prompt inheritance to use eval config's fields when specified
- [ ] Verify single-action evals use SA prompts during MA training
- [ ] Verify MA evals use MA prompts during SA training

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

**Files to investigate:**
- `verl/trainer/ppo/multi_env_evaluator.py` - likely where prompt config is passed
- Look for where instruction_prompt is resolved
