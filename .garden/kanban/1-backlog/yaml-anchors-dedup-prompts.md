# YAML Anchors for Duplicated Prompts

## Summary
Use YAML anchors to reduce copy/paste drift in eval config files where instruction prompts are duplicated across multiple environments.

## Context
In `overcooked_evals_multi_action.yaml` and `snake_evals_multi_action.yaml`, the same multi-action instruction prompts are duplicated across multiple eval configs. This creates maintenance burden and risk of drift.

## Proposed Solution
Use YAML anchors (`&anchor_name`) and aliases (`*anchor_name`) to define prompts once and reference them:

```yaml
# Define once at top
_prompts:
  overcooked_multi_action: &overcooked_ma_prompt |
    [Instructions]
    You are playing Overcooked solo...
    ...

# Reference in configs
environments:
  - name: "Overcooked-CrampedRoom-Greedy"
    instruction_prompt: *overcooked_ma_prompt
```

## Notes
- Need to verify Hydra/OmegaConf handles YAML anchors correctly
- May need underscore prefix (`_prompts`) to exclude from config schema
- Test with a small example first

## Priority
Low - Cosmetic/maintenance improvement, not functional
