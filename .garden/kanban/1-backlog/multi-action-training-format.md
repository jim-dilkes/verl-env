# Multi-Action Training Format

## Status
- Created: 2026-01-11

## Summary
Determine optimal training format for multi-action reasoning outputs. Test full output vs stripped vs masked approaches.

## Research Goal
Understand if/how the training format affects learning when using multi-action reasoning prompts.

## Problem Statement
Model generates:
```
<actions>
<action name="up">reasoning about up...</action>
<action name="down">reasoning about down...</action>
...
</actions>
<decision>down</decision>
```

Current behavior: Train on entire response (Option 1). Need to test if alternatives improve learning.

## Training Format Options

### Option 1: Full Output (Current - Running)
- Train on entire response including all action reasoning
- Jobs 518956/518957 testing this now
- Pros: On-policy, model learns reasoning pattern
- Cons: Learns reasoning for unchosen actions

### Option 2: Strip to Decision Only
- Post-process: keep only `<decision>X</decision>` for training
- Pros: Cleaner signal, less tokens, faster training
- Cons: Off-policy - context mismatch between generation/training

### Option 3: Token Masking (Complex)
- Keep full response but mask loss on `<actions>` block
- Only backprop through `<decision>` portion
- Pros: On-policy context, focused signal
- Cons: Requires training loop changes

## Implementation Plan

### For Option 2 (Strip)
Modify response before training:
```python
def strip_to_decision(response_text):
    # Keep only <decision>X</decision>
    match = re.search(r'<decision>(\w+)</decision>', response_text)
    if match:
        return f"<decision>{match.group(1)}</decision>"
    return response_text
```

Where to apply:
- After generation, before batch construction
- Or: during batch construction in trainer
- Need to re-tokenize stripped response

### For Option 3 (Mask)
Modify `response_mask` in training batch:
```python
# Find token positions of <actions>...</actions>
actions_start = find_token_position("<actions>")
actions_end = find_token_position("</actions>")
# Zero out mask in that range
response_mask[actions_start:actions_end] = 0
```

## Experiment Design
1. **Baseline**: Single-action format (existing runs)
2. **Full output**: Multi-action, train on all (jobs 518956/518957)
3. **Stripped**: Multi-action generate, train on decision only
4. Compare: learning curves, final performance, token efficiency

## Priority
Worth testing empirically - run comparison experiments to determine if off-policy stripping matters.

## Key Metrics
- Learning curve speed
- Final task performance
- Tokens per training step
- Valid action ratio over training

## Related
- Depends on: context-driven-action-selection (done)
- Related: action-probability-sampling

## Interview Notes
- Off-policy concern is "worth testing" not critical blocker
- Compare against all baselines
- Empirical comparison preferred over theoretical concerns
