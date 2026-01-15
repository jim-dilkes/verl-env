# Action Probability Sampling

## Status
- Created: 2026-01-11

## Summary
Get action probability distributions for sampling. Two approaches: (A) extract renormalized logprobs at decision token, (B) prompt model to assign confidence scores after reasoning. Augments epsilon-greedy with probability-based sampling.

## Research Goals
- **Controllable exploration**: Tune exploration via sampling without prompt changes
- **Training signal quality**: Better learning signal from multi-action reasoning
- **Exploration signal**: Use entropy/uncertainty to drive exploration
- **Action selection**: Sample actions proportional to model confidence

## Scope
**In scope:**
- **Approach A**: Extract & renormalize logprobs for allowed actions at `<decision>` token
- **Approach B**: Prompt model to assign confidence per action AFTER generating all reasoning
- Sampling strategies: temperature, top-k, nucleus
- Augment epsilon-greedy: sample from probs OR epsilon-override to uniform
- Configurable entropy bonus/penalty in reward signal
- Metrics: action entropy, confidence distribution

**Out of scope:**
- Training loop changes (separate card)

## Probability Source Options

### Approach A: Logprob Extraction at Decision Token
Same format as current multi-action reasoning. At generation time:
1. Model generates `<actions>...</actions>`
2. Model generates `<decision>`
3. **At this point**: extract logprobs for allowed action tokens ("up", "down", "left", "right")
4. Renormalize these logprobs to get action probability distribution
5. Sample from distribution (or use for analysis)

```
<actions>
<action name="up">reasoning...</action>
<action name="down">reasoning...</action>
<action name="left">reasoning...</action>
<action name="right">reasoning...</action>
</actions>
<decision>down</decision>  ← extract logprobs here for "up"/"down"/"left"/"right"
```

- **Pros**: True model confidence, no prompt changes, already have the format
- **Cons**: Need to intercept generation at right point, extract specific token logprobs

### Approach B: Model-Assigned Confidence (Two-Pass)
Model generates reasoning for ALL actions, then reviews and assigns confidence to each:

```
<actions>
<action name="up">reasoning about up...</action>
<action name="down">reasoning about down...</action>
<action name="left">reasoning about left...</action>
<action name="right">reasoning about right...</action>
</actions>
<confidence>
<action name="up">0.1</action>
<action name="down">0.6</action>
<action name="left">0.2</action>
<action name="right">0.1</action>
</confidence>
<decision>down</decision>
```

The key difference: model assigns confidence AFTER seeing all its own reasoning, not inline.

- **Pros**: Explicit deliberation over confidence, interpretable, model can compare all options
- **Cons**: More tokens, stated vs actual confidence may differ, need validation

### Approach C: Compare Both
- Implement both, run experiments
- Compare: does model-assigned confidence correlate with logprob confidence?
- Which leads to better exploration/learning?

## Implementation Plan

### For Approach A (Logprobs)
1. **Trainer side**: After vLLM generates response, find `<decision>` token position
2. **Extract logprobs**: Get logprobs for "up", "down", "left", "right" tokens at that position
3. **Renormalize**: softmax over just those 4 logprobs → action distribution
4. **Pass to env**: Include action_probs in info dict
5. **Wrapper**: Sample from distribution (with epsilon override)

Technical detail: May need vLLM to return per-token logprobs, or do a separate forward pass.

### For Approach B (Model-Assigned)
1. **Modify prompt**: Add `<confidence>` block after `<actions>`, before `<decision>`
2. **Parser**: Extract confidence values from XML
3. **Validation**: Handle invalid probs (negative, don't sum to 1) - normalize
4. **Wrapper**: Sample from distribution

### Action Selection in Wrapper
```python
def extract_action(self, action, action_probs=None):
    # Parse decision from text
    extracted = self.extract_decision_from_xml(action)

    # If model-assigned, extract from text
    if self.prob_source == "model_assigned":
        action_probs = self.extract_confidence_block(action)

    # Sample from probs if available
    if self.use_prob_sampling and action_probs:
        # Epsilon-greedy override to uniform
        if random.random() < self.epsilon:
            valid_action = random.choice(self.language_action_space)
        else:
            valid_action = self.sample_from_probs(action_probs, self.temperature)
    else:
        valid_action = extracted if extracted in self.language_action_space else self.default_action

    return full_action, extracted, valid_action, is_valid, metrics
```

### Entropy Reward (configurable)
- Compute action entropy from probabilities
- Add to reward: `reward += entropy_coeff * entropy`
- `entropy_coeff > 0`: reward exploration (uncertain → good)
- `entropy_coeff < 0`: reward confidence (certain → good)
- `entropy_coeff = 0`: no effect (default)

## Config Parameters
```yaml
prompt.prompt:
  multi_action_reasoning: true
  use_prob_sampling: true
  prob_source: "logprobs"  # or "model_assigned"
  sampling_temperature: 1.0
  epsilon: 0.1  # Override to uniform
  entropy_coeff: 0.0  # Reward modification
```

## Prompt Format (Approach B)
```
[Response Format]
You must reason about EACH available action, then assign confidence scores, then decide.

First, analyze each action:
<actions>
<action name="up">Your reasoning about moving up...</action>
<action name="down">Your reasoning about moving down...</action>
<action name="left">Your reasoning about moving left...</action>
<action name="right">Your reasoning about moving right...</action>
</actions>

Then, assign confidence to each action (must sum to 1.0):
<confidence>
<action name="up">0.X</action>
<action name="down">0.X</action>
<action name="left">0.X</action>
<action name="right">0.X</action>
</confidence>

Finally, output your decision:
<decision>your_chosen_action</decision>
```

## Baselines for Comparison
1. Current single-action (`<action>X</action>`)
2. Multi-action no sampling (jobs 518956/518957 running)
3. Random baseline (uniform actions)
4. Multi-action + logprob sampling (Approach A)
5. Multi-action + model-assigned confidence (Approach B)

## Key Questions
- For logprobs: how to extract at specific token position from vLLM?
- For model-assigned: what if probs don't sum to 1? Normalize silently or penalize?
- Does model-assigned confidence correlate with logprob confidence?
- Should sampling temperature be separate from generation temperature?

## Related
- Depends on: context-driven-action-selection (done)
- Related: multi-action-training-format

## Interview Notes
- Logprob extraction happens in trainer, passed to env
- Model-assigned: confidence block comes AFTER all reasoning, before decision
- Augments epsilon-greedy (not replaces)
- Entropy reward is configurable (bonus, penalty, or off)
- Compare against all baselines