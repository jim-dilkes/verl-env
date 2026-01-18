# Natural Conversation History Flow

**Type:** refactor
**Branch:** feat/conversation-history
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** —

## Goal
Restructure prompt history to use natural conversation flow instead of artificial headers. Remove `[My Previous Thoughts]` from assistant messages - instead let turn-taking make context self-evident.

## Problem
Current format puts `[My Previous Thoughts]` header inside assistant messages:
```
<|im_start|>assistant
[My Previous Thoughts]
<actions>...</actions>
<decision>left</decision>
<|im_end|>
```

This teaches the model to output `[My Previous Thoughts]` as part of its response (imitation).

## Solution
Natural turn structure:
```
<|im_start|>user
[Previous Observation - no diagram]
Step: 0/40
You: pos=(3,1), facing UP, holding: NOTHING
...
<|im_end|>
<|im_start|>assistant
<actions>...</actions>
<decision>left</decision>
<|im_end|>
<|im_start|>user
[Current Observation]
Took action 'stay'
...
[includes diagram]
<|im_end|>
<|im_start|>assistant
[generates here]
```

Model sees: user gave state → I responded → user gave new state → my turn. No special framing needed.

## Scope
- [ ] Store previous observation text alongside action in history events
- [ ] Include previous observation (without diagram) when including previous reasoning
- [ ] Remove `[My Previous Thoughts]` header from assistant messages
- [ ] Update tests if any exist for prompt format

## Files
- `verl/envs/captioners/prompt_builder/history.py` - main changes
- Possibly captioners that call `update_action()` / `update_reasoning()`

## Out of Scope
- Changes to system prompt structure
- Changes to action/decision XML tags (those are for LLM output, not prompt structure)

## Working Notes

### 2026-01-18 - Feature Started
**Interview decisions:**
- Store full previous user input (not summary) in history events
- Format as separate user/assistant turns (not inline headers)
- Keep to just 1 previous turn (matching current behavior)

**Approach:**
1. Modify history event storage to include the user input that preceded the action
2. When building prompt with previous reasoning, emit: user(prev_input) → assistant(prev_reasoning) → user(current_input)
3. Remove `[My Previous Thoughts]` header entirely
