# Natural Conversation History Flow

**Type:** refactor
**Branch:** feat/conversation-history
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** 2026-01-18

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
- [x] Store previous observation text alongside action in history events
- [x] Include previous observation (without diagram) when including previous reasoning
- [x] Remove `[My Previous Thoughts]` header from assistant messages
- [x] Update tests if any exist for prompt format

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

### Context from Docs

**From codebase/file-structure-scope.md:**
- `verl/envs/captioners/prompt_builder/` - HistoryPromptBuilder, history management
- Captioners: `naive.py`, `cot.py`, `base.py` - format prompts

**From environments/wrapper-interface-api.md:**
- Obs format has `text.long_term_context` + `text.short_term_context`
- Multi-action reasoning mode uses `<decision>` tag after `<actions>` block
- `extract_action_instance()` handles multi-action parsing

### 2026-01-18 - Implementation Complete
**Changes made:**
- `history.py`: Store `short_term` in obs events, `observation_text` in action events
- `history.py`: Emit prev obs as user turn before assistant reasoning
- `history.py`: Removed all artificial headers
- `history.py`: Made `get_prompt()` idempotent (non-mutating) per code review feedback
- `test_history_prompt_builder.py`: Rewrote with 11 tests matching current API

**Output format (no headers):**
```
<|im_start|>user
Game started. Score: 0
Grid: 10x10
<|im_end|>
<|im_start|>assistant
<actions>...</actions>
<decision>right</decision>
<|im_end|>
<|im_start|>user
Moved right. Score: 0
Grid: 10x10
<|im_end|>
```
