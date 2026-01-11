# Context-Driven Action Selection

## Status
- Created: 2026-01-11
- Started: 2026-01-11
- Completed:

## Scope
**In scope:**
- Multi-action reasoning prompt: model generates reasoning for each valid action
- Structured XML output format
- Model ranks and picks final action
- Epsilon-greedy exploration (override model's pick with random action)
- FastSnake environment initially

**Out of scope:**
- Probability/logprob extraction
- Training loop changes
- Multi-agent coordination
- Other environments (Overcooked etc) - can extend later

## Goals
- [ ] Design multi-action reasoning prompt template
- [ ] Modify FastSnake captioner to generate action-reasoning prompt
- [ ] Implement response parser for XML action format
- [ ] Add epsilon-greedy override mechanism
- [ ] Test rollouts with new prompt format

## Acceptance Criteria
- Model generates `<action name='X'>reasoning</action>` for each valid action
- Model outputs `<decision>X</decision>` to select final action
- Parser extracts decision correctly
- Epsilon parameter controls random action override rate
- Rollouts complete successfully with new format

## Test Cases
- Valid XML parsing with all action types
- Decision extraction matches valid actions
- Epsilon=0 → always use model's pick
- Epsilon=1 → always random
- Graceful handling of malformed model output

## Constraints
- Must work within existing LLMAgentsWrapper interface
- Context length may need adjustment (check token usage)
- Single agent only

## Context
- Related docs: `.brisk/scratchpad/environment-interface.md`, `.brisk/scratchpad/fastsnake-env.md`
- Key files: `verl/envs/captioners/`, `verl/envs/environments/FastSnake/`

## Interview Notes
- Core approach: model generates reasoning for ALL valid actions, then picks best
- Format: structured XML tags for parseability
- Exploration via epsilon-greedy on final decision (override model's pick)
- Training format TBD - get generation working first
- Start with FastSnake for faster iteration
