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

## Original Notes
Get model to generate a possible justification for each possible action (or randomly smapled subset of possible actions) then either exploit (use the best one, based on its ranking), or explore (sample from the other possible exploration routes)

- Look at probabilites of allowed action tokens for each of these threads
- My final decision is.... {token} - look at the probability distribution of the allowed actions
- Positive/negative/neutral reasons ?
- Does it give better performance than initally??
- TODO: sketch out how this would be implemented.

So the simplest version of this is to boost the allowed input context length a bit, and prompt the model with instruction (and probably a demo template) asking it to, for each listed action (from the env's own list of actions), generate the name of the action, followed by an explanation/justification/thought process as to why that action is optimal.

**Extension 1**
Once all are generated get it to assign a probability distribution over the actions, selecting the top action ?
**Extension 2**
Have a mechanism to extract the probabilities and sample from them. Then we somehow need to tune it to be more/less likely to make that decision based on the outcome??

One possibility - we generate the possible actions and probabilities - sample from them (or epsilon-greedy) and then remove the boilerplate bits, finetuning as though that was the only reasoning and action it generated.

Only thing is - if we are fine-tuning as though we didnt ask for multiple answers, will it actually affect how it generates answers with that context? I.e. we are optimising for something that is not quite "on policy" for the training scenario (and not in an epsilon sampling off policy way....). As in, would the learnings actually translate to the multi-choice situation used for training if we finetune as though they are single choice. Maybe we could finetune both the multi-choice version AND the single-choice version, so it gets the learnings for both situations ? Test both?
