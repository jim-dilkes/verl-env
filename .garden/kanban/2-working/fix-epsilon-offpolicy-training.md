# Fix Epsilon Off-Policy Training

## Status
- Created: 2026-01-12
- Started:
- Completed:

## Problem Statement

When epsilon-greedy exploration triggers in `vec_env.py`, we execute a random action but the response tensor still contains tokens for the LLM's original action choice. This creates off-policy training:

```
LLM generates: "<think>...</think><action>up</action>"
     ↓
extract_action() → executed_action = "up"
     ↓
epsilon triggers → executed_action = "left" (random)
     ↓
env.step("left") → reward for "left"
     ↓
batch['responses'] still contains "up" tokens → LOG PROBS MISMATCH
```

Training on (original_text, epsilon_reward) pairs is incorrect - the reward came from a different action than what the text describes.

## Options Considered

### Option A: Modify text + re-tokenize (CHOSEN)
Modify response text in vec_env to match executed action, return to trainer for re-tokenization.

**Pros:**
- On-policy: log probs computed on correct text
- RL signal preserved from epsilon samples
- Clean separation: vec_env handles text, trainer handles tokenization
- Environment diversity AND learning signal

**Cons:**
- Re-tokenization complexity (different token lengths)
- Semantic mismatch (reasoning written for original action, but text says new action)

### Option B: Mask epsilon steps
Set `frozen_mask=1` or zero advantage for epsilon samples.

**Pros:**
- Simple, uses existing infrastructure
- No tokenization issues

**Cons:**
- Wastes compute (still forward pass)
- Reduces effective training data
- Epsilon only helps env diversity, no direct learning signal

### Option C: Importance Sampling
IS ratio ≈ 0 for epsilon samples (random action has ~0 probability under policy).

**Pros:**
- Theoretically principled

**Cons:**
- Essentially equivalent to masking
- More complex implementation

### Option D: Direct token replacement
Find and replace action tokens in tensor.

**Cons:**
- Very complex: locate exact positions, handle length differences
- Not worth the complexity

## Decision: Option A

### Important scope / clarification

- **Trainer, not VecEnv, should own the text rewrite + re-tokenization.**
    - VecEnv already returns `info["executed_action_text"]` and `info["metrics"]["behavior/epsilon_explored"]`.
    - Sending the full modified response string back through multiprocessing pipes is unnecessary overhead.
    - The trainer already has the decoded response string (`actions[i]`) for each env step.

- **Trajectory consistency requirement**
    - The “semantic mismatch” concern is acceptable **only** in **multi-action mode** where the model produces reasoning for multiple candidate actions and then selects one.
    - In the future we may remove non-selected reasonings (keep only selected reasoning + selected action).

- **Token-tensor consistency requirement (critical)**
    - When we change the response text, we must update **all** model-input tensors that depend on it:
        - `responses` (response-only ids)
        - `input_ids` (prompt+response concatenation)
        - `attention_mask`, `position_ids`
    - Updating only `responses` is insufficient because logprob computation uses the full input tensors.

- **Operating mode interaction (critical)**
    - If `bypass_recomputing_logprobs=True`, training may anchor to `rollout_log_probs` rather than recomputed `old_log_probs`.
    - Epsilon-modified samples must not silently train against stale logprobs. The spec below requires either:
        - force recompute for batches containing epsilon modifications, OR
        - mask epsilon-modified samples in bypass mode.

## Implementation Plan

### Phase 1: vec_env.py modifications

**File:** `verl/envs/vec_env.py`
**Function:** `env_step()` (inside `worker()`)

1. Fix epsilon-step validity bookkeeping:
    - When epsilon changes `executed_action`, we must **recompute** validity for the executed action.
    - Do not pass `is_valid` from the original extracted action into `env.step(executed_action, is_valid)`.

2. Keep VecEnv responsible for exploration + telemetry only:
    - Continue setting `metrics["behavior/epsilon_explored"]`.
    - Continue returning `info["executed_action_text"]`.
    - Do **not** send the full modified response string via `info` (avoid IPC overhead).

3. VecEnv cached-info edge case:
    - When `freeze_completed_episodes=True`, `VecEnv.step()` returns cached `info` for `__SKIP__` envs.
    - Ensure any epsilon-specific fields (if added later) are cleared on cached returns, or only use `executed_action_text` + `epsilon_explored` for the current active step.

**Code sketch (conceptual):**
```python
def env_step(action_text):
    extract_fn = getattr(env, 'extract_action_instance', env.extract_action)
    full_action, extracted_action, executed_action, is_valid, metrics = extract_fn(action_text)

    explored = False
    if epsilon > 0 and random.random() < epsilon:
        executed_action = random.choice(env.language_action_space)
        explored = True
        # IMPORTANT: recompute validity for executed_action
        is_valid = env.is_valid_action(executed_action) if hasattr(env, "is_valid_action") else True

    metrics["behavior/epsilon_explored"] = float(explored)
    env_obs, reward, terminated, truncated, info = env.step(executed_action, is_valid)
    info["executed_action_text"] = executed_action
    info["metrics"] = metrics
    return captioner.get_obs(env_obs), reward, terminated, truncated, info, image
```

### Phase 2: ray_multistep_trainer.py modifications

**File:** `verl/trainer/ppo/ray_multistep_trainer.py`
**Location:** After `self.env.step(actions)` call (~line 1125)

1. After `obs_vec, reward_vec, ..., info_vec = self.env.step(actions)`:
    - For each active env `i`:
      - If `info["metrics"].get("behavior/epsilon_explored") == 1.0`, get `executed_action_text`.
      - Rewrite the decoded response string `actions[i]` to match the executed action.
         - **Standard mode:** replace the last well-formed `<action>...</action>`.
         - **Multi-action mode:** prefer updating the `<decision>...</decision>` selection.
            - (Optional future) drop non-selected reasonings.

2. Re-tokenize and update **all dependent tensors** for the affected samples:
    - `responses[i]` (response-only ids)
    - `input_ids[i]` (prompt+response)
    - `attention_mask[i]`, `position_ids[i]`
    - Ensure shape matches `max_prompt_length + max_response_length` and `max_response_length`.

3. Operating mode requirement:
    - If `bypass_recomputing_logprobs=True` and epsilon modifications occurred in this rollout step:
      - Either force `bypass_recomputing_logprobs=False` for this batch, OR
      - Mask these samples from training (equivalent to Option B) for correctness.

**Code sketch (conceptual):**
```python
obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = self.env.step(actions)

for i, info in enumerate(info_vec):
    if not active_envs[i]:
        continue
    if info.get("metrics", {}).get("behavior/epsilon_explored", 0.0) != 1.0:
        continue
    executed_action = info.get("executed_action_text")
    if not executed_action:
        continue

    # Rewrite decoded response string locally (no IPC string payload)
    new_response_text = rewrite_action_in_response(
        response_text=actions[i],
        executed_action=executed_action,
        mode=captioner_or_env_mode,
    )

    # Re-tokenize response text and update responses + full input tensors consistently
    update_sample_tokens_in_batch(
        batch_output=gen_batch_output_or_full_batch_output,
        sample_index=i,
        new_response_text=new_response_text,
        tokenizer=self.tokenizer,
        max_prompt_length=self.config.data.max_prompt_length,
        max_response_length=self.config.data.max_response_length,
    )

if bypass_recomputing_logprobs and any_epsilon_modified:
    # correctness guard
    raise_or_mask_or_force_recompute()
```

**Complexity: Frozen environment handling**

When `freeze_completed_episodes=True`, only active environments generate. Need to:
1. Track which original index maps to which active index
2. Apply re-tokenization to correct position in active_gen_batch_output
3. Then the full_batch reconstruction handles spreading back to all positions

**Index mapping example:**
```
n_rollouts = 8
env_frozen = [F, F, T, F, T, T, F, F]  # T=frozen, F=active
active_envs = [T, T, F, T, F, F, T, T]

original_to_active = {0:0, 1:1, 3:2, 6:3, 7:4}
# i=2,4,5 are frozen → skip
```

**Reconstructing input_ids from prompt + new response:**

The `gen_batch_output` from `generate_sequences()` contains:
- `input_ids`: `[batch, prompt_len + response_len]` (full sequence)
- `responses`: `[batch, response_len]` (response only)
- `attention_mask`, `position_ids`: `[batch, prompt_len + response_len]`

To update after epsilon modification:
1. Get prompt tokens: `prompt_ids = gen_batch.batch['input_ids'][i]` (pre-generation input)
2. Re-tokenize new response: `new_response_ids = tokenizer(new_response_text, ...)`
3. Concatenate: `new_input_ids = concat(prompt_ids, new_response_ids)`
4. Rebuild attention_mask and position_ids accordingly
5. Update all four tensors in `gen_batch_output` or `active_gen_batch_output`

**Key variables at insertion point (~line 1125-1140):**
- `gen_batch`: input to generation (contains prompt-only `input_ids`)
- `gen_batch_output` or `active_gen_batch_output`: output from generation
- `actions`: decoded response strings (already available)
- `info_vec`: contains `executed_action_text` and `epsilon_explored` metric

### Phase 3: Testing

1. **Unit test:** Verify action-tag rewrite works for both tag formats and edge cases
    - multiple tags present → only selection tag updated
    - missing tag → safe fallback behavior (e.g., append tag or mark sample as invalid/masked)
    - malformed XML → no crash; sample masked or left unchanged with metric

2. **Login node test (REQUIRED per CLAUDE.md):**
   - Create `experiments/snake/test_login_node_epsilon.sh` based on baseline script
   - Override: `prompt.epsilon=0.5` to trigger epsilon exploration frequently
   - Verify:
     - `behavior/epsilon_explored` metric ≈ 0.5
     - No crashes/shape mismatches from tensor updates
     - Training completes 3 steps successfully
   - Run: `bash experiments/snake/test_login_node_epsilon.sh` on iridis3 login node

3. **Integration test:** Run with epsilon=0.5, verify:
   - Responses tensor contains modified tokens when epsilon triggers
   - Metrics show `behavior/epsilon_explored` rate ~= epsilon
   - Log probs are computed on modified text (can verify by logging)

4. **Cluster test:** Full training run with epsilon=0.25

## Acceptance Criteria

- [ ] When epsilon triggers, **decoded response text** used for training matches `executed_action_text`
- [ ] All dependent tensors remain consistent: `responses`, `input_ids`, `attention_mask`, `position_ids`
- [ ] Log probs are computed against the modified tokens (verified via logging)
- [ ] Handles `<decision>` tag format (multi-action mode only)
- [ ] Works with `freeze_completed_episodes=True` (frozen env handling)
- [ ] Metrics track epsilon exploration rate
- [ ] No regression in non-epsilon training
- [ ] Login node test passes: `experiments/snake/test_login_node_epsilon.sh`

## Explicit Non-Goals (for now)

- We do not attempt to make the reasoning prefix "true" in standard `<think>/<plan>/<action>` mode.
- We do not implement full off-policy correction for epsilon steps via IS/RS here; this is an on-policy consistency fix.

## Clarifications (from discussion 2026-01-12)

1. **bypass_recomputing_logprobs handling:** Force recompute when epsilon modifications occur (don't mask).

2. **Tag format support:** Only support multi-action mode (`<decision>` tag). Single-action mode (`<action>` tag) not supported for epsilon re-tokenization - if used with epsilon, will skip re-tokenization and log metric.

3. **Malformed/missing tag handling:** Leave unchanged + add `epsilon_retokenize_failed` metric (don't mask).

4. **Debug logging in tests:** Add DEBUG prints showing tensor shapes before/after re-tokenization to verify correctness.

## Files to Modify

1. `verl/envs/vec_env.py` - Add text modification in `env_step()`
2. `verl/trainer/ppo/ray_multistep_trainer.py` - Add re-tokenization after env.step()

## Constraints

- Must maintain backwards compatibility when epsilon=0
- Must handle variable token lengths (re-tokenize with padding/truncation)
- Must work with frozen environment logic

## Related

- Depends on: epsilon centralization in vec_env (done, uncommitted)
- Related to: bugfix-multi-act-eval-prompt (in progress)

## Current State (uncommitted in vec_env.py)

Epsilon centralization is already implemented but uncommitted:
- `VecEnv.__init__` reads `config.prompt.prompt.epsilon`
- Passes epsilon to worker processes
- Worker `env_step()` does epsilon-greedy: `if epsilon > 0 and random.random() < epsilon`
- Sets `metrics["behavior/epsilon_explored"]`
- Sets `info["executed_action_text"]` (the actually executed action)

**Still needs fixing in vec_env.py:**
- `is_valid` bug: currently passes original `is_valid` even when epsilon changes action
- Should set `is_valid=True` when epsilon triggers (random action from action space is always valid)

**Trainer-side work (this card's main focus):**
- Text rewrite + re-tokenization
- Full tensor update
- bypass_recomputing_logprobs handling
