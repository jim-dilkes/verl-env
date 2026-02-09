# Epsilon Re-tokenization for On-Policy Training

## Overview

When epsilon-greedy exploration triggers during training, the executed action differs from what the model generated. Without correction, this creates off-policy training: the model learns from rewards for actions it didn't actually output.

This feature fixes this by re-tokenizing the response text to match the executed action, ensuring on-policy training.

## Problem

```
LLM generates: "<actions>...<decision>up</decision>"
     ↓
epsilon triggers → executed_action = "left" (random)
     ↓
env.step("left") → reward for "left"
     ↓
Without fix: batch['responses'] contains "up" tokens → OFF-POLICY
With fix: batch['responses'] re-tokenized to contain "left" → ON-POLICY
```

## Configuration

Epsilon is configured via prompt config:

```yaml
# verl/trainer/config/prompt/snake.yaml
prompt:
  epsilon: 0.0  # 0 = no exploration, 1 = always random
  multi_action_reasoning: true  # Required for re-tokenization
```

Re-tokenization **only works with multi-action mode** (`<decision>` tag format).

## Key Components

### 1. VecEnv (verl/envs/vec_env.py)

- Centralized epsilon-greedy exploration in `worker()` function
- Sets `info["epsilon_explored"] = True` when epsilon triggers
- Sets `info["executed_action_text"]` to the actual action taken
- Fixes `is_valid = True` for epsilon actions (always valid from action space)

### 2. Trainer (verl/trainer/ppo/ray_multistep_trainer.py)

**Helper functions:**

- `rewrite_decision_tag(response_text, new_action)`: Replaces LAST `<decision>X</decision>` tag
- `retokenize_epsilon_sample(...)`: Re-tokenizes and rebuilds all tensors

**Main logic (after env.step()):**

1. Check `info["epsilon_explored"]` for each sample
2. Rewrite `<decision>` tag with `executed_action_text`
3. Re-tokenize response, preserving original prompt tensors
4. Update `responses`, `input_ids`, `attention_mask`, `position_ids`

**Logprob handling:**

When `bypass_recomputing_logprobs=True` but epsilon modifications occurred, forces recompute to ensure correct logprobs.

## Tensor Update Details

When re-tokenizing, we must update four tensors consistently:

| Tensor | Shape | Update Method |
|--------|-------|---------------|
| `responses` | `[max_response_length]` | Re-tokenize new text, pad/truncate |
| `input_ids` | `[max_prompt_length + max_response_length]` | Concat original prompt + new response |
| `attention_mask` | Same as input_ids | Copy prompt mask + build response mask |
| `position_ids` | Same as input_ids | Copy prompt positions + continue sequence |

Critical: Prompt portion is **copied exactly** from original, not recomputed.

## Metrics

| Metric | Description |
|--------|-------------|
| `behavior/epsilon_explored` | Float 0/1, whether epsilon triggered for this sample |
| `epsilon_retokenized` | Count of successfully re-tokenized samples per step |
| `epsilon_retokenize_failed` | Count of failed re-tokenizations (no `<decision>` tag) |

## Limitations

1. **Multi-action mode only**: Single-action mode (`<action>` tag) not supported; logs failure metric
2. **Semantic mismatch**: Reasoning text still describes original action, only `<decision>` tag updated
3. **Regex-based**: Assumes `<decision>` tag contains no nested `<` characters

## Testing

**Unit tests:** `tests/trainer/ppo/test_epsilon_retokenization.py` (21 tests)

**Login node test:** `experiments/snake/test_login_node_epsilon.sh`
- epsilon=0.5 for high exploration rate
- multi-action mode enabled
- DEBUG prints for tensor shape verification

## Usage Example

```bash
# Enable epsilon exploration in training
python -m verl.trainer.main_ppo \
  prompt=snake \
  prompt.prompt.epsilon=0.25 \
  prompt.prompt.multi_action_reasoning=true \
  envs.captioner.type=multi_action \
  ...
```

## Adaptive Epsilon-Greedy

Static epsilon uses a fixed exploration rate. Adaptive epsilon adjusts based on reward trend:
- **Improving reward** (positive slope) → low epsilon (consolidate)
- **Plateau/declining** (zero/negative slope) → high epsilon (explore)

### Class: `AdaptiveEpsilon` (`verl/trainer/ppo/adaptive_epsilon.py`)

Pure Python, no torch. Maintains a sliding window of batch mean rewards.

**Formula:** `epsilon = epsilon_max * sigmoid(-k * normalized_slope)`

Where `normalized_slope` is the least-squares slope over the window, with rewards normalized by window-local std for scale invariance.

**Warmup:** epsilon=0 until the reward buffer fills (`window_size` steps).

### Config

```yaml
prompt:
  adaptive_epsilon:
    enabled: false        # Toggle adaptive (overrides static epsilon when true)
    epsilon_max: 0.3      # Maximum epsilon value
    window_size: 50       # Reward history window for slope computation
    k: 5.0               # Sigmoid steepness (higher = sharper transition)
    update_every_n_steps: 1  # How often to push new epsilon to workers
```

### Integration Flow

1. Trainer `__init__`: creates `AdaptiveEpsilon` if `adaptive_epsilon.enabled=true`
2. After each rollout: computes `train/episode_return_mean` from batch rewards
3. Feeds mean return to `AdaptiveEpsilon.update()` → gets new epsilon
4. Calls `self.env.update_epsilon(new_eps)` to push to VecEnv workers
5. Workers receive via `update_epsilon` pipe command, updating the closure variable

### VecEnv `update_epsilon` Command

Added `update_epsilon` command to worker command loop (`vec_env.py`). VecEnv sends new epsilon to all workers via pipes; workers update the `epsilon` variable read by `env_step()` closure. Same reassignment pattern used by `hard_reset`.

### Metrics

| Metric | Description |
|--------|-------------|
| `adaptive_epsilon/value` | Current epsilon value |
| `adaptive_epsilon/slope_normalised` | Normalized reward slope |
| `adaptive_epsilon/buffer_fill` | Fraction of window filled (0→1) |
| `train/episode_return_mean` | Mean episode return (logged always, not just adaptive) |

### Testing

**Login node test:** `experiments/snake/test_login_node_adaptive_epsilon.sh`
- `window_size=3` for quick buffer fill during 3-step test
- Verify `adaptive_epsilon/value=0` during warmup, changes after buffer fills

## Related Files

- `verl/envs/vec_env.py` - Epsilon exploration logic + `update_epsilon` command
- `verl/trainer/ppo/ray_multistep_trainer.py` - Re-tokenization + adaptive epsilon integration
- `verl/trainer/ppo/adaptive_epsilon.py` - AdaptiveEpsilon class
- `verl/envs/captioners/multi_action.py` - Multi-action captioner
- `verl/envs/environments/FastSnake/base.py` - `extract_action()` with multi-action support
