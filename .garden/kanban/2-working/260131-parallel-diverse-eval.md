# Parallel Diverse Evaluation Implementation

**Type:** feat  
**Branch:** feat/parallel-diverse-eval  
**Created:** 2026-01-31 22:15  
**Started:** 2026-01-31 22:15  
**Completed:** —

## Goal

Implement Dipper-style parallel diverse prompting for evaluation. Run K parallel inferences with different prompt suffixes per step, aggregate via majority vote.

## Scope

### Phase 1: Core Implementation
- [x] Create eval config file (`overcooked_diverse.yaml`)
- [x] Add `_expand_for_diverse_prompts()` helper method
- [x] Add `_aggregate_diverse_responses()` helper method  
- [x] Add `_compute_diverse_metrics()` helper method
- [x] Modify `_evaluate_single_env_body()` to handle parallel diverse mode
- [x] Track agreement metrics

### Phase 2: Testing
- [x] Unit tests for expansion/aggregation helpers
- [x] Integration test with mock responses
- [ ] Login node test with actual model (requires cluster)

### Phase 3: Metrics & Logging
- [ ] Per-prompt performance tracking
- [ ] Agreement rate logging to wandb
- [ ] Episode logging with all K responses (optional)

## Out of Scope
- Training/finetuning changes
- Learned aggregation (confidence-weighted etc.)
- Prompt optimization (hand-crafted prompts only)

## Key Decisions

1. **Additive changes only**: All new code in `multi_env_evaluator.py` as new methods. Minimal modification to existing `_evaluate_single_env_body()`.

2. **Injection point**: After `apply_chat_template()` returns text (line ~904), before tokenization. This keeps changes isolated to evaluator.

3. **Suffix injection strategy**: Append suffix to the end of the prompt text (after chat template is applied). This works because:
   - Chat template already formatted the messages
   - Suffix appears just before the assistant's turn
   - No need to parse message structure

4. **Batch strategy**: Single batched inference of K×B prompts rather than K separate batches. More efficient on GPU.

5. **Response reshaping**: After inference, reshape (K×B,) → (B, K) then aggregate per rollout.

## Technical Design

### New Helper Methods (pure functions, no side effects)

```python
def _expand_for_diverse_prompts(
    self, 
    prompt_texts: List[str],  # B prompts
    diverse_config: Dict,     # parallel_diverse config
) -> Tuple[List[str], int]:
    """Expand B prompts to K×B prompts with suffix injection.
    
    Returns:
        expanded_prompts: K×B prompts (first K are variants of prompt 0, etc.)
        n_prompts: K (number of prompt variants)
    """
    prompts_config = diverse_config['prompts']
    K = len(prompts_config)
    
    expanded = []
    for prompt_text in prompt_texts:
        for prompt_cfg in prompts_config:
            suffix = prompt_cfg.get('suffix')
            if suffix:
                # Append suffix before the final generation prompt
                expanded.append(prompt_text.rstrip() + "\n\n" + suffix.strip())
            else:
                expanded.append(prompt_text)
    
    return expanded, K


def _aggregate_diverse_responses(
    self,
    responses: List[str],         # K×B responses
    n_rollouts: int,              # B
    n_prompts: int,               # K
    action_extraction_fn: Callable,
    aggregation: str = "majority_vote",
) -> Tuple[List[str], List[Dict]]:
    """Aggregate K×B responses to B actions via voting.
    
    Returns:
        final_responses: B responses (the winning response for each rollout)
        agreement_info: B dicts with agreement stats per rollout
    """
    # Reshape: (K×B,) → (B, K)
    responses_by_rollout = []
    for i in range(n_rollouts):
        rollout_responses = [responses[i * n_prompts + k] for k in range(n_prompts)]
        responses_by_rollout.append(rollout_responses)
    
    final_responses = []
    agreement_info = []
    
    for rollout_responses in responses_by_rollout:
        # Extract actions from each response
        actions = []
        action_to_response = {}
        for resp in rollout_responses:
            _, _, executed, is_valid, _ = action_extraction_fn(resp)
            actions.append(executed)
            if executed not in action_to_response:
                action_to_response[executed] = resp
        
        # Majority vote
        from collections import Counter
        action_counts = Counter(actions)
        winner = action_counts.most_common(1)[0][0]
        
        # Use the first response that produced the winning action
        final_responses.append(action_to_response[winner])
        
        # Agreement stats
        agreement_info.append({
            "unanimous": len(set(actions)) == 1,
            "winner_votes": action_counts[winner],
            "n_prompts": n_prompts,
            "unique_actions": len(set(actions)),
        })
    
    return final_responses, agreement_info
```

### Modification to `_evaluate_single_env_body()`

```python
# After line 904 (apply_chat_template):
val_input_obs_text = self.tokenizer.apply_chat_template(
    obs_vec, tokenize=False, add_generation_prompt=True
)

# === NEW CODE STARTS HERE ===
diverse_mode = False
diverse_config = env_config.get('parallel_diverse')
if diverse_config and diverse_config.get('enabled', False):
    diverse_mode = True
    val_input_obs_text, n_diverse_prompts = self._expand_for_diverse_prompts(
        val_input_obs_text, diverse_config
    )
# === NEW CODE ENDS HERE ===

# ... existing tokenization and inference code ...

# After line 982 (batch_decode):
full_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

# === NEW CODE STARTS HERE ===
if diverse_mode:
    full_responses, step_agreement_info = self._aggregate_diverse_responses(
        full_responses,
        batch_n,
        n_diverse_prompts,
        action_extraction_fn,
        diverse_config.get('aggregation', 'majority_vote'),
    )
    # Accumulate agreement metrics (add to step-level tracking)
# === NEW CODE ENDS HERE ===
```

### Config Schema

```yaml
parallel_diverse:
  enabled: true
  aggregation: majority_vote  # Only option for now
  track_agreement: true
  track_per_prompt: true      # Future: per-prompt metrics
  log_all_responses: false    # Future: log all K responses
  
  prompts:
    - name: baseline
      suffix: null
    - name: cautious
      suffix: |
        [IMPORTANT PRIORITY]
        Safety first...
    # ... more prompts
```

### Metrics to Track

| Metric | Description |
|--------|-------------|
| `diverse/unanimous_ratio` | % of steps where all K prompts agreed |
| `diverse/mean_unique_actions` | Average distinct actions per step |
| `diverse/winner_vote_ratio` | Average vote share of winning action |

## Working Notes

### 2026-02-01 00:15 - Phase 2 Unit Tests Complete

Created `tests/trainer/ppo/test_parallel_diverse_eval.py` with:
- `TestExpandForDiversePrompts`: 6 test cases for suffix injection
- `TestAggregateDiverseResponses`: 7 test cases for majority vote
- `TestComputeDiverseMetrics`: 4 test cases for metrics computation
- `TestDiverseEvalIntegration`: 2 integration-style tests

All tests pass locally with mock action extractor (no actual LLM needed).

Note: Full integration test requires cluster with actual model - will need to be run on Iridis.

**Commits:**
- `fbec6543` - test: Add unit tests for parallel diverse prompting

### 2026-01-31 23:35 - Phase 1 Implementation Complete

Implemented core functionality in `multi_env_evaluator.py`:

**New helper methods (lines 1599-1720):**
- `_expand_for_diverse_prompts()` - Expands B prompts to K×B with interleaved suffixes
- `_aggregate_diverse_responses()` - Majority vote to aggregate K×B→B responses
- `_compute_diverse_metrics()` - Computes unanimous_ratio, mean_unique_actions, winner_vote_ratio

**Modifications to `_evaluate_single_env_body()`:**
1. Added diverse config extraction + accumulators (lines 778-793)
2. Added prompt expansion after apply_chat_template (lines 970-975)
3. Added response aggregation after batch_decode (lines 1011-1041)
4. Added diverse metrics to output (lines 1419-1422)

**Metrics tracked:**
- `diverse/unanimous_ratio` - % of steps where all K prompts agreed
- `diverse/mean_unique_actions` - Average distinct actions per step
- `diverse/winner_vote_ratio` - Avg vote share of winning action (e.g., 0.8 = 4/5)
- `diverse/n_prompts` - K (number of prompt variants)
- `diverse/total_decisions` - Total aggregation decisions made

**Commits:**
- `1e3816fa` - feat: Add parallel diverse prompting evaluation support

**Next:** Phase 2 - Testing

### 2026-01-31 22:15 - Initial Planning

Analyzed codebase via `.garden/docs-agent/`:
- `multi_env_evaluator.py` is the right place for changes
- `obs_vec` is list of message dicts from captioners
- `apply_chat_template()` converts to text strings
- Injection point: after text conversion, before tokenization

Key insight: Can work entirely with text prompts (post chat-template). No need to modify message structures or captioners.

Design decision: Make all changes **additive**:
- New helper methods: `_expand_for_diverse_prompts()`, `_aggregate_diverse_responses()`
- Minimal changes to `_evaluate_single_env_body()` (just conditionals around existing flow)
- New eval config file (doesn't modify existing configs)

### Files to Modify

1. `verl/trainer/ppo/multi_env_evaluator.py`
   - Add `_expand_for_diverse_prompts()` method
   - Add `_aggregate_diverse_responses()` method
   - Modify `_evaluate_single_env_body()` with conditional diverse mode

2. `verl/trainer/config/evaluation/overcooked_diverse.yaml`
   - Already created (draft config)

### Questions for Jim

1. **Suffix location**: Append to very end of prompt text, or try to insert before "Response Format" section?
   - Current plan: End of text (simplest, should work)

2. **Response selection**: When aggregating, should we return the response that produced the winning action, or synthesize a new response?
   - Current plan: Return first response that produced winning action

3. **Memory**: With K=5 prompts, we're doing 5× inference. Should we reduce n_rollouts proportionally in diverse eval configs?
   - Current plan: Keep as-is, user adjusts batch_size if needed

## Original Notes
Implementing Dipper-style parallel diverse prompting for Overcooked evaluation.
Bio-inspired prompt diversity: cautious, aggressive, strategic, cooperative.
Majority vote aggregation as in Dipper paper.
