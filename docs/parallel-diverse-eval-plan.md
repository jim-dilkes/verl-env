# Parallel Diverse Reasoning - Evaluation Implementation Plan

**Date:** 2026-01-31  
**Branch:** feat/parallel-diverse-eval  
**Status:** Planning

---

## Goal

Implement Dipper-style parallel diverse prompting for **evaluation only** (no finetuning changes). This allows measuring how ensemble reasoning from diverse prompts affects performance on SDM tasks.

---

## Background: Current Architecture

### Evaluation Flow
```
multi_env_evaluator.py
  └── _evaluate_single_env_body()
        └── For each step:
              1. Captioner builds prompt (observation → messages)
              2. Tokenizer encodes prompt
              3. actor_rollout_wg.generate_sequences() → response
              4. Extract action from response
              5. env.step(action)
```

### Key Components
- **Captioners** (`verl/envs/captioners/`): Build prompts from observations
  - `naive.py`: Basic prompt with think/plan/action tags
  - `multi_action.py`: Explicit reasoning about each action
- **Eval configs** (`verl/trainer/config/evaluation/*.yaml`): Define eval environments
  - `instruction_prompt`: Override system instruction per eval
  - `generation`: Sampling parameters (temperature, top_p, etc.)
- **VecEnv**: Parallel environments (same seed groups for coverage metrics)

---

## Proposed Design

### 1. Eval Config Extension

Add `parallel_diverse` section to evaluation environment configs:

```yaml
- name: "Snake-Diverse-Ensemble"
  n_rollouts: 50
  episode_length: 20
  # ... standard config ...
  
  parallel_diverse:
    enabled: true
    
    # Diverse prompts (each runs in parallel)
    prompts:
      - name: "cautious"
        instruction_suffix: |
          IMPORTANT: Prioritize safety. Avoid any move that could lead to death.
          Think carefully about risks before acting.
      
      - name: "aggressive"  
        instruction_suffix: |
          IMPORTANT: Prioritize scoring points. Take calculated risks to eat apples.
          Be bold in pursuing rewards.
      
      - name: "strategic"
        instruction_suffix: |
          IMPORTANT: Think several steps ahead. Consider the long-term consequences.
          Optimize your position for future moves.
      
      - name: "baseline"
        instruction_suffix: null  # No modification (control)
    
    # Aggregation method
    aggregation: "majority_vote"  # or "confidence_weighted", "first_valid"
    
    # Metrics to track
    track_agreement: true  # Log when prompts agree/disagree
    track_per_prompt: true  # Log metrics per prompt variant
```

### 2. Implementation Approach

**Option A: Modify Captioner (prompt-level)**
- Create `DiverseCaptioner` that wraps existing captioner
- Appends instruction_suffix to each prompt variant
- Pro: Clean separation, reusable
- Con: Need to handle parallel inference

**Option B: Modify Evaluator (inference-level)** ← Recommended
- Modify `_evaluate_single_env_body()` to handle parallel prompts
- Run K parallel inferences per step (one per prompt variant)
- Aggregate actions before env.step()
- Pro: All diversity logic in one place, easier to track metrics
- Con: More changes to evaluator

### 3. Inference Strategy

For K diverse prompts and B batch size (n_rollouts):
- **Naive**: K × B inferences per step (expensive but simple)
- **Batched**: Concatenate all K prompts across all B rollouts → single inference of K×B
- **Sequential**: Run K inferences of B each (memory-friendly)

Recommend **Batched** for efficiency (single forward pass).

### 4. Aggregation Options

| Method | Description | When to Use |
|--------|-------------|-------------|
| `majority_vote` | Most common action wins | Default, robust |
| `first_valid` | First prompt with valid action | Fast, respects priority |
| `confidence_weighted` | Weight by response confidence | Needs confidence extraction |
| `unanimous_only` | Only act if all agree, else default | Conservative |

### 5. Metrics to Track

**Agreement metrics:**
- `prompt_agreement_ratio`: % of steps where all prompts agree
- `prompt_pairwise_agreement`: Agreement between each pair
- `minority_correct_ratio`: % of steps where minority vote was actually better

**Per-prompt metrics:**
- `{prompt_name}_valid_action_ratio`
- `{prompt_name}_would_have_scored`: Reward if this prompt was sole decider

**Ensemble metrics:**
- `ensemble_vs_baseline`: Score improvement over single-prompt baseline
- `ensemble_diversity`: Entropy of action distribution across prompts

---

## Implementation Steps

### Phase 1: Minimal Viable Implementation
1. [ ] Add `parallel_diverse` config parsing in `_create_env_config()`
2. [ ] Modify `_evaluate_single_env_body()` to detect diverse eval mode
3. [ ] Implement prompt suffix injection (modify observation before tokenization)
4. [ ] Implement batched parallel inference (K×B prompts → single generate call)
5. [ ] Implement majority vote aggregation
6. [ ] Track basic agreement metrics

### Phase 2: Extended Features
7. [ ] Add per-prompt metrics tracking
8. [ ] Implement confidence-weighted aggregation
9. [ ] Add minority-was-correct analysis

### Phase 3: Analysis Tools
10. [ ] Wandb dashboard for diversity metrics
11. [ ] Episode logging with all prompt responses (for qualitative analysis)

---

## Technical Implementation Details

### Config Structure
```yaml
parallel_diverse:
  enabled: true
  aggregation: "majority_vote"  # majority_vote | first_valid | unanimous
  
  prompts:
    - name: "baseline"
      suffix: null
    - name: "cautious"
      suffix: "CRITICAL: Safety first. Never collide..."
    - name: "aggressive"
      suffix: "PRIORITY: Serve dishes fast..."
    # ... more prompts
    
  # Metrics config
  track_agreement: true
  track_per_prompt: true
  log_all_responses: false  # Expensive, for debugging
```

### Code Changes (multi_env_evaluator.py)

**1. Config parsing in `_create_env_config()`:**
```python
# Extract parallel_diverse config
parallel_diverse = env_config.get('parallel_diverse', None)
if parallel_diverse and parallel_diverse.get('enabled', False):
    temp_config.parallel_diverse = parallel_diverse
```

**2. New method `_apply_prompt_suffix()`:**
```python
def _apply_prompt_suffix(self, obs_text: str, suffix: Optional[str]) -> str:
    """Append suffix to the last user message content."""
    if suffix is None:
        return obs_text
    # Insert before the response format section or at end
    return obs_text + "\n\n" + suffix.strip()
```

**3. Modified inference loop:**
```python
# In _evaluate_single_env_body, if parallel_diverse enabled:
diverse_cfg = env_config.get('parallel_diverse')
prompts = diverse_cfg['prompts']
K = len(prompts)

# Expand batch: B observations → K×B observations with different suffixes
expanded_obs = []
for obs in val_input_obs_text:  # B observations
    for prompt_cfg in prompts:  # K prompts
        suffix = prompt_cfg.get('suffix')
        expanded_obs.append(self._apply_prompt_suffix(obs, suffix))

# Single batched inference: K×B prompts
# ... tokenize expanded_obs ...
val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)

# Reshape responses: (K×B,) → (B, K)
responses = responses.reshape(batch_size, K)

# Aggregate per rollout
final_actions = []
for rollout_responses in responses:
    actions = [extract_action(r) for r in rollout_responses]
    final_action = self._majority_vote(actions)
    final_actions.append(final_action)
```

**4. Majority vote aggregation:**
```python
def _majority_vote(self, actions: List[str]) -> str:
    """Return most common action; tie-break by first occurrence."""
    from collections import Counter
    counts = Counter(actions)
    return counts.most_common(1)[0][0]
```

**5. Agreement metrics:**
```python
def _compute_agreement_metrics(self, all_actions: np.ndarray) -> Dict:
    """Compute agreement across prompts for each step."""
    # all_actions: shape (n_steps, K)
    agreement_ratios = []
    for step_actions in all_actions:
        unique = len(set(step_actions))
        agreement_ratios.append(1.0 if unique == 1 else 0.0)
    return {
        "unanimous_ratio": np.mean(agreement_ratios),
        "mean_unique_actions": np.mean([len(set(a)) for a in all_actions]),
    }
```

### Eval Config File
Create: `verl/trainer/config/evaluation/overcooked_diverse.yaml`

---

## Decisions (from Jim)

1. **Prompt design**: Suffix with persona/approach + bio-inspired separation
2. **Aggregation**: Majority vote (Dipper default)
3. **Cost**: Accept K× inference — no backprop = more memory for batch
4. **Environment**: Overcooked
5. **Baseline**: Yes, include unmodified baseline

---

## Bio-Inspired Prompt Diversity

Drawing from biological parallels (motor planning, bee swarms, Bayesian brain):

| Prompt Name | Bio Inspiration | Focus |
|-------------|-----------------|-------|
| `baseline` | Control | No suffix — standard behavior |
| `cautious` | Threat avoidance (amygdala) | Prioritize safety, avoid collisions/fire |
| `aggressive` | Reward-seeking (dopamine) | Prioritize scoring, take calculated risks |
| `strategic` | Prefrontal planning | Think ahead, optimize position |
| `myopic` | Reactive (brainstem) | Immediate step only, fastest valid action |
| `cooperative` | Social behavior | Focus on teammate coordination |

**Overcooked-specific prompts:**
```yaml
prompts:
  - name: "baseline"
    suffix: null
    
  - name: "cautious"
    suffix: |
      CRITICAL: Safety first. Never collide with your teammate.
      Avoid crowded areas. Wait rather than risk collision.
    
  - name: "aggressive"  
    suffix: |
      PRIORITY: Serve dishes as fast as possible. 
      Take the shortest path to ingredients. Speed over safety.
    
  - name: "strategic"
    suffix: |
      THINK AHEAD: Consider what your teammate is doing.
      Position yourself for the next dish, not just the current one.
      Optimize your path for multiple future actions.
    
  - name: "cooperative"
    suffix: |
      TEAMWORK FOCUS: Watch your teammate's position and likely goal.
      Stay out of their way. Pass items when it helps.
      Two coordinated players beat two independent players.
```

---

## Files to Modify

```
verl/trainer/ppo/multi_env_evaluator.py  # Main changes
verl/trainer/config/evaluation/           # Add diverse eval configs
verl/envs/captioners/                     # Optional: DiverseCaptioner
```

---

## Connection to Dipper Paper

**Dipper approach (EMNLP 2025):**
- 3 components: Prompt Generator → Prompt Selector → Response Aggregator
- Optimizes prompt diversity using DPP-inspired selection (fidelity × diversity)
- Uses majority vote aggregation
- Key result: 3× Qwen2-1.5B ensemble beats single 7B model on MATH

**Our adaptation for SDM:**
- Hand-crafted bio-inspired diverse prompts (v1, can optimize later)
- Sequential decision-making (not single-shot reasoning)
- Track per-step agreement + per-episode outcomes
- Can measure exploration benefits across trajectory

**Key differences:**
| Aspect | Dipper | Our Approach |
|--------|--------|--------------|
| Task type | Single-shot reasoning | Sequential decisions |
| Prompt source | LLM-generated + optimized | Hand-crafted + bio-inspired |
| Metrics | Accuracy | Reward + agreement + coverage |
| Training | None | None (v1), RL on ensemble (v2) |

**Future extensions:**
- Learn optimal diverse prompt set (DPP-style selection)
- RL training on ensemble outputs (reward all agreeing prompts)
- Prompt baking (Project 6 integration)

---

## Next Steps

1. Get Jim's input on key questions above
2. Create minimal Snake diverse eval config
3. Implement Phase 1 in multi_env_evaluator.py
4. Test on local run
5. Deploy to Iridis for full experiments
