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
10. [ ] Create diverse prompt templates for Snake/Overcooked

### Phase 3: Analysis Tools
11. [ ] Wandb dashboard for diversity metrics
12. [ ] Episode logging with all prompt responses (for qualitative analysis)

---

## Key Questions for Jim

1. **Prompt design**: Should prompts be:
   - Instruction suffixes (append to existing prompt)?
   - Full instruction replacements?
   - Persona-based ("You are a cautious player...")?

2. **Aggregation priority**: Which method to implement first?
   - Majority vote (simplest)
   - Confidence-weighted (needs token prob extraction)

3. **Batching strategy**: 
   - Accept K× inference cost per step?
   - Or run fewer rollouts to compensate?

4. **Environments**: Start with Snake only, or both Snake + Overcooked?

5. **Baseline comparison**: Should we always include a "no suffix" baseline prompt in the ensemble?

---

## Files to Modify

```
verl/trainer/ppo/multi_env_evaluator.py  # Main changes
verl/trainer/config/evaluation/           # Add diverse eval configs
verl/envs/captioners/                     # Optional: DiverseCaptioner
```

---

## Connection to Dipper Paper

**Dipper approach:**
- Diverse prompts optimized for diversity (not hand-crafted)
- Inference-time ensemble (no training)
- Voting for final answer

**Our adaptation:**
- Start with hand-crafted diverse prompts (cautious/aggressive/strategic)
- Sequential decision-making (not single-shot)
- Can measure exploration benefits over episodes

**Future extensions:**
- Learn optimal diverse prompt set
- RL training on ensemble outputs
- Prompt baking (Project 6 integration)

---

## Next Steps

1. Get Jim's input on key questions above
2. Create minimal Snake diverse eval config
3. Implement Phase 1 in multi_env_evaluator.py
4. Test on local run
5. Deploy to Iridis for full experiments
