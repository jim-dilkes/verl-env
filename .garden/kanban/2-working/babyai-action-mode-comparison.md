# BabyAI Action Mode Comparison Experiments

## Goal
Create BabyAI experiment configs matching snake/overcooked action mode comparison structure:
- Baseline (single-act) training on GoToLocalS5N2
- Multi-act training variants (eps=0.0, eps=0.2)
- Combined eval suite with single-act + multi-act versions

## Scope

### 1. Create `babyai_evals_combined.yaml`
- [ ] GoToLocalS5N2 greedy (single-act) - in-distribution
- [ ] 3 harder eval tasks greedy (single-act) - OOD generalization
- [ ] MA versions of all 4 tasks above
- [ ] Entropy check for GoToLocalS5N2
- [ ] State visitation for GoToLocalS5N2

### 2. Validate/Update Prompts
- [ ] Verify `babyai.yaml` single-act prompt works with current wrapper
- [ ] Verify `babyai_multi_action.yaml` multi-act prompt works
- [ ] Ensure prompts include all 6 BabyAI actions

### 3. Create Experiment Sbatch Files
Directory: `experiments/babyai/260116_action_mode_comparison/`

**GoToLocalS5N2 training:**
- [ ] `BAI_PPO_4B_GoTo_BL_1.sbatch` - baseline (naive, 16 steps)
- [ ] `BAI_PPO_4B_GoTo_BL_8st_1.sbatch` - baseline (8 steps)
- [ ] `BAI_PPO_4B_GoTo_multi_eps0_1.sbatch` - multi-act, eps=0.0
- [ ] `BAI_PPO_4B_GoTo_multi_eps02_1.sbatch` - multi-act, eps=0.2

**PickupDist training:**
- [ ] `BAI_PPO_4B_Pickup_multi_eps01_1.sbatch` - multi-act, eps=0.1

**UnlockLocalDist training:**
- [ ] `BAI_PPO_4B_Unlock_multi_eps01_1.sbatch` - multi-act, eps=0.1

### 4. Config Alignment
All configs should match snake/overcooked:
- `cpus-per-task=32`
- `save_freq=100`, `test_freq=100`
- `max_cot_history=1`
- `max_response_length=256` (single-act), `512` (multi-act)
- Model: Qwen3-4B-Instruct-2507

## Decisions

### Eval Tasks (OOD generalization)
1. **PickupDist** - adds pickup skill
2. **UnlockLocalDist** - adds key/door skills
3. **GoToObjMazeS4R2** - harder navigation (maze)

### Training Config
- **max_steps**: 16
- **Model**: Qwen3-4B-Instruct-2507 (same as snake/overcooked)

### Eval Episode Lengths
- GoToLocalS5N2: 20 steps
- PickupDist: 20 steps
- UnlockLocalDist: 20 steps
- GoToObjMazeS4R2: 60 steps (maze exploration needs more)

### Entropy/StateVisit
- Copy structure from snake_evals_combined
- State visitation: 400 rollouts, seed_group_size=20

## Technical Notes

### Existing Prompts
Single-act (`babyai.yaml`):
```yaml
naive_instruction: |
  <plan>...</plan>
  <action>...</action>
```

Multi-act (`babyai_multi_action.yaml`):
```yaml
naive_instruction: |
  <actions>
  <action name="turn left">...</action>
  ...all 6 actions...
  </actions>
  <decision>...</decision>
```

### BabyAI Actions (6 total)
```
turn left, turn right, go forward, pick up, drop, toggle
```

### Wrapper Support
- `multi_action_reasoning` flag already supported in `BabyAILLMAgentsWrapper`
- Default prompts built-in for both modes
- `instruction_prompt` can be overridden per-eval

## Working Notes
- Created: 2026-01-16
