# Experimental Configurations

## Active Experiment Conditions

| Config | Description |
|--------|-------------|
| `Hpt001` | entropy_coeff=0.001 (baseline) |
| `Hpt005` | entropy_coeff=0.005 |
| `150tok` | max_tokens=150 (fewer tokens/action) |
| `clipcov` | Clip-based covariance entropy |
| `klcov` | KL-based covariance entropy |

## Key Config Parameters

```yaml
# Environment
envs.env_name: babyai | fastsnake | webshop | overcooked | ...
envs.task: BabyAI-PickupDist-v0 | ...
envs.n_rollouts: 32
envs.freeze_completed_episodes: True
envs.format_penalty: 1.0  # Penalty for invalid actions
envs.binary_reward: False
envs.group_rollout_size: null  # For GRPO grouping
envs.group_initial_seed: random | <int>

# Captioner
envs.captioner.type: naive | cot | multi_action
#   naive: Appends response template (think/plan/action XML)
#   cot: Chain-of-thought format
#   multi_action: Relies on instruction_prompt for format (no appended template)
envs.captioner.max_text_history: 0-16
envs.captioner.max_image_history: 0-1

# Prompt (multi-action mode)
prompt.prompt.multi_action_reasoning: false  # Enable multi-action reasoning format
prompt.prompt.epsilon: 0.0  # Epsilon-greedy exploration (0=deterministic)

# Algorithm
algorithm.adv_estimator: GAE | GRPO | RLOO | REINFORCE_PLUS_PLUS
algorithm.step_gamma: 0.99
algorithm.step_lam: 0.95
algorithm.token_gamma: 1.0
algorithm.token_lam: 1.0

# Training
data.train_batch_size: 256
trainer.critic_warmup: 40
trainer.test_freq: 5
trainer.total_epochs: 1000
trainer.total_training_steps: 1000
```

## Checkpoint Resumption

When resuming from a checkpoint:
- Checkpoint loads `global_steps` from folder name (e.g., `global_step_1000` → `global_steps=1000`)
- Training increments to `global_steps+1` before first iteration (e.g., resumes at step 1001)
- Training stops when `global_steps >= total_training_steps`

**To continue training for additional steps:**
- Set `total_training_steps` to the **final desired step count** (not the number of additional steps)
- Example: Checkpoint at step 1000, want 1500 more steps → set `total_training_steps=2500`

```yaml
# Resume from step 1000, train to step 2500 (1500 additional steps)
trainer.total_training_steps=2500 \
trainer.total_epochs=2500 \
```

**Note:** Set both `total_epochs` and `total_training_steps` to the same value for safety. The trainer stops at whichever limit is reached first. Since `total_training_steps` is checked first, it controls the actual stop point

## Ideas to Try (from PhD notes)
- Episode sampling (if text gen faster)
- Fewer tokens per step for faster inference
- Halve critic warmup from 40 to 20
- Double discount PPO with intra-step discount=1
- Forking tokens (train only high-entropy tokens)
- Episode truncating for longer trajectories
- Muon/MuonClip optimizer instead of AdamW
