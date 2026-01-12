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
```

## Ideas to Try (from PhD notes)
- Episode sampling (if text gen faster)
- Fewer tokens per step for faster inference
- Halve critic warmup from 40 to 20
- Double discount PPO with intra-step discount=1
- Forking tokens (train only high-entropy tokens)
- Episode truncating for longer trajectories
- Muon/MuonClip optimizer instead of AdamW
