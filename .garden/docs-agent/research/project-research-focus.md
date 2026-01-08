# Project Research Focus

## Research Area
Exploration in Sequential Decision Making for LLMs - studying how to encourage LLM agents to explore effectively in multi-step environments using RL.

## Key Research Questions
1. Token entropy vs action entropy - high token diversity doesn't guarantee diverse actions
2. Entropy regularization approaches: loss-based (entropy coeff) vs covariance-based (clip-cov, kl-cov)
3. Model scaling: 0.5B -> 4B -> 7B -> 14B Qwen models
4. Memory-free multi-step RL: GRPO/PPO without full episode context (scales better)

## Novel Contributions
- Multi-step GRPO/PPO without memory (better scaling)
- 14B param models on multi-step envs
- Token entropy regularization for multi-step envs
- Studying inference speed vs generalization tradeoffs

## Comparison Papers
- RAGEN - StarPO trajectory-level RL
- Context-lite Multi-turn RL - Dual discount GAE
- ARPO - Entropy-based adaptive rollout
- Search-R1 - Pseudo-environment masking
- LOOP - Leave-one-out policy estimator
