# AAAI Harness — Test Matrix & Launch Plan

8 interventions × 2 envs × {PPO, GRPO} × seeds. PPO+Overcooked first, n=1 before repeats.

Interventions: `baseline H005 H01 H05 clip_cov kl_cov adaptive adaptive_decay`

## Phase 1 — FIRST QUEUED SET (n=1, Overcooked, PPO) ← the 8 most important
The reference + every intervention at seed 1, so we see the full "coverage-up / reward-flat /
task-progress-flat / pass@k" picture for all methods before committing repeats.

```
for I in baseline H005 H01 H05 clip_cov kl_cov adaptive adaptive_decay; do
  sbatch --job-name=AAAI_ppo_overcooked_${I}_s1 \
    --export=ALL,ALGO=ppo,ENV=overcooked,INTERVENTION=$I,SEED=1 \
    experiments/_template/run.sbatch
done
```
6 CPU/job, swarm_a100,swarm_h100, 5-day limit, 2 GPU, 600 steps, eval every 50.

## Phase 2 — Overcooked PPO repeats (seeds 2..5) — after Phase 1 looks healthy
`./launch.sh ppo overcooked <I> 2 5` per intervention (or loop).

## Phase 3 — Snake PPO (needs stochastic pass@k eval block first — see DESIGN_NOTES), then GRPO
Snake n=1 all interventions, then repeats. GRPO = same matrix + ALGO=grpo (critic_warmup=0).

## Notes
- Metrics auto-logged per eval: token entropy, action entropy (probe block), state-action coverage
  + `unique_executed_actions_per_unique_text` (StateVisitation block), `toks_out`, milestone reach-rate
  (overcooked, 6 tiers), pass@k (`passk/exp_best_at_k`, `passk/solve_at_k` via delivered milestone,
  `passk/best_of_group_mean`) on shared-seed blocks, rewards/score.
- WANDB offline → `wandb sync wandb/<run>` after.
- Capture: eval trajectories in validation_data_dir; training rollouts in rollout_data_dir.
