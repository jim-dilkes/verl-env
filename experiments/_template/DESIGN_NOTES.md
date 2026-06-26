# AAAI Entropy/Exploration Harness — Design Notes

Running log of decisions/assumptions for the standardised harness (Ch3 → AAAI).
Branch `feat/aaai-harness`. Summarise to Jim at end.

## Claim being supported
Standard entropy/diversity interventions don't improve task performance in LLM-agent RL even
when they raise token entropy + state-action coverage — because ψ (tokens→action) is many-to-one,
so extra diversity is "noise-shaped not strategy-shaped." New metrics (per-tier reach-rate, pass@k)
turn "coverage up, reward flat" into "coverage up, task-progress flat, and best-of-group can't
rescue it either."

## What was already done (verified, no new code)
- **#2 two-block all-metrics eval ALREADY EXISTS** in `overcooked_evals_combined.yaml` +
  `snake_evals_combined.yaml`: an `*-Entropy-Check` block (`action_entropy.enabled=true,
  exclusive_metric=true`, n_samples=20) PLUS a `*-StateVisitation` block (no action_entropy →
  coverage logged), so token entropy + action entropy + coverage + `unique_executed_actions_per_unique_text`
  collapse metric all land. Confirmed gotcha real: `multi_env_evaluator.py:766-767`
  `track_standard_metrics = not(entropy_enabled and exclusive_metric)`.
- **#3 Overcooked per-tier reach-rate ALREADY BUILT** by the V2 milestone merge (pre-dates this plan).
  `overcooked/milestones.py` emits 6 tiers (hold_ingredient → ingredient_in_pot → pot_cooking →
  hold_dish → hold_soup → delivered); evaluator logs `milestone/{k}_{name}_reached` per eval (a
  reach-rate time-series), NOT gated by track_standard_metrics. `emit_milestones=True` by default
  in the wrapper → fires on every overcooked eval. DECISION: use the existing 6 tiers as-is
  (Jim confirmed) — finer than the plan's 4, zero new metric code.
- Entropy interventions (H sweep, clip_cov, kl_cov, adaptive, decay) all implemented & wired;
  config keys live under `actor_rollout_ref.actor.*`.

## New code written here
- **#4 pass@k / best-of-group** — `verl/trainer/ppo/eval_metrics.py` (new, tested):
  `best_of_group`, `expected_best_of_k` (unbiased continuous, brute-force-verified),
  `pass_at_k_binary` (Codex estimator), `compute_group_score_metrics`. Wired into the evaluator's
  `track_standard_metrics` block (`multi_env_evaluator.py:~1321`). DECISION (Jim): record EVERY
  variant, choose for paper later — logs `passk/best_of_group_mean`, `passk/exp_best_at_{1,2,4,8,16}`
  (continuous, on task score, falling back to reward), and `passk/solve_at_{1,2,4,8,16}` (binary,
  success = any-positive-reward = Overcooked delivery / Snake positive). Group = seed-group
  (rollouts sharing an initial env seed via `_compute_seed_sequence`), so pass@k = "k attempts at
  the SAME start state" — the meaningful diversity-collapse semantics. ks clamped to ≤ group_size.

## Intervention matrix (8 interventions; values verified from existing/archived scripts)
- `baseline`       : entropy_coeff=0.001, loss_mode=vanilla
- `H005/H01/H05`   : entropy_coeff=0.005/0.01/0.05, loss_mode=vanilla   (the H_sweep = 3 configs)
- `clip_cov`       : loss_mode=clip_cov, entropy_coeff=0.0, clip_cov_ratio=0.0002, lb=1.0, ub=5.0
- `kl_cov`         : loss_mode=kl_cov,  entropy_coeff=0.0, kl_cov_ratio=0.0002
- `adaptive`       : entropy_coeff=0.002, coeff_low=0.0005, coeff_high=0.02, coeff_lr=0.002,
                     entropy_low=0.5, entropy_high=0.7, entropy_top_p=0.33, loss_mode=vanilla
- `adaptive_decay` : adaptive + entropy_low_final=0.35, entropy_high_final=0.55 (cosine band decay)
DECISION: clip_cov/kl_cov set entropy_coeff=0.0 — they REPLACE the entropy bonus (cui_entropy_2025),
not additive (plan §). Adaptive values = canonical Ch3 run `experiments/snake/archive/260106_entropy_decay/`.

## Seeds (reproducibility)
- Set `data.seed=$SEED`, `data.shuffle=True`, `envs.group_initial_seed=$SEED`.
- CAVEAT: runs use a placeholder.parquet, so `data.seed` (dataloader sampler, main_ppo.py:488)
  is largely inert — the real trajectory RNG is `envs.group_initial_seed` (env init seeds). vLLM
  sampling may still inject nondeterminism; seed-reproducibility verified empirically in smoke (same
  SEED → near-identical early entropy). `run_number=$SEED` kept for naming only.

## Capture (durable, for qualitative noise-vs-strategy figure)
- `trainer.log_val_generations=20` (was 1).
- `trainer.validation_data_dir=<scratch>/val_dumps/<run>` — eval trajectory dumps (the figure source).
- `trainer.rollout_data_dir=<scratch>/rollout_dumps/<run>` — training rollouts. CAVEAT: large on disk
  over 600 steps × many runs; can be unset if /scratch tight. Kept per "never re-run" philosophy.

## sbatch template choices
- Based on `OC_PPO_4B_BL_2.sbatch` (preserves working vLLM/FSDP/dual-discount-PPO/rollout_correction).
- `--cpus-per-task=6` (min that clears Ray's 2-GPU placement group; was 32 → drained priority).
- `--partition=swarm_a100,swarm_h100`, `--time=5-00:00:00` (5-day max), 2 GPUs.
- Env-var driven via `--export`: {ALGO, ENV, INTERVENTION, SEED, MODEL}. ALGO=ppo (default gae);
  grpo adds `algorithm.adv_estimator=grpo` + `critic_warmup=0`. ENV ∈ {overcooked, snake(fastsnake)}.
- total_training_steps=600 (chapter-faithful; OC BL used 1000). test_freq=50 (richer time-series),
  save_freq=200. DECISION revisit if eval cost too high.
- WANDB offline (cluster) → needs `wandb sync` after.

## Review round 1 fixes (independent Opus reviewer)
- **P1-A (crash):** `adaptive`/`adaptive_decay` use `entropy_top_p=0.33`<1, which is incompatible with
  fused kernels (`dp_actor.py` raises). Fixed: those two interventions set `USE_FUSED_KERNELS=False`
  (a bash var so it actually wins over the template's default `True`).
- **P1-B (saturated metric):** binary `solve_at_k` was fed any-positive-reward, but under
  `shaped_reward=True` an onion pickup is positive ⇒ saturates ≈1, NOT delivery. Fixed: pass@k binary
  success now uses the **`delivered` milestone** (true sparse delivery) for overcooked; falls back to
  any-positive-reward only when milestones absent (snake).
- **P1-C (wrong groups):** pass@k is only meaningful for SAME-SEED groups. The 40-step reward blocks
  had no `seed_group_size` ⇒ distinct seeds ⇒ pass@k mixed start states. Fixed two ways: (1) evaluator
  now gates pass@k on `initial_seed is not None and 1 < seed_group_size < n_rollouts` (shared-seed
  groups only — distinct-seed blocks emit nothing misleading); (2) new `overcooked_evals_harness.yaml`
  gives the two 40-step task blocks `seed_group_size=10` (delivery pass@k, k≤8 @ eval temp 0.6),
  keeps StateVisitation (group 20, 8-step early-exploration pass@k), drops MA blocks (cost). Template
  points overcooked eval at `overcooked_evals_harness`.
- Confirmed-correct by reviewer (no action): pass@k math (brute-force verified), group ordering
  (i//seed_group_size matches `_compute_seed_sequence`), all Hydra keys valid, adaptive activation
  gates, two-block eval + milestone-on-baseline, `partner_policy=none` parses as string, bash safety.
- NOTE for paper: a higher-temp dedicated pass@k block (temp~1.0, larger groups) would strengthen the
  diversity-collapse argument; current delivery pass@k is at eval temp 0.6. Recorded for later.

## Review round 2 (converged — no major issues)
- All 3 P1s VERIFIED fixed by independent reviewer. No new P1.
- Minor: trailing symlink line tripped `set -e` on success → would mark a successful SLURM job FAILED;
  fixed (if-block + `exit 0`). Confirmed via login-node HYDRA_DRY compose (all keys resolve clean).
- DEFERRED (snake, secondary): snake reward blocks are greedy (`do_sample:False`) ⇒ naive pass@k is
  degenerate (k identical samples). Snake pass@k currently comes only from its stochastic
  StateVisitation block (gate handles it). A dedicated stochastic snake pass@k block is future work;
  snake stays on `snake_evals_combined`. Overcooked (priority) unaffected.

## Smoke result (login L4, Qwen3-0.6B, overcooked, val_before_train + 2 steps) — PASS
Exit 0, no errors. ALL metrics logged in ONE run (exclusive-metric gotcha avoided):
- token entropy `actor/entropy` (training); action entropy + `unique_executed_actions_per_unique_text`
  (Entropy-Check exclusive block); coverage `distinct_state_actions_valid_coverage` (StateVisitation);
  `toks_out_mean`, `rewards_mean`.
- milestone reach-rate `milestone/0..5_*_reached` (both reward + statevisitation blocks).
- pass@k `passk/{best_of_group_mean,exp_best_at_k,solve_at_k}` on shared-seed blocks (k≤4 at gsz=4 —
  clamp correct); CORRECTLY ABSENT on the exclusive Entropy-Check block (gating verified live).
- wandb `generations` table logged (qualitative trajectory capture).

### CAPTURE caveat (important)
`validation_data_dir` / `rollout_data_dir` have NO consumers in the multi-step trainer/evaluator —
setting them does nothing (dirs stay empty). They're only written by base verl `RayPPOTrainer`, which
this multi-step path does not use. The DURABLE capture is the **wandb OFFLINE run dir** (`wandb/offline-run-*`),
which persists all metrics + the eval generations table (log_val_generations=20) on disk for later
analysis (pull via `wandb sync` or read the local files). The qualitative noise-vs-strategy figure
sources from these generations. The dir flags are left in the template (harmless) but are inert; a
proper file-based eval-trajectory dump in multi_env_evaluator is possible future work if wandb is
insufficient.

## Self-review fix (ultrathink pass) — eval-config sampling shortcoming
FOUND: adding `seed_group_size=10` to the 40-step REWARD blocks (to enable pass@k) silently cut the
distinct evaluated start states from 50 → 5 (10 rollouts share each of 5 seeds), degrading the PRIMARY
`rewards_mean`/coverage estimate on those blocks. Clean task-reward (many distinct seeds) and pass@k
(few shared-seed groups, many repeats) are different sampling regimes and must be SEPARATE blocks.
FIX: reward blocks reverted to 50 DISTINCT seeds (clean reward/coverage); added a dedicated
`Overcooked-CrampedRoom-PassK` block (n_rollouts=64, seed_group_size=16 → 4 groups, 40-step, temp 1.25,
stochastic) for full-horizon delivery pass@k (k up to 16). StateVisitation still gives 8-step pass@k.
Verified clean (3 of 4 other audit items): clip_ratio default=0.2 (PPO spec, no override needed);
group_initial_seed controls TRAINING env seeding (ray_multistep_trainer.py:412-420), not eval-only;
seed grouping survives batched eval (batch_seeds=all_seeds[batch_start:batch_end] → i//gsz holds).

## Open / to verify in smoke
- coverage + entropy + passk + milestones all log in ONE run (exclusive-metric not dropping coverage).
- validation_data_dir actually written; generations table has 20 samples.
- same-SEED reproducibility of early entropy.
