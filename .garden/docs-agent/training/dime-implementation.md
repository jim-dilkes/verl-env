# DIME (Diverse Instruction-Masked Exploration)

## What It Does
Appends a random "focus instruction" per rollout during generation, then uses parallel teacher/student optimisation: the teacher sees instructed prompts, the student sees base prompts, and both contribute to the policy gradient. Optional KL terms distill knowledge between teacher and student.

## Code Locations
- **Registry:** `verl/envs/environments/focus_instructions.py` — `has_focus_instructions()`, `get_focus_instructions()`, `has_dime_instructions()`, `get_dime_instructions()`, `sample_focus_for_episode()`, `inject_focus_into_obs()`
- **Config:** `verl/trainer/config/prompt/overcooked.yaml` → `prompt.prompt.dime.*`
- **Trainer integration:** `verl/trainer/ppo/ray_multistep_trainer.py`
  - DIME config read + sampling: ~L1266-1298 (before episode loop)
  - Dual tokenize: ~L1314-1319 (inside `text_gen_proc` timer)
  - `swap_all_instructed_to_base()`: top-level function ~L203-241
  - Teacher data prep + swap + metrics: ~L1646-1695 (after rollout loop, before `compute_log_prob`)
  - **Validation injection:** `_validate()` — samples focus + injects before `apply_chat_template`
- **Actor dual forward pass:** `verl/workers/actor/dp_actor.py`
  - DIME key selection + config extraction: ~L534-548
  - Batch sorting (non-instructed first): ~L561-564
  - Dual forward pass + combined loss: micro-batch loop DIME branch
- **Evaluator integration:** `verl/trainer/ppo/multi_env_evaluator.py`
  - DIME setup in `_evaluate_single_env_body()` — checks `inherit_dime` + `dime.enabled`

## How It Works

### Rollout (each step)
1. `inject_focus_into_obs(obs_vec, focus_per_rollout, template)` — deepcopy obs_vec, append focus text to last user message
2. Tokenize WITH focus → `input_obs` for generation
3. Tokenize WITHOUT focus → `base_input_obs`, stored in `base_prompt_tokens_by_step`
4. Generation uses focus-injected tokens

### Before Training (Trainer-side)
1. **Save teacher batch:** Clone `input_ids`, `attention_mask`, `position_ids` (with instructed prompts)
2. **Compute teacher old_log_probs:** Forward pass on instructed prompts → `teacher_old_log_probs`
3. **Swap to base:** `swap_all_instructed_to_base()` replaces all instructed prompts with base prompts
4. **Attach to batch:** `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`, `teacher_old_log_probs`, `has_instruction`
5. **Pass config:** `dime_alpha`, `dime_kl_beta_teacher`, `dime_kl_beta_student` via `meta_info`
6. Standard recompute of `old_log_probs` produces student anchors on base prompts

### Actor-side Dual Forward Pass
In `dp_actor.py` `update_policy`:

1. **Sort batch:** Non-instructed samples sorted first so early micro-batches can skip redundant teacher forward pass
2. **Student forward:** Standard forward on base prompts (current `input_ids`)
3. **Teacher forward:** Forward on instructed prompts (from `teacher_input_ids`), skipped if micro-batch has no instructed samples
4. **Combined PG loss:** `pg_loss = α * teacher_pg_loss + (1-α) * student_pg_loss`
   - Non-instructed samples: teacher == student (same conditioning), so contribution is just the standard PG loss
5. **KL terms** (optional, masked to instructed samples only):
   - Teacher KL: `D_KL(π^T || sg(π^S))` — gradient through teacher, constrains teacher toward student
   - Student KL: `D_KL(sg(π^T) || π^S)` — gradient through student, pulls student toward teacher
   - Uses `kl_penalty_forward(..., 'k3')` (Schulman low-variance approximation)

### Batch Layout
`[step0_env0, step0_env1, ..., step0_envN, step1_env0, ..., stepE_envN]`
- `sample_idx = step * n_rollouts + env`

## Config Parameters
```yaml
prompt.prompt.dime:
  enabled: false          # master toggle
  source: "specific"      # "specific" (env instructions + template) or "generic" (standalone principles)
  assignment: stochastic  # stochastic | deterministic (covering: n_duplicates each + n_no_instruction)
  no_supplement_prob: 0.125 # stochastic mode only. No auto-compute — must be explicit.
  n_duplicates: 1         # deterministic mode: copies of EACH instruction per group (n_rollouts)
  n_no_instruction: 3     # deterministic mode: unconditioned samples per group
  alpha: 0.5              # teacher/student PG weighting: α·L_teacher + (1-α)·L_student (Asymmetric-RL/SD: 1.0)
  kl_beta_teacher: 0.0    # coefficient for D_KL(π^T || sg(π^S)) — teacher→student
  kl_beta_student: 0.0    # coefficient for D_KL(sg(π^T) || π^S) — student→teacher
  kl_estimator: k3        # k3 (Schulman per-token) | mean_logprob (paper-faithful per-sample logprob diff)
  kl_filter: none         # KL_S target filter: none | return_positive | top_pct
  kl_filter_top_pct: 0.5  # top_pct: fraction of instructed episodes (by return) kept
  eval_unconditioned: false # inline _validate() skips focus injection (measure deployed student)
  template: '...'         # focus instruction template with {STEP_TEXT} placeholder
```

### Asymmetric-RL/SD variation (paper-faithful)
The EMNLP-26 ICE π-distill "Asymmetric-RL/SD" config = `alpha=1.0` (teacher-only RL),
`kl_beta_student>0`, `kl_beta_teacher=0`, `kl_estimator=mean_logprob`, and a KL_S filter
(`return_positive` / `top_pct`). The student forward still runs at α=1 (needed for KL_S);
only the (1-α) student PG term vanishes.
- **kl_estimator=mean_logprob** (`verl/trainer/ppo/distill_kl.py`): per-sample mean over
  response tokens of `sg(logπ_T) − logπ_S` (KL_S) / `logπ_T − sg(logπ_S)` (KL_T). Student &
  teacher share response tokens + response_mask in dp_actor, so means are taken over one
  shared index set. `k3` (branch default) is preserved as an option.
- **kl_filter** restricts KL_S to selected teacher rollouts (whole episodes, broadcast to
  steps). Keep-set computed in trainer (`compute_kl_filter_keep`, has episode returns),
  passed to actor as per-sample `dime_kl_filter_mask`; actor applies it to KL_S only.
- **assignment=deterministic** (`assign_focus_deterministic`): exactly n_duplicates of each
  instruction + n_no_instruction unconditioned per group, shuffled per training step.
  Requires `n_instructions*n_duplicates + n_no_instruction == n_rollouts`.

### Instruction Sources
- **`specific`**: Environment-specific instructions from `FOCUS_REGISTRY` (e.g., Overcooked [How to Cook] steps). Wrapped in deliberative `template`.
- **`generic`**: `GENERIC_FOCUS_INSTRUCTIONS` — 10 meta-cognitive/strategy-level principles that work across any environment. No template wrapping (passthrough `{STEP_TEXT}`).

Unified retrieval: `get_dime_instructions(env_name, source)` / `has_dime_instructions(env_name, source)`. Generic always available; specific requires env registration.

### Adaptive Supplement Ratio
```yaml
prompt.prompt.dime.adaptive:
  enabled: false
  supplement_min: 0.1
  supplement_max: 0.9
  window_size: 10
  k: 5.0
```
- **Code:** `verl/trainer/ppo/adaptive_dime.py` — `AdaptiveDIME` class
- Same sliding-window + slope + sigmoid pattern as `AdaptiveEpsilon`
- **Input signal:** `reward/base_mean` (base episodes only, no focus instructions)
- Improving rewards → low supplement_prob (consolidate); stagnating → high (explore)

### Reward Split Metrics
- Split `episode_returns` by `focus_per_rollout` (generation context)
- `reward/base_mean`, `reward/base_std` — rollouts with no focus (None)
- `reward/dime_mean`, `reward/dime_std` — rollouts with focus instruction
- `reward/internalization_gap` — `dime_mean - base_mean`

### WandB Metrics
- `dime/supplement_rate` — fraction of rollouts with focus
- `dime/unique_instructions` — count of distinct focus instructions
- `dime/has_instruction_rate` — fraction of rollouts that are instructed
- `dime/teacher_pg_loss` — teacher policy gradient loss
- `dime/student_pg_loss` — student policy gradient loss
- `dime/alpha` — teacher/student weighting
- `dime/kl_teacher` — teacher-branch KL (when kl_beta_teacher > 0)
- `dime/kl_student` — student-branch KL (when kl_beta_student > 0)
- `dime/kl_student_keep_frac` — (mean_logprob path) fraction of instructed samples kept by the KL_S filter, per micro-batch
- `dime/kl_filter_keep_rate` — (trainer) fraction of instructed episodes selected by kl_filter
- `dime/teacher_*` — teacher-specific PG sub-metrics
- `dime/student_*` — student-specific PG sub-metrics

## Validation & Evaluation Injection

### Validation (`_validate()`)
Injects focus when `dime.enabled=True`, UNLESS `dime.eval_unconditioned=true` (then no
injection — measures the deployed unconditioned student, e.g. for Asymmetric-RL/SD).
For tracking BOTH with- and without-focus performance, use the split eval specs in
`multi_env_evaluator.py` (`inherit_dime` + `dime_proportion`).

### Evaluation (`multi_env_evaluator.py`)
Per-env opt-in via `inherit_dime: true` in eval environment config. Default is `false`.

**`dime_proportion` override:** Per-eval-env param controlling fraction of rollouts WITH focus.

Guard chain: `inherit_dime=true` AND `dime.enabled=true` AND `has_dime_instructions(env_name, source)`.

### Split Eval Pattern (Internalization Measurement)
Paired eval blocks measure **internalization gap**: how much diversity is "rented" (context-dependent) vs "owned" (learned).

## Adding Focus Instructions for New Environments
1. Add instruction list to `FOCUS_REGISTRY` in `focus_instructions.py`
2. Key must match `config.envs.env_name` (lowercased)
3. Add `dime:` section to the environment's prompt YAML
4. For `source=generic`, no env registration needed

## Gotchas
- `dime_enabled` must be initialized OUTSIDE the `if self.global_steps == 1 or ...` block (critic warmup scoping)
- Focus is sampled once per episode, not per step
- `swap_all_instructed_to_base` forces `bypass_recomputing_logprobs = False` since tokens changed
- Template uses `{STEP_TEXT}` placeholder (not f-string)
- `no_supplement_prob` is REQUIRED when `dime.enabled=true`
- **Memory:** Two forward passes per micro-batch ≈ 2x activation memory. May need to halve `ppo_micro_batch_size_per_gpu` for DIME+KL runs.
- KL terms use `kl_penalty_forward` with `'k3'` (Schulman approximation from `core_algos.py`)
- Non-instructed samples: teacher == student forward passes produce identical outputs, so KL ≈ 0 and PG losses are equal. Batch sorting puts these first to skip redundant teacher forward.

## Tests
- `tests/test_dime_focus_instructions.py` — registry, sampling, injection [P1]
- `tests/test_dime_dual_tokenize.py` — tokenization shape/content consistency [P0] (requires model)
- `tests/test_dime_swap.py` — swap correctness + gold standard [P0]
- `tests/test_dime_edge_cases.py` — single rollout, zero episode, empty response [P1]
- `tests/test_dime_distill.py` — mean_logprob KL (values/grad directions), kl_filter modes, deterministic assignment [P0]
- `experiments/overcooked/test_dime_login_node.sh` — parallel opt without KL (α=0.5, β=0)
- `experiments/overcooked/test_dime_kl_login_node.sh` — parallel opt with KL (α=0.5, β=0.01)
- `experiments/overcooked/test_dime_asymmetric_login_node.sh` — Asymmetric-RL/SD (α=1, mean_logprob KL_S, return_positive filter, eval_unconditioned)

Run with: `PYTHONPATH=. python tests/test_dime_swap.py`

## Future work (V2)

### Advantage-weighted KL_S (AWR-style)
Weight the per-sample KL_S distillation term by the teacher rollout's advantage so the
student is pulled harder toward high-advantage teacher trajectories — a *soft*
generalization of the hard `kl_filter`.

- **Do NOT use raw signed (GRPO) advantages.** KL_S = `mean_t(sg[logπ_T] − logπ_S)`;
  minimizing raises `logπ_S` on teacher tokens (bounded, logπ_S ≤ 0). A *negative* weight
  flips the gradient to lower `logπ_S`, which is **unbounded below** (cross-entropy → +∞):
  the student drives those actions' probability toward 0 with no minimum → anti-distillation
  collapse / instability (not clipped like the PPO surrogate).
- **Use positive, mean-normalized weights** (Advantage-Weighted Regression / RWR):
  `w_i = N·softmax(A_i/τ)` → always >0, mean ≈ 1, preserves overall KL_S magnitude while
  reallocating emphasis. `τ→∞` = uniform (current); `τ→0` = only the best rollout. The hard
  `kl_filter` (top_pct/return_positive) is a hard special case of this soft weighting.
- **Advantage source (multi-step):** per-episode group-normalized `episode_returns` (same
  basis as the filter's per-episode keep), broadcast to steps. Compute in the trainer, pass
  per-sample (like `dime_kl_filter_mask`), apply as a weighted mean in the actor's
  `mean_logprob` path (extend `masked_sample_mean`). Sketch: `dime.kl_weight: none|awr`,
  `dime.kl_weight_temp: <τ>`; consider capping max weight.

### Per-focus eval breakdown + per-trajectory milestone tracking
Pin each focus to its own eval condition (reuse `assign_focus_deterministic`); track which
ordered sub-task milestone each trajectory reaches (from env info / shaped-reward events).
