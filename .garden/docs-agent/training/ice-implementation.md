# ICE (Instruction-Conditioned Exploration)

## What It Does
Appends a random "focus instruction" per rollout during generation, then uses parallel teacher/student optimisation: the teacher sees instructed prompts, the student sees base prompts, and both contribute to the policy gradient. Optional KL terms distill knowledge between teacher and student.

## Code Locations
- **Registry:** `verl/envs/environments/focus_instructions.py` — `has_focus_instructions()`, `get_focus_instructions()`, `has_ice_instructions()`, `get_ice_instructions()`, `sample_focus_for_episode()`, `inject_focus_into_obs()`
- **Config:** `verl/trainer/config/prompt/overcooked.yaml` → `prompt.prompt.ice.*`
- **Trainer integration:** `verl/trainer/ppo/ray_multistep_trainer.py`
  - ICE config read + sampling: ~L1266-1298 (before episode loop)
  - Dual tokenize: ~L1314-1319 (inside `text_gen_proc` timer)
  - `swap_all_instructed_to_base()`: top-level function ~L203-241
  - Teacher data prep + swap + metrics: ~L1646-1695 (after rollout loop, before `compute_log_prob`)
  - **Validation injection:** `_validate()` — samples focus + injects before `apply_chat_template`
- **Actor dual forward pass:** `verl/workers/actor/dp_actor.py`
  - ICE key selection + config extraction: ~L534-548
  - Batch sorting (non-instructed first): ~L561-564
  - Dual forward pass + combined loss: micro-batch loop ICE branch
- **Evaluator integration:** `verl/trainer/ppo/multi_env_evaluator.py`
  - ICE setup in `_evaluate_single_env_body()` — checks `inherit_ice` + `ice.enabled`

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
5. **Pass config:** `ice_alpha`, `ice_kl_beta_teacher`, `ice_kl_beta_student` via `meta_info`
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
prompt.prompt.ice:
  enabled: false          # master toggle
  source: "specific"      # "specific" (env instructions + template) or "generic" (standalone principles)
  assignment: stochastic  # stochastic | deterministic (covering: n_duplicates each + n_no_instruction)
  no_supplement_prob: 0.125 # stochastic mode only. No auto-compute — must be explicit.
  n_duplicates: 1         # deterministic mode: copies of EACH instruction per group (n_rollouts)
  n_no_instruction: 3     # deterministic mode: unconditioned samples per group
  alpha: 0.5              # teacher/student PG weighting: α·L_teacher + (1-α)·L_student (Asymmetric-RL/SD: 1.0)
  kl_beta_teacher: 0.0    # coefficient for D_KL(π^T || sg(π^S)) — teacher→student
  kl_beta_student: 0.0    # coefficient for D_KL(sg(π^T) || π^S) — student→teacher
  kl_estimator: k3        # k3 (Schulman per-token) | mean_logprob (paper-faithful, STUDENT KL only)
                          # mean_logprob + kl_beta_teacher>0 is rejected: the mean-logprob teacher
                          # (reverse-KL) gradient is wrong-direction; use k3 for teacher-side KL.
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
  passed to actor as per-sample `ice_kl_filter_mask`; actor applies it to KL_S only.
- **assignment=deterministic** (`assign_focus_deterministic`): exactly n_duplicates of each
  instruction + n_no_instruction unconditioned per group, shuffled per training step.
  Requires `n_instructions*n_duplicates + n_no_instruction == n_rollouts`.

### Instruction Sources
- **`specific`**: Environment-specific instructions from `FOCUS_REGISTRY` (e.g., Overcooked [How to Cook] steps). Wrapped in deliberative `template`.
- **`generic`**: `GENERIC_FOCUS_INSTRUCTIONS` — 10 meta-cognitive/strategy-level principles that work across any environment. No template wrapping (passthrough `{STEP_TEXT}`).

Unified retrieval: `get_ice_instructions(env_name, source)` / `has_ice_instructions(env_name, source)`. Generic always available; specific requires env registration.

### Adaptive Supplement Ratio
```yaml
prompt.prompt.ice.adaptive:
  enabled: false
  supplement_min: 0.1
  supplement_max: 0.9
  window_size: 10
  k: 5.0
```
- **Code:** `verl/trainer/ppo/adaptive_ice.py` — `AdaptiveICE` class
- Same sliding-window + slope + sigmoid pattern as `AdaptiveEpsilon`
- **Input signal:** `reward/base_mean` (base episodes only, no focus instructions)
- Improving rewards → low supplement_prob (consolidate); stagnating → high (explore)

### Reward Split Metrics
- Split `episode_returns` by `focus_per_rollout` (generation context)
- `reward/base_mean`, `reward/base_std` — rollouts with no focus (None)
- `reward/ice_mean`, `reward/ice_std` — rollouts with focus instruction
- `reward/internalization_gap` — `ice_mean - base_mean`

### WandB Metrics
- `ice/supplement_rate` — fraction of rollouts with focus
- `ice/unique_instructions` — count of distinct focus instructions
- `ice/has_instruction_rate` — fraction of rollouts that are instructed
- `ice/teacher_pg_loss` — teacher policy gradient loss
- `ice/student_pg_loss` — student policy gradient loss
- `ice/alpha` — teacher/student weighting
- `ice/kl_teacher` — teacher-branch KL (when kl_beta_teacher > 0)
- `ice/kl_student` — student-branch KL (when kl_beta_student > 0)
- `ice/kl_student_keep_frac` — (mean_logprob path) fraction of instructed samples kept by the KL_S filter, per micro-batch
- `ice/kl_filter_keep_rate` — (trainer) fraction of instructed episodes selected by kl_filter
- `ice/teacher_*` — teacher-specific PG sub-metrics
- `ice/student_*` — student-specific PG sub-metrics

## Validation & Evaluation Injection

### Validation (`_validate()`)
Injects focus when `ice.enabled=True`, UNLESS `ice.eval_unconditioned=true` (then no
injection — measures the deployed unconditioned student, e.g. for Asymmetric-RL/SD).
For tracking BOTH with- and without-focus performance, use the split eval specs in
`multi_env_evaluator.py` (`inherit_ice` + `ice_proportion`).

### Evaluation (`multi_env_evaluator.py`)
Per-env opt-in via `inherit_ice: true` in eval environment config. Default is `false`.

**`ice_proportion` override:** Per-eval-env param controlling fraction of rollouts WITH focus.

Guard chain: `inherit_ice=true` AND `ice.enabled=true` AND `has_ice_instructions(env_name, source)`.

### Split Eval Pattern (Internalization Measurement)
Paired eval blocks measure **internalization gap**: how much diversity is "rented" (context-dependent) vs "owned" (learned).

## Adding Focus Instructions for New Environments
1. Add instruction list to `FOCUS_REGISTRY` in `focus_instructions.py`
2. Key must match `config.envs.env_name` (lowercased)
3. Add `ice:` section to the environment's prompt YAML
4. For `source=generic`, no env registration needed

## Gotchas
- `ice_enabled` must be initialized OUTSIDE the `if self.global_steps == 1 or ...` block (critic warmup scoping)
- Focus is sampled once per episode, not per step
- `swap_all_instructed_to_base` forces `bypass_recomputing_logprobs = False` since tokens changed
- Template uses `{STEP_TEXT}` placeholder (not f-string)
- `no_supplement_prob` is REQUIRED when `ice.enabled=true`
- **Memory:** Two forward passes per micro-batch ≈ 2x activation memory. May need to halve `ppo_micro_batch_size_per_gpu` for ICE+KL runs.
- KL terms use `kl_penalty_forward` with `'k3'` (Schulman approximation from `core_algos.py`).
  **Argument order matters:** k3 estimates `KL(A||B)` for samples from `A` (first arg).
  Samples are teacher rollouts, so the first arg must be the (detached) teacher.
  KL_T = `kl_penalty_forward(teacher, student.detach())` (grad→teacher); KL_S =
  `kl_penalty_forward(teacher.detach(), student)` (grad→student, bounded). Swapping KL_S's
  args estimates the wrong divergence with an exploding gradient when π_S≪π_T.
- **No dynamic batching / balancing under ICE:** the dual forward sizes micro-batches from
  the base prompt but the teacher forward uses the longer instructed prompt. `dp_actor`
  raises if `actor.use_dynamic_bsz=True` with ICE; keep `trainer.balance_batch=False` too.
- **Interpreting `ice/student_*` clip metrics at 0<α<1:** the student ratio π_S/π_T_old
  is a cross-policy IS ratio, NOT a proximal (own-old) ratio, so PPO clipping bounds
  student-vs-teacher divergence rather than the student's step size. Large
  `ice/student_ppo_kl` / `pg_clipfrac` and some response-length growth are EXPECTED for
  α<1 (the paper notes this; it motivates the KL terms) — not an implementation defect.
  Moot at α=1 (student weight 0).
- **Student & teacher PG share the `teacher_old_log_probs` anchor.** Both terms are
  estimated on teacher rollouts, so the PPO behaviour anchor is the old teacher policy:
  teacher ratio = π_T/π_T_old, student ratio = π_S/π_T_old (= the paper's IS-corrected
  `J_S^R = E_{π_T}[(π_S/sg[π_T])·R]`). For non-instructed rows teacher_old == base
  old_log_probs by construction, so they're unaffected. Neither term applies
  `rollout_is_weights` (those are base-context, wrong for a teacher anchor).
- Non-instructed samples: teacher == student forward passes produce identical outputs, so KL ≈ 0 and PG losses are equal. Batch sorting puts these first to skip redundant teacher forward.

## Tests
- `tests/test_ice_focus_instructions.py` — registry, sampling, injection [P1]
- `tests/test_ice_dual_tokenize.py` — tokenization shape/content consistency [P0] (requires model)
- `tests/test_ice_swap.py` — swap correctness + gold standard [P0]
- `tests/test_ice_edge_cases.py` — single rollout, zero episode, empty response [P1]
- `tests/test_ice_distill.py` — mean_logprob KL (values/grad directions), kl_filter modes, deterministic assignment [P0]
- `experiments/overcooked/test_ice_login_node.sh` — parallel opt without KL (α=0.5, β=0)
- `experiments/overcooked/test_ice_kl_login_node.sh` — parallel opt with KL (α=0.5, β=0.01)
- `experiments/overcooked/test_ice_asymmetric_login_node.sh` — Asymmetric-RL/SD (α=1, mean_logprob KL_S, return_positive filter, eval_unconditioned)

Run with: `PYTHONPATH=. python tests/test_ice_swap.py`

### Advantage-weighted KL_S (AWR-style) — IMPLEMENTED (V2)
Weights the per-sample KL_S by the teacher rollout's advantage (positive, exp-of-z-score),
a *soft* generalization of the hard `kl_filter`. Config: `ice.kl_weight: none|awr`,
`ice.kl_weight_temp` (default 1.0), `ice.kl_weight_cap` (null). Only the `mean_logprob` path.
- **`compute_awr_weights`** (`distill_kl.py`): `A_i` = z-scored `episode_return` over the
  INSTRUCTED episodes; `w_i = exp(A_i/τ)`, optional cap. Always > 0 — signed advantage is
  used only inside `exp`, so KL_S is never multiplied by a negative coefficient (that would
  be unbounded anti-distillation; see below). `τ→∞` uniform, `τ→0` only-best.
- **Trainer** computes per-episode weights, broadcasts to a per-sample `ice_kl_weight`
  (absent / `none` ⇒ uniform); **actor** applies them via a *weighted* `masked_sample_mean`
  (`sum(w·keep·kl)/sum(w·keep)`) — a convex combination, so the loss SCALE is preserved (no
  separate mean-normalization needed). Orthogonal to `kl_filter`: the filter selects the kept
  set, AWR re-weights within it. Metrics: `ice/kl_weight_mean`, `ice/kl_weight_max_batch`
  (trainer, over instructed episodes), `ice/kl_weight_max` (actor, per micro-batch kept).
  Requires `kl_estimator=mean_logprob` (k3 ignores weights → rejected). `temp>0` and
  `cap>0|null` are enforced; the `exp` argument is clamped to avoid overflow at tiny `temp`.
- **Why not raw signed GRPO advantages:** KL_S = `mean_t(sg[logπ_T] − logπ_S)`; minimizing
  raises `logπ_S` on teacher tokens (bounded, logπ_S ≤ 0). A *negative* weight flips it to
  lower `logπ_S` — unbounded below → anti-distillation collapse. AWR's `exp(A/τ)` keeps
  weights strictly positive.

## Future work (V2)

### k3 teacher KL is also a heuristic (β_T>0 exploratory)
k3-on-teacher's autograd gives only the pathwise `1 − π_S/π_T` gradient; a faithful
reverse-KL `D_KL(π_T‖sg[π_S])` gradient on self-sampled actions also needs the
score-function (REINFORCE) term. Directionally sensible but not unbiased. The paper's
Asymmetric-RL/SD uses β_T=0, so this only matters for exploratory β_T>0 configs.

### Proper mean_logprob teacher (reverse-KL) estimator
Currently `kl_estimator=mean_logprob` is student-only (rejected with `kl_beta_teacher>0`);
teacher-side KL must use `k3`. A correct sample-based reverse-KL `D_KL(π_T‖sg[π_S])`
gradient needs the score-function (REINFORCE) term, since the sampling distribution `π_T`
depends on the teacher params. Add this if a paper-faithful teacher KL is wanted.

### Per-focus eval breakdown + per-trajectory milestone tracking
Pin each focus to its own eval condition (reuse `assign_focus_deterministic`); track which
ordered sub-task milestone each trajectory reaches (from env info / shaped-reward events).
