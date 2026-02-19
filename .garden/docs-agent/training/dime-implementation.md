# DIME (Diverse Instruction-Masked Exploration)

## What It Does
Appends a random "focus instruction" per rollout during generation, strips it before the training log_prob pass. Model learns focus-guided behaviors without depending on the instruction at inference.

## Code Locations
- **Registry:** `verl/envs/environments/focus_instructions.py` — `has_focus_instructions()`, `get_focus_instructions()`, `has_dime_instructions()`, `get_dime_instructions()`, `sample_focus_for_episode()`, `sample_mask_decisions()`, `inject_focus_into_obs()`
- **Config:** `verl/trainer/config/prompt/overcooked.yaml` → `prompt.prompt.dime.*`
- **Trainer integration:** `verl/trainer/ppo/ray_multistep_trainer.py`
  - DIME config read + sampling: ~L1196-1216 (before episode loop)
  - Dual tokenize: ~L1229-1260 (inside `text_gen_proc` timer)
  - `swap_dime_prompts()`: top-level function ~L199-231
  - Prompt swap call + metrics: ~L1544-1555 (after rollout loop, before `compute_log_prob`)
  - **Validation injection:** `_validate()` — samples focus + injects before `apply_chat_template` (mirrors training generation)
- **Evaluator integration:** `verl/trainer/ppo/multi_env_evaluator.py`
  - DIME setup in `_evaluate_single_env_body()` — checks `inherit_dime` + `dime.enabled` + `has_focus_instructions()`
  - Per-batch focus sampling after `vec_envs.reset()`
  - Focus injection before `apply_chat_template()` in step loop

## How It Works

### Rollout (each step)
1. `inject_focus_into_obs(obs_vec, focus_per_rollout, template)` — deepcopy obs_vec, append focus text to last user message
2. Tokenize WITH focus → `input_obs` for generation
3. Tokenize WITHOUT focus → `base_input_obs`, stored in `base_prompt_tokens_by_step`
4. Generation uses focus-injected tokens

### Before Training
`swap_dime_prompts(batch, base_prompt_tokens_by_step, n_rollouts, episode_len, rlen, mask_per_rollout)`:
- For each sample where `mask_per_rollout[env_idx]` is True: replace `input_ids[:plen]` with base prompt tokens
- Unmasked rollouts keep their focus-injected prompt for training (context-conditional learning)
- Keep `input_ids[-rlen:]` (response) identical
- Rebuild `attention_mask` and `position_ids`
- `mask_per_rollout=None` swaps all (backwards compatible)

### Batch Layout
`[step0_env0, step0_env1, ..., step0_envN, step1_env0, ..., stepE_envN]`
- `sample_idx = step * n_rollouts + env`

## Config Parameters
```yaml
prompt.prompt.dime:
  enabled: false          # master toggle
  source: "specific"      # "specific" (env instructions + template) or "generic" (standalone principles)
  mask_for_training: true # false = control (focus visible in training too)
  mask_probability: 1.0   # probability of masking per focus-injected rollout (1.0=always, 0.0=never, 0.5=half-half)
  no_supplement_prob: 0.125 # REQUIRED when enabled. No auto-compute — must be explicit.
  template: 'Pay particular attention to this aspect of the task: "{STEP_TEXT}". Consider how it could apply in the current situation before choosing your action.'
  diagnostics: false      # Enable prompt conditioning diagnostics (adds 1 extra forward pass per step)
```

### Instruction Sources
- **`specific`**: Environment-specific instructions from `FOCUS_REGISTRY` (e.g., Overcooked [How to Cook] steps). Wrapped in deliberative `template`.
- **`generic`**: `GENERIC_FOCUS_INSTRUCTIONS` — 10 meta-cognitive/strategy-level principles that work across any environment. No template wrapping (passthrough `{STEP_TEXT}`).

Unified retrieval: `get_dime_instructions(env_name, source)` / `has_dime_instructions(env_name, source)`. Generic always available; specific requires env registration.

### Adaptive Supplement Ratio
```yaml
prompt.prompt.dime.adaptive:
  enabled: false
  supplement_min: 0.1    # minimum fraction getting focus (even when improving)
  supplement_max: 0.9    # maximum fraction getting focus (when stuck)
  window_size: 10
  k: 5.0
```
- **Code:** `verl/trainer/ppo/adaptive_dime.py` — `AdaptiveDIME` class
- Same sliding-window + slope + sigmoid pattern as `AdaptiveEpsilon`
- Reuses `_std`, `_sigmoid` helpers from `adaptive_epsilon.py`
- **Input signal:** `reward/base_mean` (base episodes only, no focus instructions)
- Avoids feedback contamination: supplement ratio does not influence its own input
- Update skipped when no base episodes in batch (window not polluted with zeros)
- Improving rewards → low supplement_prob (consolidate); stagnating → high (explore)
- Overrides `no_supplement_prob` when enabled; no env pipe needed (used directly in trainer)
- Init in `__init__` (~L498); update after metrics (~L1842) using `reward/base_mean`; override in DIME setup (~L1267)
- Metrics: `dime/adaptive_supplement_prob`, `dime/adaptive_slope`, `dime/adaptive_buffer_fill`, `dime/adaptive_update_skipped` (when no base episodes)

### Probabilistic Training Mask
- `mask_probability` (default 1.0): per-rollout Bernoulli decision on whether to strip focus for training
- `sample_mask_decisions(focus_per_rollout, mask_probability)` → `list[bool]` (True=mask, False=keep)
- Only focus-injected rollouts (focus != None) are candidates; no-focus rollouts always False
- `dime/mask_rate` metric: fraction of focus-injected rollouts that were masked (denominator = supplement count)
- `mask_probability=0.5` enables dual gradient signal: half internalization (base prompt), half context-conditional (focus prompt)

### Reward Split Metrics
- Split `episode_returns` by `focus_per_rollout` (generation context, not training mask)
- `reward/base_mean`, `reward/base_std` — rollouts with no focus (None)
- `reward/dime_mean`, `reward/dime_std` — rollouts with focus instruction
- `reward/internalization_gap` — `dime_mean - base_mean` (shrinking = model internalizing strategies)
- Only logged when `dime_enabled` and respective condition has rollouts
- Computed at ~L1605 after `episode_returns` tensor, before training pass

## Validation & Evaluation Injection

### Validation (`_validate()`)
Always injects focus when `dime.enabled=True`. Mirrors training generation conditions exactly. No dual tokenization needed (val doesn't compute gradients).

### Evaluation (`multi_env_evaluator.py`)
Per-env opt-in via `inherit_dime: true` in eval environment config. Default is `false` (bare prompts for deployment-style evals).

**`dime_proportion` override:** Per-eval-env param controlling fraction of rollouts WITH focus. Overrides training's `no_supplement_prob` for this eval only. `eval_dime_no_supp = 1.0 - dime_proportion`. Silently ignored if `inherit_dime` is false or DIME not active.

```yaml
environments:
  - name: "OC-CrampedRoom"           # deployment eval — bare prompts
    env_name: overcooked
  - name: "OC-Entropy"               # base diagnostic — no focus (internalization)
    env_name: overcooked
    inherit_training_multiaction: true
    action_entropy: { enabled: true, ... }
  - name: "OC-Entropy-Ctx"           # context-augmented diagnostic (ceiling)
    env_name: overcooked
    inherit_dime: true
    dime_proportion: 0.8             # 80% with focus, 20% clean
    inherit_training_multiaction: true
    action_entropy: { enabled: true, ... }
```

Guard chain: `inherit_dime=true` AND `dime.enabled=true` AND `has_dime_instructions(env_name, source)`. If any false, bare prompts used. Debug log emitted when `inherit_dime=true` but DIME not active.

Focus sampled once per batch (same instruction for entire episode per rollout). Entropy probes see focus-injected `val_input_obs_text` naturally.

### Split Eval Pattern (Internalization Measurement)
Paired eval blocks measure **internalization gap**: how much diversity is "rented" (context-dependent) vs "owned" (learned).

- **Base evals** (`-Entropy-Check`, `-StateVisitation`): no `inherit_dime`, no focus injection. Measures what model actually learned.
- **Context-augmented evals** (`-Entropy-Check-Ctx`, `-StateVisitation-Ctx`): `inherit_dime: true` + `dime_proportion: 0.8`. Measures diversity ceiling with focus active.
- **Gap** = Ctx metric - Base metric. Shrinking gap over training = successful internalization.
- `-Ctx` suffix naming convention (short, WandB-friendly)

Config files:
- `overcooked_evals_dime_split.yaml` — full split evals (deployment + base diagnostic + ctx diagnostic)
- `snake_evals_dime_split.yaml` — same pattern for FastSnake
- `*_minimal.yaml` variants — tiny rollout counts for login node smoke tests

Only diagnostic evals (entropy + state visitation) get split. Deployment evals test task performance, not diversity.

## Adding Focus Instructions for New Environments
1. Add instruction list to `FOCUS_REGISTRY` in `focus_instructions.py`
2. Key must match `config.envs.env_name` (lowercased)
3. Add `dime:` section to the environment's prompt YAML
4. For `source=generic`, no env registration needed — works with any environment

## Gotchas
- `dime_enabled` must be initialized OUTSIDE the `if self.global_steps == 1 or ...` block (critic warmup scoping)
- Focus is sampled once per episode, not per step
- `swap_dime_prompts` forces `bypass_recomputing_logprobs = False` since tokens changed (skipped entirely if `any(mask_per_rollout)` is False)
- Template uses `{STEP_TEXT}` placeholder (not f-string)
- Focus instructions must match YAML `environment_instruction` terminology exactly (e.g., "meal" not "soup")
- `no_supplement_prob` is REQUIRED when `dime.enabled=true` — raises ValueError if null/missing
- Overcooked has 7 focus instructions (matching 7 steps in YAML [How to Cook])

## Prompt Conditioning Diagnostics

When `dime.diagnostics: true`, an extra forward pass computes logprobs on the **focus-injected** prompts before swap. After swap + recompute on base prompts, the per-token difference is logged:

**Metrics (all W&B):**
- `dime/prompt_kl_mean` — Mean |log π(y_t|x_focus) - log π(y_t|x_base)| across response tokens. Higher = focus instructions shift distribution more.
- `dime/prompt_kl_max` — Max per-token absolute log-ratio. Identifies extreme shifts.
- `dime/prompt_logprob_shift` — Signed mean log-ratio. Positive = focus generally increases token likelihood.
- `dime/prompt_kl_per_seq_mean` — Mean per-sequence sum of |log ratios|. Proxy for full-sequence KL.
- `dime/prompt_kl_per_seq_max` — Max per-sequence sum. Worst-case divergence.
- `dime/is_ratio_proxy_mean` — exp(|sum of signed log ratios per sequence|). Approximates what the importance sampling ratio magnitude would be.
- `dime/is_ratio_proxy_max` — Max IS ratio proxy. If >>10, importance sampling would be impractical.

**Interpretation:**
- `prompt_kl_mean` < 0.01: Focus instructions barely shift distribution. IS correction negligible.
- `prompt_kl_mean` 0.01-0.1: Moderate shift. IS correction meaningful but tractable.
- `prompt_kl_mean` > 0.1: Strong shift. IS ratios will be high, need aggressive clipping.
- `is_ratio_proxy_max` > 100: Off-policy IS would be impractical for these sequences.

**Cost:** One extra `compute_log_prob` forward pass per training step. Only for DIME-masked rollouts. Gate behind config flag for production runs.

## Tests
- `tests/test_dime_focus_instructions.py` — registry, sampling, injection [P1]
- `tests/test_dime_dual_tokenize.py` — tokenization shape/content consistency [P0] (requires model)
- `tests/test_dime_swap.py` — swap correctness + gold standard [P0]
- `tests/test_dime_edge_cases.py` — single rollout, zero episode, empty response [P1]

Run with: `PYTHONPATH=. python tests/test_dime_swap.py`
