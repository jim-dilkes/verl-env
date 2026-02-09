# DIME (Diverse Instruction-Masked Exploration)

## What It Does
Appends a random "focus instruction" per rollout during generation, strips it before the training log_prob pass. Model learns focus-guided behaviors without depending on the instruction at inference.

## Code Locations
- **Registry:** `verl/envs/environments/focus_instructions.py` — `has_focus_instructions()`, `get_focus_instructions()`, `has_dime_instructions()`, `get_dime_instructions()`, `sample_focus_for_episode()`, `inject_focus_into_obs()`
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
`swap_dime_prompts(batch, base_prompt_tokens_by_step, n_rollouts, episode_len, rlen)`:
- For each sample: replace `input_ids[:plen]` with base prompt tokens
- Keep `input_ids[-rlen:]` (response) identical
- Rebuild `attention_mask` and `position_ids`

### Batch Layout
`[step0_env0, step0_env1, ..., step0_envN, step1_env0, ..., stepE_envN]`
- `sample_idx = step * n_rollouts + env`

## Config Parameters
```yaml
prompt.prompt.dime:
  enabled: false          # master toggle
  source: "specific"      # "specific" (env instructions + template) or "generic" (standalone principles)
  mask_for_training: true # false = control (focus visible in training too)
  no_supplement_prob: null # auto: 1/(N+1) where N = len(instructions). Override with explicit float.
  template: 'Before acting, carefully consider...'  # used for source=specific; ignored for source=generic
```

### Instruction Sources
- **`specific`**: Environment-specific instructions from `FOCUS_REGISTRY` (e.g., Overcooked [How to Cook] steps). Wrapped in deliberative `template`.
- **`generic`**: `GENERIC_FOCUS_INSTRUCTIONS` — 7 meta-cognitive/strategy-level principles that work across any environment. No template wrapping (passthrough `{STEP_TEXT}`).

Unified retrieval: `get_dime_instructions(env_name, source)` / `has_dime_instructions(env_name, source)`. Generic always available; specific requires env registration.

## Validation & Evaluation Injection

### Validation (`_validate()`)
Always injects focus when `dime.enabled=True`. Mirrors training generation conditions exactly. No dual tokenization needed (val doesn't compute gradients).

### Evaluation (`multi_env_evaluator.py`)
Per-env opt-in via `inherit_dime: true` in eval environment config. Default is `false` (bare prompts for deployment-style evals).

```yaml
environments:
  - name: "OC-CrampedRoom"           # deployment eval — bare prompts
    env_name: overcooked
  - name: "OC-CrampedRoom-Entropy"   # training-diagnostic — focus injected
    env_name: overcooked
    inherit_dime: true
    action_entropy:
      enabled: true
```

Guard chain: `inherit_dime=true` AND `dime.enabled=true` AND `has_dime_instructions(env_name, source)`. If any false, bare prompts used. Debug log emitted when `inherit_dime=true` but DIME not active.

Focus sampled once per batch (same instruction for entire episode per rollout). Entropy probes see focus-injected `val_input_obs_text` naturally.

## Adding Focus Instructions for New Environments
1. Add instruction list to `FOCUS_REGISTRY` in `focus_instructions.py`
2. Key must match `config.envs.env_name` (lowercased)
3. Add `dime:` section to the environment's prompt YAML
4. For `source=generic`, no env registration needed — works with any environment

## Gotchas
- `dime_enabled` must be initialized OUTSIDE the `if self.global_steps == 1 or ...` block (critic warmup scoping)
- Focus is sampled once per episode, not per step
- `swap_dime_prompts` forces `bypass_recomputing_logprobs = False` since tokens changed
- Template uses `{STEP_TEXT}` placeholder (not f-string)
- Focus instructions must match YAML `environment_instruction` terminology exactly (e.g., "meal" not "soup")
- `no_supplement_prob: null` auto-computes as `1/(N+1)` in trainer; explicit float overrides
- Overcooked has 7 focus instructions (matching 7 steps in YAML [How to Cook])

## Tests
- `tests/test_dime_focus_instructions.py` — registry, sampling, injection [P1]
- `tests/test_dime_dual_tokenize.py` — tokenization shape/content consistency [P0] (requires model)
- `tests/test_dime_swap.py` — swap correctness + gold standard [P0]
- `tests/test_dime_edge_cases.py` — single rollout, zero episode, empty response [P1]

Run with: `PYTHONPATH=. python tests/test_dime_swap.py`
