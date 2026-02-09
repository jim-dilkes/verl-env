# DIME (Diverse Instruction-Masked Exploration)

## What It Does
Appends a random "focus instruction" per rollout during generation, strips it before the training log_prob pass. Model learns focus-guided behaviors without depending on the instruction at inference.

## Code Locations
- **Registry:** `verl/envs/environments/focus_instructions.py` — `get_focus_instructions()`, `sample_focus_for_episode()`, `inject_focus_into_obs()`
- **Config:** `verl/trainer/config/prompt/overcooked.yaml` → `prompt.prompt.dime.*`
- **Trainer integration:** `verl/trainer/ppo/ray_multistep_trainer.py`
  - DIME config read + sampling: ~L1196-1216 (before episode loop)
  - Dual tokenize: ~L1229-1260 (inside `text_gen_proc` timer)
  - `swap_dime_prompts()`: top-level function ~L199-231
  - Prompt swap call + metrics: ~L1544-1555 (after rollout loop, before `compute_log_prob`)

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
  mask_for_training: true # false = control (focus visible in training too)
  no_supplement_prob: 0.143  # ~1/(N+1) for N=6 instructions
  template: 'Before acting, carefully consider...'
```

## Adding Focus Instructions for New Environments
1. Add instruction list to `FOCUS_REGISTRY` in `focus_instructions.py`
2. Key must match `config.envs.env_name` (lowercased)
3. Add `dime:` section to the environment's prompt YAML

## Gotchas
- `dime_enabled` must be initialized OUTSIDE the `if self.global_steps == 1 or ...` block (critic warmup scoping)
- Focus is sampled once per episode, not per step
- `swap_dime_prompts` forces `bypass_recomputing_logprobs = False` since tokens changed
- Template uses `{STEP_TEXT}` placeholder (not f-string)

## Tests
- `tests/test_dime_focus_instructions.py` — registry, sampling, injection [P1]
- `tests/test_dime_dual_tokenize.py` — tokenization shape/content consistency [P0] (requires model)
- `tests/test_dime_swap.py` — swap correctness + gold standard [P0]
- `tests/test_dime_edge_cases.py` — single rollout, zero episode, empty response [P1]

Run with: `PYTHONPATH=. python tests/test_dime_swap.py`
