# WandB metrics (found in this repo)

This page is a **work in progress** intended to become the “single source of truth” for all WandB metrics emitted by this repo.

For a correctness/interpretation review of each metric (assumptions, likely bugs, and meaning), see:
- [.garden/docs-user/wandb-metrics-audit.md](.garden/docs-user/wandb-metrics-audit.md)

What’s in here today:
- Metrics explicitly documented in the repo’s existing reference: [wandb.md](wandb.md)
- Metrics found by scanning core `verl/` training code paths (`SFT`, `PPO`, evaluation/validation helpers)
- Metric *patterns* used by the `analysis/` CLI for comparisons

What’s *not* complete yet:
- Some metrics are generated dynamically (e.g. per-task eval metrics) and don’t appear as fixed string literals.
- Recipe-specific metrics under `recipe/` are only partially captured in this first pass.

If you want, next pass can be: “enumerate every literal metric key across `verl/` + `recipe/` and link each to its exact log site(s)”.

## Where metrics are logged

Most training code logs via the unified tracking interface:
- [Tracking](verl/utils/tracking.py#L1)

That class forwards `tracking.log(data=metrics, step=...)` to WandB (and optionally other backends). It does **not** impose any naming conventions beyond using a string key.

## Canonical reference (already in repo)

The repo already contains a partial metrics reference:
- [wandb.md](wandb.md)

That file currently documents a subset of **Actor** metrics (plus explanations). This page extends it with the additional metrics that are implemented in code.

## Core PPO metrics (Actor/Critic)

### Actor (PPO update)

Primary implementation:
- [dp_actor.py](verl/workers/actor/dp_actor.py#L587)
- Policy loss “clip” metrics are produced by the loss functions in: [core_algos.py](verl/trainer/ppo/core_algos.py#L1131)

| Metric | What it measures | How it’s calculated (high-level) | Where in code |
|---|---|---|---|
| `actor/pg_loss` | Policy gradient loss | Value returned by the selected policy loss fn; logged per micro-batch and scaled by loss scale factor | [dp_actor.py](verl/workers/actor/dp_actor.py#L713), [losses.py](verl/workers/utils/losses.py#L131) |
| `actor/pg_clipfrac` | PPO clip fraction | Fraction of tokens where the clipped objective is active | [core_algos.py](verl/trainer/ppo/core_algos.py#L1131) |
| `actor/pg_clipfrac_lower` | PPO lower-clip fraction (advantages < 0) | Tracks how often the “lower clip” branch activates for negative advantages | [core_algos.py](verl/trainer/ppo/core_algos.py#L1133) |
| `actor/ppo_kl` | Approx KL(current || old) used by PPO | Computed from old vs current logprobs in the policy loss implementation | [core_algos.py](verl/trainer/ppo/core_algos.py#L1132) |
| `actor/grad_norm` | Actor grad norm | Gradient norm after clipping/optimizer step | [dp_actor.py](verl/workers/actor/dp_actor.py#L749) |
| `actor/entropy` | Token-level policy entropy (aggregated) | Aggregated entropy over response tokens; may use top-p clamped entropy if configured | [dp_actor.py](verl/workers/actor/dp_actor.py#L665) |
| `actor/entropy_full` | Unclamped entropy (aggregated) | Aggregated “full” entropy (if computed by model forward) | [dp_actor.py](verl/workers/actor/dp_actor.py#L671) |
| `actor/entropy_loss` | Legacy alias for entropy | Set equal to `actor/entropy` | [dp_actor.py](verl/workers/actor/dp_actor.py#L666) |
| `actor/entropy_mask_empty` | Entropy masking guard | 1 if response mask has no valid tokens; else 0 | [dp_actor.py](verl/workers/actor/dp_actor.py#L661) |
| `actor/entropy_nan` | Entropy numeric guard | 1 if entropy aggregation produced non-finite values; else 0 | [dp_actor.py](verl/workers/actor/dp_actor.py#L676) |
| `actor/entropy_coeff_reset` | Entropy coeff reset flag | 1 if entropy coefficient became non-finite and was reset; else 0 | [dp_actor.py](verl/workers/actor/dp_actor.py#L587) |
| `actor/entropy_coeff_used` | Adaptive entropy coefficient used (mini-batch) | Logged when adaptive entropy is enabled and entropy was actually applied to loss | [dp_actor.py](verl/workers/actor/dp_actor.py#L720) |
| `actor/mini_batch_entropy_avg` | Mini-batch entropy (for adaptive update) | Weighted average entropy over tokens in the mini-batch | [dp_actor.py](verl/workers/actor/dp_actor.py#L742) |
| `actor/entropy_coeff_final` | Adaptive entropy coefficient after the step | Final coefficient after mini-batch update(s) | [dp_actor.py](verl/workers/actor/dp_actor.py#L755) |
| `actor/entropy_low_current` | Current entropy lower bound (if decayed) | Bound after optional decay schedule | [dp_actor.py](verl/workers/actor/dp_actor.py#L745) |
| `actor/entropy_high_current` | Current entropy upper bound (if decayed) | Bound after optional decay schedule | [dp_actor.py](verl/workers/actor/dp_actor.py#L746) |
| `actor/kl_loss` | KL loss to a reference policy | Aggregated KL penalty (type depends on config), scaled into loss | [dp_actor.py](verl/workers/actor/dp_actor.py#L700), [megatron_actor.py](verl/workers/actor/megatron_actor.py#L504) |
| `actor/kl_coef` | KL coefficient | `config.kl_loss_coef` | [dp_actor.py](verl/workers/actor/dp_actor.py#L701), [megatron_actor.py](verl/workers/actor/megatron_actor.py#L505) |
| `actor/reward_kl_penalty` | KL penalty used in reward shaping | KL value between policy and reference used for penalty | [ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L158) |
| `actor/reward_kl_penalty_coeff` | KL penalty coefficient | Beta for the reward KL penalty | [ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L158) |
| `actor/lr` | Actor learning rate | Taken from optimizer/lr scheduler (engine dependent) | [fsdp_workers.py](verl/workers/fsdp_workers.py#L895), [engine_workers.py](verl/workers/engine_workers.py#L357), [megatron_workers.py](verl/workers/megatron_workers.py#L735) |

### Critic (PPO update)

Primary implementations:
- [dp_critic.py](verl/workers/critic/dp_critic.py#L251)
- [metric_utils.py](verl/trainer/ppo/metric_utils.py#L1) contains “batch/rollout statistics” metrics under `critic/...`

| Metric | What it measures | How it’s calculated (high-level) | Where in code |
|---|---|---|---|
| `critic/vf_loss` | Value function loss | From `compute_value_loss(...)`, aggregated and scaled | [dp_critic.py](verl/workers/critic/dp_critic.py#L251) |
| `critic/vf_clipfrac` | Value function clip fraction | Fraction of values that got clipped | [dp_critic.py](verl/workers/critic/dp_critic.py#L252) |
| `critic/vpred_mean` | Mean predicted value | Mean of predicted values over response mask | [dp_critic.py](verl/workers/critic/dp_critic.py#L253) |
| `critic/grad_norm` | Critic grad norm | Gradient norm after clipping/optimizer step | [dp_critic.py](verl/workers/critic/dp_critic.py#L260) |
| `critic/lr` | Critic learning rate | Taken from optimizer/lr scheduler (engine dependent) | [fsdp_workers.py](verl/workers/fsdp_workers.py#L1572), [engine_workers.py](verl/workers/engine_workers.py#L526), [megatron_workers.py](verl/workers/megatron_workers.py#L1211) |

## PPO batch/rollout statistics (`critic/*`, lengths, tools)

Defined in:
- [metric_utils.py](verl/trainer/ppo/metric_utils.py#L1)

These are computed from tensors already present in a PPO batch (token-level scores/rewards, advantages, returns, values, masks).

### Scores / rewards / advantages / returns / values

| Metric | What it measures | How it’s calculated | Where in code |
|---|---|---|---|
| `critic/score/mean|max|min` | Sequence score stats | Sum token-level scores, then compute mean/max/min over non-aborted samples | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L106) |
| `critic/rewards/mean|max|min` | Sequence reward stats | Sum token-level rewards, then compute mean/max/min over non-aborted samples | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L107) |
| `critic/advantages/mean|max|min` | Advantage stats over response tokens | Mask-select advantages by response mask, then reduce | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L162) |
| `critic/returns/mean|max|min` | Return stats over response tokens | Mask-select returns by response mask, then reduce | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L166) |
| `critic/values/mean|max|min` | Value prediction stats over response tokens | Mask-select values by response mask, then reduce | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L176) |
| `critic/vf_explained_var` | Explained variance of critic | $1 - \mathrm{Var}(\text{returns}-\text{values}) / (\mathrm{Var}(\text{returns}) + 10^{-5})$ | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L183) |

### Prompt/response lengths

| Metric | What it measures | How it’s calculated | Where in code |
|---|---|---|---|
| `response_length/mean|max|min` | Response length stats | Derived from attention masks | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L189) |
| `response_length/clip_ratio` | Fraction at max response length | Mean of `(response_length == max_response_length)` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L206) |
| `response_length_non_aborted/mean|max|min` | Response lengths excluding aborted | Same as above, but excludes samples with zero response length | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L126) |
| `response_length_non_aborted/clip_ratio` | Clip ratio excluding aborted | Same as `clip_ratio`, excluding aborted | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L139) |
| `response/aborted_ratio` | Fraction aborted | Mean of `(response_length == 0)` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L121) |
| `prompt_length/mean|max|min` | Prompt length stats | Derived from attention masks | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L213) |
| `prompt_length/clip_ratio` | Fraction at max prompt length | Mean of `(prompt_length == max_prompt_length)` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L216) |
| `num_turns/min|max|mean` | Multi-turn conversation turns | Aggregates `__num_turns__` if present in non-tensor batch | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L220) |
| `tool_call_counts/min|max|mean` | Tool call count stats | Aggregates `tool_call_counts` if present in non-tensor batch | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L225) |

## Timing + performance metrics

Defined in:
- [metric_utils.py](verl/trainer/ppo/metric_utils.py#L241)

These are emitted when the trainer collects per-stage timings (seconds) and passes them into metric utils.

| Metric | What it measures | How it’s calculated | Where in code |
|---|---|---|---|
| `timing_s/{stage}` | Raw stage duration (s) | Taken directly from `timing_raw[stage]` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L262) |
| `timing_per_token_ms/{stage}` | Stage time per token (ms) | `timing_s[stage] * 1000 / tokens_for_stage` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L263) |
| `perf/total_num_tokens` | Total tokens processed | Sum of `batch.meta_info["global_token_num"]` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L294) |
| `perf/time_per_step` | Total step time (s) | `timing_raw["step"]` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L295) |
| `perf/throughput` | Tokens/sec/GPU | `total_tokens / (time * n_gpus)` | [metric_utils.py](verl/trainer/ppo/metric_utils.py#L296) |

Known stage names (based on token normalization logic): `gen`, `ref`, `values`, `adv`, `update_critic`, `update_actor`, plus overall `step`.

## SFT metrics

### Classic SFT trainer

Defined in:
- [sft_trainer.py](verl/trainer/sft_trainer.py#L304)

| Metric | What it measures | How it’s calculated | Where in code |
|---|---|---|---|
| `train/loss` | Training loss | Renamed from engine-produced `loss` | [sft_trainer.py](verl/trainer/sft_trainer.py#L304) |
| `train/grad_norm` | Gradient norm | Renamed from engine-produced `grad_norm` | [sft_trainer.py](verl/trainer/sft_trainer.py#L305) |
| `train/lr` | Learning rate | Renamed from engine-produced `lr` | [sft_trainer.py](verl/trainer/sft_trainer.py#L306) |
| `train/mfu` | Model FLOPs utilization | Renamed from engine-produced `mfu` | [sft_trainer.py](verl/trainer/sft_trainer.py#L307) |
| `train/global_tokens` | Tokens in this step | Sum of batch sequence lengths | [sft_trainer.py](verl/trainer/sft_trainer.py#L308) |
| `train/total_tokens(B)` | Total tokens so far (billions) | Cumulative sum of `train/global_tokens` divided by $10^9$ | [sft_trainer.py](verl/trainer/sft_trainer.py#L311) |
| `val/loss` | Validation loss | Mean loss over val dataloader, reduced across DP | [sft_trainer.py](verl/trainer/sft_trainer.py#L341) |

### FSDP SFT trainer

Defined in:
- [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L521)

| Metric | What it measures | How it’s calculated | Where in code |
|---|---|---|---|
| `train/loss` | Training loss (DP reduced) | Average loss across DP ranks | [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L521) |
| `train/lr(1e-3)` | LR scaled by $10^3$ | `lr * 1e3` | [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L522) |
| `train/time(s)` | Time per step | Wall-clock per training step | [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L523) |
| `val/loss` | Validation loss | Mean of per-batch validation losses | [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L690) |

## Validation metrics (multistep PPO trainer)

Defined in:
- [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L806)

These summarize rollout validation trajectories in the multistep trainer:

| Metric | What it measures | How it’s calculated | Where in code |
|---|---|---|---|
| `val/rewards_mean` | Mean trajectory return | Mean of per-trajectory cumulative reward | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L806) |
| `val/rewards_std` | Std of trajectory return | Std of per-trajectory cumulative reward | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L810) |
| `val/traj_length_mean` | Mean trajectory length | Mean of steps until done | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L808) |
| `val/traj_length_std` | Std of trajectory length | Std of steps until done | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L812) |
| `val/pos_reward_total_prop_mean` | Fraction with positive final return | Mean of `(total_return > 0)` | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L807) |
| `val/pos_reward_total_prop_std` | Std of positive-final-return indicator | Std of `(total_return > 0)` | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L811) |
| `val/pos_reward_any_prop_mean` | Fraction with any positive reward | Mean of `(any_step_reward > 0)` | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L809) |
| `val/pos_reward_any_prop_std` | Std of “any positive reward” indicator | Std of `(any_step_reward > 0)` | [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L813) |

## Evaluation metrics (MultiEnvEvaluator)

`MultiEnvEvaluator` runs evaluation across one or more environments and returns a flat dict `metric_dict` which is then **namespaced per environment**.

In WandB, each metric is logged under:

- `eval_{eval_name}/{metric_name}`

Where:
- `eval_name` is the logical name from the evaluation config (e.g. `snake`, `webshop`, etc.)
- `metric_name` is a key from `metric_dict`

Implementation:
- Metric computation and `metric_dict` assembly: [multi_env_evaluator.py](verl/trainer/ppo/multi_env_evaluator.py#L863)
- Namespacing logic (`prefixed_key = f"eval_{eval_name}/{key}"`): [multi_env_evaluator.py](verl/trainer/ppo/multi_env_evaluator.py#L93)

### Standard rollout metrics (when enabled)

These are reported when `track_standard_metrics` is true (i.e. normal eval rollouts rather than “entropy-only” probing).

| Metric name (suffix) | WandB key example | What it measures | How it’s calculated |
|---|---|---|---|
| `rewards_mean` | `eval_snake/rewards_mean` | Mean trajectory return | Mean of per-rollout cumulative rewards |
| `rewards_std` | `eval_snake/rewards_std` | Std trajectory return | Std of per-rollout cumulative rewards |
| `pos_reward_any_prop_mean` | `eval_snake/pos_reward_any_prop_mean` | Fraction of rollouts with any positive reward | Mean of `(any_step_reward > 0)` per rollout |
| `pos_reward_any_prop_std` | `eval_snake/pos_reward_any_prop_std` | Std of “any positive reward” indicator | Std of the per-rollout indicator |
| `traj_length_mean` | `eval_snake/traj_length_mean` | Mean trajectory length | Mean steps until done |
| `traj_length_std` | `eval_snake/traj_length_std` | Std trajectory length | Std steps until done |
| `score_mean` | `eval_snake/score_mean` | Mean score (if env provides) | Aggregated from `info["score"]`; current implementation likely misses terminal-step scores for rollouts that end on the current step (see audit doc) |
| `score_std` | `eval_snake/score_std` | Std score (if env provides) | Aggregated from `info["score"]`; see audit doc caveat |
| `toks_out_mean` | `eval_snake/toks_out_mean` | Mean output tokens (last step only) | Mean of non-pad response token counts from the last generated step |
| `toks_out_std` | `eval_snake/toks_out_std` | Std output tokens (last step only) | Std of last-step token counts |
| `tokens_per_rollout` | `eval_snake/tokens_per_rollout` | Tokens generated per rollout | `total_tokens_generated / n_rollouts` |
| `tokens_per_step` | `eval_snake/tokens_per_step` | Tokens generated per executed env step | `total_tokens_generated / attempted_actions_total` |
| `tokens_per_step_cap` | `eval_snake/tokens_per_step_cap` | Tokens generated per max-possible step | `total_tokens_generated / (episode_length * n_rollouts)` |

### Inference time metrics (always logged)

These are logged regardless of whether standard rollout metrics are enabled.

| Metric name (suffix) | WandB key example | What it measures | How it’s calculated |
|---|---|---|---|
| `inference_time_seconds` | `eval_snake/inference_time_seconds` | Total inference time during eval | Sum of per-batch wall-clock times around `generate_sequences` |
| `inference_time_per_rollout` | `eval_snake/inference_time_per_rollout` | Inference time per rollout | `inference_time_seconds / n_rollouts` |
| `inference_time_per_step` | `eval_snake/inference_time_per_step` | Inference time per executed env step | `inference_time_seconds / attempted_actions_total` |
| `inference_time_per_step_cap` | `eval_snake/inference_time_per_step_cap` | Inference time per max-possible step | `inference_time_seconds / (episode_length * n_rollouts)` |

### Action validity metrics (always logged)

These track how many actions were attempted vs. valid (based on env `info`).

| Metric name (suffix) | WandB key example | What it measures | How it’s calculated |
|---|---|---|---|
| `total_len_of_trajs` | `eval_snake/total_len_of_trajs` | Total trajectory lengths across rollouts | Sum of per-rollout lengths (0 if lengths unavailable) |
| `valid_actions_total` | `eval_snake/valid_actions_total` | Count of valid actions | Sum over steps/rollouts where `info["action_was_valid"]` and rollout not ended |
| `attempted_actions_total` | `eval_snake/attempted_actions_total` | Count of attempted actions | Sum over steps/rollouts not already ended |
| `executed_steps_total` | `eval_snake/executed_steps_total` | Total executed env steps | Alias of `attempted_actions_total`; explicit denominator for per-step metrics |
| `executed_steps_per_rollout` | `eval_snake/executed_steps_per_rollout` | Executed env steps per rollout | `executed_steps_total / n_rollouts` |
| `valid_action_ratio` | `eval_snake/valid_action_ratio` | Valid/attempted ratio | `valid_actions_total / max(1, attempted_actions_total)` |

### Action entropy metrics (optional)

Enabled via `evaluation.environments[].action_entropy.enabled: true`.

| Metric name (suffix) | WandB key example | What it measures | How it’s calculated |
|---|---|---|---|
| `action_entropy_mean` | `eval_snake/action_entropy_mean` | Mean Shannon entropy of executed actions | For selected steps, sample `n_samples` completions per active rollout; compute entropy over executed actions |
| `action_entropy_std` | `eval_snake/action_entropy_std` | Std Shannon entropy | Std of the per-rollout entropies from probes |
| `action_entropy_num_measurements` | `eval_snake/action_entropy_num_measurements` | Number of entropy measurements | Count of per-rollout entropy values recorded |
| `action_entropy_probe_time_seconds` | `eval_snake/action_entropy_probe_time_seconds` | Time spent on entropy probes | Wall-clock time around probe generations |
| `unique_executed_actions_per_unique_text_mean` | `eval_snake/unique_executed_actions_per_unique_text_mean` | Action diversity per distinct raw response | Ratio aggregated per probe step then averaged |
| `unique_executed_actions_per_unique_text_std` | `eval_snake/unique_executed_actions_per_unique_text_std` | Std of action diversity per raw response | Std over probe-step ratios |
| `unique_valid_actions_per_unique_valid_text_mean` | `eval_snake/unique_valid_actions_per_unique_valid_text_mean` | Valid-action diversity per distinct valid raw response | Ratio aggregated per probe step then averaged |
| `unique_valid_actions_per_unique_valid_text_std` | `eval_snake/unique_valid_actions_per_unique_valid_text_std` | Std of valid-action diversity per valid raw response | Std over probe-step ratios |
| `unique_texts_step_mean` | `eval_snake/unique_texts_step_mean` | Mean unique raw response count per probe step | Mean over probe steps |
| `unique_texts_step_std` | `eval_snake/unique_texts_step_std` | Std unique raw response count per probe step | Std over probe steps |
| `unique_executed_actions_step_mean` | `eval_snake/unique_executed_actions_step_mean` | Mean unique executed-action count per probe step | Mean over probe steps |
| `unique_executed_actions_step_std` | `eval_snake/unique_executed_actions_step_std` | Std unique executed-action count per probe step | Std over probe steps |
| `unique_valid_actions_step_mean` | `eval_snake/unique_valid_actions_step_mean` | Mean unique valid-action count per probe step | Mean over probe steps |
| `unique_valid_actions_step_std` | `eval_snake/unique_valid_actions_step_std` | Std unique valid-action count per probe step | Std over probe steps |
| `val/entropy_dist` | `eval_snake/val/entropy_dist` | JSON of executed-action empirical distribution | Normalized `Counter(executed_action)` across all probes, JSON-encoded |

Note: `val/entropy_dist` includes a `/` in the metric name, so after prefixing the final key becomes `eval_{eval_name}/val/entropy_dist`.

### Seed-group diversity / coverage metrics (optional)

These are only computed when `seed_group_size` is set such that `n_groups > 1`.

| Metric name (suffix) | WandB key example | What it measures | How it’s calculated |
|---|---|---|---|
| `n_distinct_state_actions_valid_mean` | `eval_snake/n_distinct_state_actions_valid_mean` | Mean distinct (state, executed_action) count per group (valid-only) | For each group, count unique `"{observation_text} {executed_action}"` strings where action was valid |
| `n_distinct_state_actions_valid_std` | `eval_snake/n_distinct_state_actions_valid_std` | Std distinct valid (state, action) per group | Std across groups |
| `n_distinct_state_actions_mean` | `eval_snake/n_distinct_state_actions_mean` | Mean distinct (state, action) per group (all actions) | Same as above but includes invalid actions |
| `n_distinct_state_actions_std` | `eval_snake/n_distinct_state_actions_std` | Std distinct (state, action) per group (all actions) | Std across groups |
| `distinct_state_actions_per_frame_mean` | `eval_snake/distinct_state_actions_per_frame_mean` | Distinct (state, action) per frame | Distinct count divided by total frames per group, averaged |
| `distinct_state_actions_per_frame_std` | `eval_snake/distinct_state_actions_per_frame_std` | Std distinct (state, action) per frame | Std across groups |
| `distinct_state_actions_valid_per_frame_mean` | `eval_snake/distinct_state_actions_valid_per_frame_mean` | Distinct valid (state, action) per frame | Valid distinct count divided by total frames per group, averaged |
| `distinct_state_actions_valid_per_frame_std` | `eval_snake/distinct_state_actions_valid_per_frame_std` | Std distinct valid (state, action) per frame | Std across groups |
| `distinct_state_actions_valid_coverage_mean` | `eval_snake/distinct_state_actions_valid_coverage_mean` | Coverage of valid distinct actions vs opportunity | Distinct-valid count divided by `(seed_group_size * episode_length)`, averaged |
| `distinct_state_actions_valid_coverage_std` | `eval_snake/distinct_state_actions_valid_coverage_std` | Std valid coverage | Std across groups |
| `distinct_state_actions_coverage_mean` | `eval_snake/distinct_state_actions_coverage_mean` | Coverage of all distinct actions vs opportunity | Distinct-all count divided by `(seed_group_size * episode_length)`, averaged |
| `distinct_state_actions_coverage_std` | `eval_snake/distinct_state_actions_coverage_std` | Std coverage | Std across groups |

## Logged tables / artifacts

| Key | What it is | Where in code |
|---|---|---|
| `val/generations` | WandB table of sampled validation generations | [tracking.py](verl/utils/tracking.py#L399) |

## Metrics used by the `analysis/` CLI

The analysis CLI doesn’t define metrics itself, but it contains **default patterns** it will look for in WandB run summaries:
- [Default patterns](analysis/config.py#L52)

Patterns include:
- `eval_*/rewards_mean`
- `eval_*/score_mean`
- `eval_*/traj_length_mean`
- `eval_*/pos_reward_any_prop_mean`
- `eval_*/tokens_per_step`
- `generation/success_rate`
- `val/*/rewards_mean`

Those metrics may be emitted by task-specific evaluators and therefore may not appear as fixed literals in core trainer code.

