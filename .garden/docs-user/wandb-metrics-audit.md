# WandB metrics audit (how they’re generated + correctness notes)

This document audits every metric currently listed in [wandb-metrics.md](.garden/docs-user/wandb-metrics.md) by:
- tracing where/how it is computed/logged
- validating whether the implementation matches its *intended* purpose (as commonly understood for PPO/SFT/eval)
- calling out assumptions and potential bugs or interpretation traps

Date: 2026-01-16

## High-impact findings (actionable)

1. **Likely bug: MultiEnvEvaluator `score_mean/std` can miss terminal-step scores**
   - In [multi_env_evaluator.py](verl/trainer/ppo/multi_env_evaluator.py#L720-L820), `score_of_traj` is updated with `np.where(~end_of_traj, score_values, score_of_traj)` *after* `end_of_traj` is updated for `done`.
   - This prevents updating the score on the same step a rollout becomes done (because `~end_of_traj` becomes false).
   - Impact: `eval_{eval_name}/score_mean`, `eval_{eval_name}/score_std`, and `episode_total_score` can be stale/incorrect (especially if `info["score"]` is only meaningful at termination).

2. **Hardened: PPO data metrics no longer crash if *all* samples are aborted**
  - Previously, [metric_utils.py](verl/trainer/ppo/metric_utils.py) raised when `response_length == 0` for all samples.
  - This is now hardened: metrics fall back to `0.0` values for the non-aborted-only aggregations instead of raising.

3. **Throughput/tokens metrics depend on whether a batch is “global” vs “local”**
   - `perf/total_num_tokens` is computed as `sum(batch.meta_info["global_token_num"])` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L277-L307).
   - In PPO, `global_token_num` is set from `attention_mask` in [ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1128-L1136). This is “global” only if the driver’s `batch` already aggregates sequences from all GPUs.
   - Interpretation: `perf/throughput = total_tokens / (time * n_gpus)` is correct *if* `total_tokens` spans all GPUs; it underestimates if `total_tokens` is already per-GPU.
  - Note: code now clamps the denominator to avoid `inf/NaN` if step time is zero.

4. **Hardened: timing per-token metrics skip zero-token cases**
  - `timing_per_token_ms/{stage}` now skips a stage if the token count for that stage is 0 to avoid division-by-zero.

## Audit details by metric

### Actor (PPO update)

Source: [dp_actor.py](verl/workers/actor/dp_actor.py#L600-L820), policy-loss math in [core_algos.py](verl/trainer/ppo/core_algos.py#L1100-L1160)

- `actor/pg_loss`
  - Generated: `pg_loss.detach().item() * loss_scale_factor` per micro-batch in [dp_actor.py](verl/workers/actor/dp_actor.py#L705-L770).
  - Purpose fit: Yes as a training signal proxy, but it is *scaled* for gradient accumulation/dynamic-bsz.
  - Assumptions: You compare runs only when loss scaling behavior is consistent, or you treat this as “contribution to backward per micro-step” not “true objective magnitude”.
  - Potential confusion: If you expect a per-token/per-sequence mean loss, this is not guaranteed.

- `actor/pg_clipfrac`
  - Generated: masked mean of `pg_losses2 > pg_losses1` in [core_algos.py](verl/trainer/ppo/core_algos.py#L1110-L1140).
  - Purpose fit: Yes; monitors how often PPO clipping activates.
  - Assumptions: `advantages`, `ratio` and `response_mask` are correctly aligned with response tokens.
  - Potential confusion: This is token-level fraction (mask-weighted), not sequence-level.

- `actor/pg_clipfrac_lower`
  - Generated: masked mean of lower-clipping activation for negative advantages in [core_algos.py](verl/trainer/ppo/core_algos.py#L1115-L1150).
  - Purpose fit: Yes; helps diagnose asymmetry and negative-adv clipping behavior.
  - Caveat: It’s only counting where `advantages < 0`.

- `actor/ppo_kl`
  - Generated: `ppo_kl` in the policy-loss implementation (computed from `log_prob - old_log_prob`), logged in [core_algos.py](verl/trainer/ppo/core_algos.py#L1145-L1160).
  - Purpose fit: Yes; approximate KL drift to the “old” policy.
  - Assumptions: `old_log_prob` corresponds to the correct snapshot policy and is not overwritten by rollout correction / bypass modes.
  - Potential confusion: Approx-KL definitions vary; don’t compare directly to “true” KL unless you confirm exact formula.

- `actor/grad_norm`
  - Generated: return value of `_optimizer_step()` in [dp_actor.py](verl/workers/actor/dp_actor.py#L780-L805).
  - Purpose fit: Yes; monitors gradient explosion/over-clipping.
  - Caveat: “Grad norm” is whatever `_optimizer_step` returns (often post-clipping norm).

- `actor/entropy`
  - Generated: aggregated entropy over response tokens via `agg_loss(entropy, response_mask, ...)` in [dp_actor.py](verl/workers/actor/dp_actor.py#L640-L705).
  - Purpose fit: Yes; tracks policy stochasticity.
  - Assumptions: `entropy` tensor corresponds to the same positions as `response_mask`.
  - Caveat: If `entropy_top_p` is used, this is *clamped/top-p entropy*, not full entropy.

- `actor/entropy_full`
  - Generated: aggregated `entropy_full` if available in [dp_actor.py](verl/workers/actor/dp_actor.py#L668-L690).
  - Purpose fit: Yes; provides unclamped baseline.
  - Caveat: Only present if model forward returns it.

- `actor/entropy_loss`
  - Generated: alias `actor/entropy_loss = actor/entropy` in [dp_actor.py](verl/workers/actor/dp_actor.py#L657-L675).
  - Purpose fit: Weak; name suggests it’s the term *added to loss*, but the sign/coefficient application is separate.
  - Potential misunderstanding: It is not `-entropy_coeff * entropy`.

- `actor/entropy_mask_empty`
  - Generated: 1 when `response_mask.sum() == 0` else 0 in [dp_actor.py](verl/workers/actor/dp_actor.py#L651-L665).
  - Purpose fit: Yes; early-warning for broken masking / empty generations.

- `actor/entropy_nan`
  - Generated: 1 when aggregated entropy is non-finite in [dp_actor.py](verl/workers/actor/dp_actor.py#L675-L690).
  - Purpose fit: Yes; numeric stability guard.

- `actor/entropy_coeff_reset`
  - Generated: set when adaptive entropy coefficient becomes non-finite and is reset (see `dp_actor.py` around the adaptive entropy coefficient management).
  - Purpose fit: Yes.
  - Assumptions: Adaptive entropy is enabled.

- `actor/entropy_coeff_used`
  - Generated: only when adaptive entropy is enabled and at least one micro-batch had valid entropy; set in [dp_actor.py](verl/workers/actor/dp_actor.py#L780-L800).
  - Purpose fit: Yes; shows coefficient used to update loss.
  - Caveat: Missing on steps where entropy couldn’t be computed (`entropy_ok` false).

- `actor/mini_batch_entropy_avg`
  - Generated: weighted average entropy across tokens (or across ok micro-batches) in [dp_actor.py](verl/workers/actor/dp_actor.py#L780-L805).
  - Purpose fit: Yes; the quantity driving adaptive entropy updates.

- `actor/entropy_coeff_final`
  - Generated: stored as a one-element list (`[coeff]`) at end of step in [dp_actor.py](verl/workers/actor/dp_actor.py#L807-L812). This is intentional because this worker returns metrics as “lists of scalars” for later reduction.
  - Purpose fit: Yes.
  - Caveat: If any code path logs this dict to WandB *without* reduction, WandB will receive a list, not a float.

- `actor/entropy_low_current`, `actor/entropy_high_current`
  - Generated: only when entropy-bounds decay is enabled, see [dp_actor.py](verl/workers/actor/dp_actor.py#L796-L805).
  - Purpose fit: Yes; sanity-check the bound schedule.

- `actor/kl_loss`, `actor/kl_coef`
  - Generated: KL penalty to a reference policy, aggregated with `agg_loss(kld, response_mask, ...)` then scaled and logged in [dp_actor.py](verl/workers/actor/dp_actor.py#L690-L740).
  - Purpose fit: Yes; tracks strength of explicit KL regularization.
  - Caveats:
    - `actor/kl_loss` is multiplied by `loss_scale_factor` in this backend, similar to `actor/pg_loss`.
    - In the generic worker loss implementation [losses.py](verl/workers/utils/losses.py#L90-L160), the keys are `kl_loss` and `kl_coef` (without the `actor/` prefix). If you’re mixing backends/configs, you may see different metric names.

- `actor/reward_kl_penalty`, `actor/reward_kl_penalty_coeff`
  - Generated: when `use_kl_in_reward` is enabled, in [ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L130-L171) as:
    - `current_kl = mean(masked_mean(kld, response_mask, axis=-1))`
    - `beta = kl_ctrl.value` (adaptive)
  - Purpose fit: Yes; this is KL as used for reward shaping and the coefficient actually applied.
  - Caveat: This KL is between `old_log_probs` and `ref_log_prob` (not necessarily the same KL as `actor/ppo_kl`).

- `actor/lr`
  - Generated: from optimizer/scheduler wrappers (varies by backend).
  - Purpose fit: Yes.
  - Caveat: In some backends the logged LR may be per-parameter-group or post-scheduler-step; treat it as “the LR in effect for that step”.

### Critic (PPO update)

Source: [dp_critic.py](verl/workers/critic/dp_critic.py#L200-L320) and shared loss math in [losses.py](verl/workers/utils/losses.py#L150-L210)

- `critic/vf_loss`
  - Generated: `vf_loss.detach().item() * loss_scale_factor` per micro-batch in [dp_critic.py](verl/workers/critic/dp_critic.py#L235-L290).
  - Purpose fit: Yes; monitors critic fitting.
  - Caveat: Like actor loss, scaled due to gradient accumulation/dynamic-bsz.

- `critic/vf_clipfrac`
  - Generated: returned by `core_algos.compute_value_loss`, logged in [dp_critic.py](verl/workers/critic/dp_critic.py#L235-L290).
  - Purpose fit: Yes; measures how often value clipping is active.
  - Caveat: Definition depends on `compute_value_loss` implementation.

- `critic/vpred_mean`
  - Generated: `masked_mean(vpreds, response_mask)` in [dp_critic.py](verl/workers/critic/dp_critic.py#L268-L285).
  - Purpose fit: Mostly; useful to detect value drift/saturation.
  - Caveat: Mean value scale depends on reward normalization.

- `critic/grad_norm`
  - Generated: from `_optimizer_step()` in [dp_critic.py](verl/workers/critic/dp_critic.py#L292-L305).
  - Purpose fit: Yes.

- `critic/lr`
  - Generated: from optimizer/scheduler wrappers (varies by backend).
  - Purpose fit: Yes.

### PPO batch/rollout statistics (logged by trainer)

Source: [metric_utils.py](verl/trainer/ppo/metric_utils.py#L40-L240)

These are computed on the *trainer-side* batch (after rollouts/rewards/advantages etc. are assembled).

- `critic/score/mean`, `critic/score/max`, `critic/score/min`
  - Generated: `sequence_score = token_level_scores.sum(-1)` and reduced over `non_aborted_mask` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L72-L120).
  - Purpose fit: Yes; tracks total per-trajectory score signal.
  - Assumptions:
    - `token_level_scores` exists and sums to a meaningful per-trajectory objective.
    - There is at least one non-aborted sample.

- `critic/rewards/mean`, `critic/rewards/max`, `critic/rewards/min`
  - Generated: `sequence_reward = token_level_rewards.sum(-1)` and reduced over `non_aborted_mask` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L72-L120).
  - Purpose fit: Yes; tracks shaped reward (including penalties if applied).

- `critic/advantages/mean|max|min`
  - Generated: masked select of `advantages` by `response_mask` then reductions in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L120-L165).
  - Purpose fit: Yes; sanity-check advantage scale.
  - Caveat: This is token-level distribution; very sensitive to masking correctness.

- `critic/returns/mean|max|min`
  - Generated: masked select of `returns` by `response_mask` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L120-L170).
  - Purpose fit: Yes.

- `critic/values/mean|max|min`
  - Generated: masked select of `values` by `response_mask` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L126-L180).
  - Purpose fit: Yes.

- `critic/vf_explained_var`
  - Generated: `1 - Var(returns - values) / (Var(returns) + 1e-5)` over response tokens in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L126-L150).
  - Purpose fit: Yes; common “critic quality” scalar.
  - Caveats:
    - Can be negative or >1; that’s not necessarily a bug.
    - If `Var(returns)` is tiny, metric is noisy.

#### Prompt/response lengths

- `response_length/mean|max|min`
  - Generated: sums of response attention mask in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L40-L70) then reductions [metric_utils.py](verl/trainer/ppo/metric_utils.py#L146-L205).
  - Purpose fit: Yes; detects truncation/collapses.

- `response_length/clip_ratio`
  - Generated: mean of `(response_length == max_response_length)` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L146-L205).
  - Purpose fit: Yes; how often you hit the response cap.
  - Caveat: Uses equality to max length; if you have padding/variable response length, ensure mask is correct.

- `response_length_non_aborted/mean|max|min` and `response_length_non_aborted/clip_ratio`
  - Generated: same stats but filtered to `response_length > 0` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L104-L145).
  - Purpose fit: Yes.
  - Potential issue: Raises if *all* are aborted.

- `response/aborted_ratio`
  - Generated: mean of `(response_length == 0)` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L104-L125).
  - Purpose fit: Yes.

- `prompt_length/mean|max|min` and `prompt_length/clip_ratio`
  - Generated: derived from attention mask and prompt segment, then reduced in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L185-L205).
  - Purpose fit: Yes.
  - Caveat: “Prompt length” includes only non-pad prompt tokens; check your dataset padding semantics.

- `num_turns/min|max|mean`
  - Generated: only if `__num_turns__` exists in `batch.non_tensor_batch` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L210-L225).
  - Purpose fit: Yes.
  - Caveat: Not always present.

- `tool_call_counts/min|max|mean`
  - Generated: only if `tool_call_counts` exists in `batch.non_tensor_batch` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L226-L240).
  - Purpose fit: Yes.

### Timing + performance metrics

Source: [metric_utils.py](verl/trainer/ppo/metric_utils.py#L241-L310)

- `timing_s/{stage}`
  - Generated: raw seconds from `timing_raw` dict in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L246-L270).
  - Purpose fit: Yes.
  - Caveat: Must interpret stages consistently across trainers.

- `timing_per_token_ms/{stage}`
  - Generated: `timing_raw[stage] * 1000 / tokens_for_stage` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L246-L270).
  - Purpose fit: Yes.
  - Potential issue: No guard for `tokens_for_stage == 0`.

- `perf/total_num_tokens`
  - Generated: `sum(batch.meta_info["global_token_num"])` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L277-L307), where PPO sets `global_token_num` from `attention_mask` in [ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1128-L1136).
  - Purpose fit: Yes.
  - Assumption: This `batch` represents the aggregate work you want to attribute to a single “trainer step”.

- `perf/time_per_step`
  - Generated: `timing_raw["step"]` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L277-L307).
  - Purpose fit: Yes.

- `perf/throughput`
  - Generated: `total_tokens / (time * n_gpus)` in [metric_utils.py](verl/trainer/ppo/metric_utils.py#L277-L307).
  - Purpose fit: Yes if `total_tokens` is global tokens across all GPUs.
  - Assumption to validate: whether the `batch` in the driver is global across GPUs.

### SFT metrics

#### Classic SFT trainer

Source: [sft_trainer.py](verl/trainer/sft_trainer.py#L280-L360)

- `train/loss`, `train/grad_norm`, `train/lr`, `train/mfu`
  - Generated: renamed from engine metrics in [sft_trainer.py](verl/trainer/sft_trainer.py#L300-L309), where `mfu` is computed in [engine_workers.py](verl/workers/engine_workers.py#L130-L170).
  - Purpose fit: Yes.
  - Assumptions/caveats:
    - `train/mfu` includes normalization by `world_size`; interpretation depends on how `estimate_flops` treats `global_token_num`.

- `train/global_tokens`
  - Generated: sum of `batch_seqlens` on the logging rank in [sft_trainer.py](verl/trainer/sft_trainer.py#L306-L312).
  - Purpose fit: Yes as “tokens processed this step *on the logging rank*”.
  - Potential misunderstanding: despite the name, this may not be “global across all DP ranks” unless `batch_seqlens` is already global.

- `train/total_tokens(B)`
  - Generated: cumulative sum of `train/global_tokens` divided by 1e9 in [sft_trainer.py](verl/trainer/sft_trainer.py#L306-L314).
  - Purpose fit: Yes.
  - Caveat: Same scope caveat as `train/global_tokens`.

- `val/loss`
  - Generated: mean of per-batch val losses, then reduced across DP with all-reduce AVG in [sft_trainer.py](verl/trainer/sft_trainer.py#L332-L360).
  - Purpose fit: Yes.

#### FSDP SFT trainer

Source: [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L500-L550) and [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L520-L535)

- `train/loss`
  - Generated: average of per-step loss across DP via all-reduce in [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L510-L540).
  - Purpose fit: Yes.

- `train/lr(1e-3)`
  - Generated: `lr * 1e3` in [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L520-L535).
  - Purpose fit: Mixed; it’s fine, but the key name is easy to misread.
  - Suggestion: Prefer logging raw LR as well to avoid confusion.

- `train/time(s)`
  - Generated: wall-clock time per step in [fsdp_sft_trainer.py](verl/trainer/fsdp_sft_trainer.py#L514-L535).
  - Purpose fit: Yes.

- `val/loss`
  - Generated: averaged validation losses (implementation later in file); matches purpose.

### Validation metrics (multistep PPO trainer)

Source: [ray_multistep_trainer.py](verl/trainer/ppo/ray_multistep_trainer.py#L780-L840)

- `val/rewards_mean`, `val/rewards_std`
  - Generated: mean/std of cumulative trajectory reward `rew_of_traj`.
  - Purpose fit: Yes.

- `val/traj_length_mean`, `val/traj_length_std`
  - Generated: mean/std of `len_of_traj` (counts steps until done).
  - Purpose fit: Yes.

- `val/pos_reward_total_prop_mean`, `val/pos_reward_total_prop_std`
  - Generated: `succ_of_traj = (rew_of_traj > 0)` (final cumulative reward positive).
  - Purpose fit: Yes for “success defined by positive final return”.
  - Assumption: Positive return corresponds to success.

- `val/pos_reward_any_prop_mean`, `val/pos_reward_any_prop_std`
  - Generated: `pos_rew_of_traj` is an OR over `(reward > 0)` during trajectory.
  - Purpose fit: Yes for “any positive reward encountered”.

### Evaluation metrics (MultiEnvEvaluator)

Source: metric accumulation and aggregation in [multi_env_evaluator.py](verl/trainer/ppo/multi_env_evaluator.py#L720-L1040)

#### Standard rollout metrics (suffixes; logged as `eval_{eval_name}/{suffix}`)

- `rewards_mean`, `rewards_std`
  - Generated: mean/std over `rew_of_traj` (cumulative reward per rollout).
  - Purpose fit: Yes.

- `pos_reward_any_prop_mean`, `pos_reward_any_prop_std`
  - Generated: OR over `(reward_vec > 0)` per rollout, then mean/std.
  - Purpose fit: Yes (any-positive reward proxy).

- `traj_length_mean`, `traj_length_std`
  - Generated: `len_of_traj` counts active steps per rollout.
  - Purpose fit: Yes.

- `score_mean`, `score_std`
  - Generated: aggregates `info["score"]` when present.
  - Purpose fit: Yes if env provides a meaningful “score”.
  - Potential bug: terminal-step update issue described in “High-impact findings”.

- `toks_out_mean`, `toks_out_std`
  - Generated: mean/std of *last-step only* response token counts, taken from `response_n_tokens_last_step`.
  - Purpose fit: Partially; it answers “how long were the final actions?”, not total tokens per rollout.
  - Potential misunderstanding: Many users expect “tokens per rollout”; this is not that.

- `tokens_per_rollout`
  - Generated: `total_tokens_generated / n_rollouts`.
  - Purpose fit: Yes.
  - Caveat: Includes only tokens from steps where generation occurred; ignores prompt tokens.

- `tokens_per_step`
  - Generated: `total_tokens_generated / attempted_actions_total`.
  - Purpose fit: Yes; this is the intuitive “tokens per executed env step” metric.
  - Assumption: `attempted_actions_total` is a faithful count of steps where an action decision was attempted.

- `tokens_per_step_cap`
  - Generated: `total_tokens_generated / (episode_length * n_rollouts)`.
  - Purpose fit: Yes as a “max-steps normalized” rate; useful when you want a fixed denominator across evals.

#### Inference time metrics

- `inference_time_seconds`
  - Generated: sum of wall-clock around `generate_sequences` calls.
  - Purpose fit: Yes.
  - Caveat: Excludes env stepping/processing time.

- `inference_time_per_rollout`
  - Generated: `inference_time_seconds / n_rollouts`.
  - Purpose fit: Yes.

- `inference_time_per_step`
  - Generated: `inference_time_seconds / attempted_actions_total`.
  - Purpose fit: Yes; intuitive “seconds per executed env step” metric.

- `inference_time_per_step_cap`
  - Generated: `inference_time_seconds / (episode_length * n_rollouts)`.
  - Purpose fit: Yes as a “max-steps normalized” rate.

#### Action validity metrics

- `total_len_of_trajs`
  - Generated: sum of `all_len_of_traj` (0 if lengths unavailable).
  - Purpose fit: Yes.

- `valid_actions_total`
  - Generated: counts `info["action_was_valid"]` for rollouts that were not already ended.
  - Purpose fit: Yes.
  - Caveat: Depends on env populating `action_was_valid` correctly.

- `attempted_actions_total`
  - Generated: counts steps/rollouts not already ended.
  - Purpose fit: Yes.

- `executed_steps_total`
  - Generated: alias of `attempted_actions_total`.
  - Purpose fit: Yes; makes per-step denominators explicit for dashboards.

- `executed_steps_per_rollout`
  - Generated: `executed_steps_total / n_rollouts`.
  - Purpose fit: Yes; pairs naturally with `tokens_per_rollout` and helps interpret early termination.

- `valid_action_ratio`
  - Generated: `valid/attempted` with denominator clamped by 1.
  - Purpose fit: Yes.

#### Action entropy metrics (optional)

- `action_entropy_mean`, `action_entropy_std`
  - Generated: Shannon entropy of executed-action distribution from probe samples.
  - Purpose fit: Yes for “stochasticity / ambiguity” of the policy at a state.
  - Assumptions:
    - `executed_action_text` extraction and validity flags are accurate.
    - Probe sampling configuration (n_samples, temperature, etc.) matches what you want to measure.

- `action_entropy_num_measurements`
  - Generated: count of entropy values recorded.
  - Purpose fit: Yes.

- `action_entropy_probe_time_seconds`
  - Generated: wall-clock time spent on probe generations.
  - Purpose fit: Yes.

- `unique_executed_actions_per_unique_text_mean/std`
  - Generated: per-probe-step ratio intended to measure “how many executed actions per distinct raw response”.
  - Purpose fit: Often yes, but this ratio can be unintuitive: if many texts map to the same action, the ratio falls.

- `unique_valid_actions_per_unique_valid_text_mean/std`
  - Generated: same as above but on the subset of valid actions/texts.
  - Purpose fit: Yes.

- `unique_texts_step_mean/std`, `unique_executed_actions_step_mean/std`, `unique_valid_actions_step_mean/std`
  - Generated: unique counts per probe step.
  - Purpose fit: Yes.

- `val/entropy_dist`
  - Generated: JSON-encoded normalized action frequency distribution across probes.
  - Purpose fit: Yes; useful for debugging mode-collapse.
  - Caveat: key contains `/` so final WandB key is `eval_{eval_name}/val/entropy_dist`.

#### Seed-group diversity / coverage metrics (optional)

- `n_distinct_state_actions_valid_mean/std`
  - Generated: per-group number of unique `"{observation_text} {executed_action}"` strings, restricted to valid actions.
  - Purpose fit: Yes as a *proxy* for state-action diversity.
  - Caveats:
    - Using raw `observation_text` makes this sensitive to formatting/noise; small textual differences explode “distinct” counts.
    - It’s not a canonical MDP state hash; treat it as heuristic.

- `n_distinct_state_actions_mean/std`
  - Generated: same as above including invalid actions.
  - Purpose fit: Yes.

- `distinct_state_actions_per_frame_mean/std`
  - Generated: `distinct_count / total_frames_in_group`.
  - Purpose fit: Yes.
  - Caveat: Frame count depends on `len_of_traj`; if rollouts terminate early, the denominator is smaller.

- `distinct_state_actions_valid_per_frame_mean/std`
  - Generated: valid-only variant.
  - Purpose fit: Yes.

- `distinct_state_actions_valid_coverage_mean/std`, `distinct_state_actions_coverage_mean/std`
  - Generated: `distinct_count / (seed_group_size * episode_length)`.
  - Purpose fit: It’s a normalized “coverage vs max opportunity”.
  - Caveat: If rollouts terminate early, “opportunity” is overstated, so coverage is understated.

### Logged tables / artifacts

Source: [tracking.py](verl/utils/tracking.py#L390-L440)

- `val/generations`
  - Generated: a `wandb.Table` containing `step` and repeated `input_i/output_i/score_i` columns.
  - Purpose fit: Yes; qualitative debugging.
  - Assumptions:
    - The number of samples logged per step stays constant. If it changes, the table schema changes and may error.

## Notes on the `analysis/` CLI metric patterns

The patterns in [analysis/config.py](analysis/config.py#L52) are selection defaults, not emitters.
- Purpose fit: Yes; provides out-of-box comparisons.
- Caveat: Pattern hits do not guarantee the metric exists for a run (depends on trainer/evaluator).
