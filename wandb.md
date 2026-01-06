# WandB Metrics Reference

This document describes the metrics logged to Weights & Biases during training.

---

## Table of Contents

- [Actor Metrics](#actor-metrics)
  - [Core Training](#core-training)
  - [Entropy Regularization](#entropy-regularization)
  - [Adaptive Entropy](#adaptive-entropy)
  - [KL Divergence](#kl-divergence)
- [Critic Metrics](#critic-metrics)
- [Rollout Metrics](#rollout-metrics)
- [Reward Metrics](#reward-metrics)
- [System Metrics](#system-metrics)

---

## Metric Granularity

Metrics are logged at different granularities and aggregated before being sent to WandB:

| Granularity | Description | Aggregation |
|-------------|-------------|-------------|
| **Per micro-batch** | Logged inside gradient accumulation loop | Averaged across all micro-batches in the step |
| **Per mini-batch** | Logged once per optimizer step | Averaged across all mini-batches in the step |
| **Per step** | Logged once at the end of `update_policy` | Reported directly (no aggregation) |

---

## Actor Metrics

### Core Training

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| `actor/pg_loss` | Per micro-batch | float | Policy gradient loss (scaled by gradient accumulation factor) |
| `actor/grad_norm` | Per mini-batch | float | Gradient norm after clipping |

### Entropy Regularization

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| `actor/entropy` | Per micro-batch | float | Aggregated entropy of the policy (using `entropy_top_p` if configured) |
| `actor/entropy_full` | Per micro-batch | float | Full entropy (without top-p clamping) |
| `actor/entropy_loss` | Per micro-batch | float | Same as `actor/entropy` (legacy alias) |
| `actor/entropy_mask_empty` | Per micro-batch | 0 or 1 | Whether the response mask had no valid tokens |
| `actor/entropy_nan` | Per micro-batch | 0 or 1 | Whether entropy computation produced NaN/Inf |

### Adaptive Entropy

These metrics are logged when adaptive entropy is enabled (`entropy_coeff_low` and `entropy_coeff_high` are both set).

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| `actor/entropy_coeff_used` | Per mini-batch | float | The entropy coefficient used for this mini-batch |
| `actor/entropy_coeff_final` | Per step | float | Final entropy coefficient at the end of the training step |
| `actor/mini_batch_entropy_avg` | Per mini-batch | float | Weighted average entropy across micro-batches (used for coefficient update) |
| `actor/entropy_coeff_reset` | Per micro-batch | 0 or 1 | Whether the coefficient was reset due to non-finite value |

**How Adaptive Entropy Works:**

The adaptive entropy coefficient adjusts based on the policy's actual entropy:
- If entropy falls **below** `entropy_low`: coefficient increases → encourages exploration
- If entropy exceeds **above** `entropy_high`: coefficient decreases → allows exploitation
- The coefficient is clamped within `[entropy_coeff_low, entropy_coeff_high]`
- Updates occur once per mini-batch (per optimizer step) using aggregated entropy

**Configuration:**

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.01        # Initial/fallback coefficient
    entropy_coeff_low: 0.001   # Minimum adaptive coefficient (required to enable)
    entropy_coeff_high: 0.1    # Maximum adaptive coefficient (required to enable)
    entropy_low: 0.5           # Target entropy lower bound
    entropy_high: 2.0          # Target entropy upper bound
    entropy_coeff_lr: 0.01     # Learning rate for coefficient updates
```

### KL Divergence

These metrics are logged when `use_kl_loss: true`.

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| `actor/kl_loss` | Per micro-batch | float | KL divergence loss between policy and reference |
| `actor/kl_coef` | Per micro-batch | float | KL loss coefficient |

---

## Critic Metrics

*TODO: Document critic metrics*

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| | | | |

---

## Rollout Metrics

*TODO: Document rollout metrics*

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| | | | |

---

## Reward Metrics

*TODO: Document reward metrics*

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| | | | |

---

## System Metrics

*TODO: Document system/performance metrics*

| Metric | Granularity | Type | Description |
|--------|-------------|------|-------------|
| | | | |

---

## Adding New Metrics

When adding new metrics to this document:

1. **Place in the correct category** (Actor, Critic, Rollout, etc.)
2. **Create a subsection** if it's part of a feature (e.g., Adaptive Entropy)
3. **Include all columns**: Metric name, Granularity, Type, Description
4. **Document configuration** if the metric depends on specific settings
5. **Explain the feature** if it's non-obvious how metrics relate to each other

