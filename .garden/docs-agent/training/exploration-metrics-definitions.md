# Exploration Metrics

## State Visitation Metrics (multi_env_evaluator.py)

| Metric | Definition |
|--------|------------|
| `n_distinct_state_actions_valid` | Unique (state_text, valid_action) pairs per seed group |
| `n_distinct_state_actions` | Including invalid->default replacements |
| `distinct_state_actions_per_frame` | distinct / total_frames |
| `distinct_state_actions_valid_coverage` | distinct / (seed_group_size * episode_length) |

## Single-Step Action Diversity (entropy probing)

| Metric | Definition |
|--------|------------|
| `action_entropy` | Shannon entropy over n_samples completions |
| `unique_texts_step` | Distinct raw LLM response strings |
| `unique_executed_actions_step` | Distinct actions after parsing |
| `unique_valid_actions_step` | Distinct correctly-parsed actions |
| `unique_executed_actions_per_unique_text` | Action diversity / text diversity |

## Validity Tracking

| Metric | Definition |
|--------|------------|
| `valid_action_ratio` | valid_actions / attempted_actions |
| `valid_actions_total` | Count of successfully parsed actions |
| `attempted_actions_total` | Total action attempts |

## Implementation Notes
- Seed grouping (`seed_group_size`) ensures same initial state for GRPO groups
- `freeze_completed_episodes=True` prevents cross-batch contamination
- Action entropy measured at configurable steps (`measure_at_steps`: start/random/every_n)
