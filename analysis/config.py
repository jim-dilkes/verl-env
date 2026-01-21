"""Configuration constants for wandb analysis tooling."""

from pathlib import Path

# Default cache directory
CACHE_DIR = Path(__file__).parent / "data"

# Default config key allowlist (from experiment-configs-params.md)
DEFAULT_CONFIG_ALLOWLIST = [
    # Environment
    "envs.env_name",
    "envs.task",
    "envs.n_rollouts",
    "envs.freeze_completed_episodes",
    "envs.format_penalty",
    "envs.binary_reward",
    "envs.group_rollout_size",
    "envs.group_initial_seed",
    # Captioner
    "envs.captioner.type",
    "envs.captioner.max_text_history",
    "envs.captioner.max_image_history",
    # Prompt
    "prompt.prompt.multi_action_reasoning",
    "prompt.prompt.epsilon",
    # Algorithm
    "algorithm.adv_estimator",
    "algorithm.step_gamma",
    "algorithm.step_lam",
    "algorithm.token_gamma",
    "algorithm.token_lam",
    # Training
    "data.train_batch_size",
    "trainer.critic_warmup",
    "trainer.test_freq",
    # Model
    "model.path",
    "model.name",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.name",
]

# Config prefixes to always ignore
IGNORE_CONFIG_PREFIXES = [
    "_wandb",
    "wandb_",
    "_runtime",
    "_timestamp",
]

# Default metrics patterns for comparison
DEFAULT_METRIC_PATTERNS = [
    # Eval metrics
    "eval_*/rewards_mean",
    "eval_*/score_mean",
    "eval_*/traj_length_mean",
    "eval_*/pos_reward_any_prop_mean",
    "eval_*/tokens_per_step",
    # Exploration metrics
    "eval_*/action_entropy_mean",
    "eval_*/n_distinct_state_actions_valid_mean",
    "eval_*/distinct_state_actions_valid_coverage_mean",
    "eval_*/valid_action_ratio",
    # Val metrics
    "val/rewards_mean",
    "val/traj_length_mean",
    "val/pos_reward_total_prop_mean",
    # Learning metrics
    "actor/entropy",
    "actor/pg_loss",
    "actor/ppo_kl",
    "critic/vf_loss",
    "critic/score/mean",
    "critic/rewards/mean",
    # Other
    "generation/success_rate",
]

# Metrics where higher is better (for "best" final mode)
# Key can be full metric name or suffix pattern
HIGHER_IS_BETTER = {
    "rewards_mean": True,
    "score_mean": True,
    "success_rate": True,
    "pos_reward_any_prop_mean": True,
    "traj_length_mean": False,  # shorter trajectories often better
    "tokens_per_step": False,  # fewer tokens = more efficient
    "tokens_per_rollout": False,
}

# Default x-axis key for learning curves
DEFAULT_X_AXIS = "global_step"

# Rate limiting
API_CALL_DELAY = 0.1  # seconds between API calls
LARGE_REQUEST_THRESHOLD = 50  # warn if fetching more runs than this

# Output formatting
DEFAULT_FLOAT_PRECISION = 4
DEFAULT_MAX_COL_WIDTH = 50
