"""Adaptive epsilon-greedy exploration based on reward trend.

Inverted exploration schedule for LLMs with pretrained priors:
  - Consolidate first (low epsilon when reward is improving)
  - Explore when stuck (high epsilon when reward plateaus/declines)
  - Consolidate discoveries (low epsilon when improvement resumes)

Epsilon is computed as: epsilon_max * sigmoid(-k * normalized_slope)
where slope is the linear regression slope over a sliding window of
reward values, normalized by the window's own std for scale invariance.
"""

import math
from collections import deque


class AdaptiveEpsilon:
    def __init__(self, epsilon_max: float, window_size: int, k: float):
        self.epsilon_max = epsilon_max
        self.window_size = window_size
        self.k = k

        self.reward_buffer = deque(maxlen=window_size)
        self.current_epsilon = 0.0
        self._step_count = 0
        self._last_slope = 0.0

    def update(self, batch_mean_reward: float) -> float:
        """Feed one batch mean reward. Returns updated epsilon."""
        self._step_count += 1
        self.reward_buffer.append(batch_mean_reward)

        if len(self.reward_buffer) < self.window_size:
            self.current_epsilon = 0.0
            return self.current_epsilon

        # Normalize by window-local std for scale invariance
        values = list(self.reward_buffer)
        std = _std(values)
        if std > 1e-8:
            mean = sum(values) / len(values)
            norm_rewards = [(r - mean) / std for r in values]
        else:
            norm_rewards = values

        slope = self._linear_regression_slope(norm_rewards)
        self._last_slope = slope

        # Positive slope (improving) → low epsilon; zero/negative → high epsilon
        self.current_epsilon = self.epsilon_max * _sigmoid(-self.k * slope)
        return self.current_epsilon

    @staticmethod
    def _linear_regression_slope(values):
        """Least-squares slope over indexed values. O(n) single pass."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(values):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx
        if den == 0:
            return 0.0
        return num / den

    def get_metrics(self) -> dict:
        return {
            'adaptive_epsilon/value': self.current_epsilon,
            'adaptive_epsilon/slope_normalised': self._last_slope,
            'adaptive_epsilon/buffer_fill': len(self.reward_buffer) / self.window_size,
        }


def _std(values):
    """Population std of a list of floats."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
