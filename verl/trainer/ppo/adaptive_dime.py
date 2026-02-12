"""Adaptive DIME supplement ratio based on reward trend.

Same inverted schedule as AdaptiveEpsilon: sliding window, linear
regression slope, sigmoid mapping. But controls the DIME supplement
ratio instead of epsilon.

- Improving rewards → low supplement_prob (consolidate with clean rollouts)
- Stagnating/declining → high supplement_prob (more focus-instruction diversity)

Reuses math helpers from adaptive_epsilon module.
"""

from collections import deque

from verl.trainer.ppo.adaptive_epsilon import _std, _sigmoid, AdaptiveEpsilon


class AdaptiveDIME:
    def __init__(
        self,
        supplement_min: float,
        supplement_max: float,
        window_size: int,
        k: float,
        inflection: float = 0.0,
    ):
        self.supplement_min = supplement_min
        self.supplement_max = supplement_max
        self.window_size = window_size
        self.k = k
        self.inflection = inflection

        self.reward_buffer: deque[float] = deque(maxlen=window_size)
        self.current_supplement_prob = supplement_min
        self._last_slope = 0.0

    def update(self, batch_mean_reward: float) -> float:
        """Feed one batch's base-only mean reward (no focus instructions).

        Caller should skip this call when no base episodes exist rather
        than passing a fallback value, to avoid contaminating the window.
        Returns updated supplement_prob.
        """
        self.reward_buffer.append(batch_mean_reward)

        if len(self.reward_buffer) < self.window_size:
            return self.current_supplement_prob

        values = list(self.reward_buffer)
        std = _std(values)
        if std > 1e-8:
            mean = sum(values) / len(values)
            norm_rewards = [(r - mean) / std for r in values]
        else:
            norm_rewards = values

        slope = AdaptiveEpsilon._linear_regression_slope(norm_rewards)
        self._last_slope = slope

        # Positive slope (improving) → low supplement; zero/negative → high
        raw = (self.supplement_max - self.supplement_min) * _sigmoid(-self.k * (slope - self.inflection))
        self.current_supplement_prob = self.supplement_min + raw
        return self.current_supplement_prob

    def get_no_supplement_prob(self) -> float:
        """Convenience: 1 - supplement_prob for sample_focus_for_episode."""
        return 1.0 - self.current_supplement_prob

    def get_metrics(self) -> dict:
        return {
            "dime/adaptive_supplement_prob": self.current_supplement_prob,
            "dime/adaptive_slope": self._last_slope,
            "dime/adaptive_inflection": self.inflection,
            "dime/adaptive_buffer_fill": len(self.reward_buffer) / self.window_size,
        }


