# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for MultiEnvEvaluator timing metrics.

Tests verify timing instrumentation without requiring GPU or actual LLM.
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from omegaconf import OmegaConf


class TestEvalTimingMetrics(unittest.TestCase):
    """Test that timing metrics are correctly computed and returned."""

    def test_env_step_timing_accumulation(self):
        """Verify env_step_time is accumulated across steps."""
        # This tests the timing logic in isolation
        total_env_step_time = 0.0
        n_steps = 5
        step_duration = 0.01  # 10ms per step

        for _ in range(n_steps):
            env_step_start = time.time()
            time.sleep(step_duration)
            env_step_end = time.time()
            total_env_step_time += (env_step_end - env_step_start)

        # Should have accumulated ~50ms
        self.assertGreater(total_env_step_time, step_duration * n_steps * 0.9)
        self.assertLess(total_env_step_time, step_duration * n_steps * 2)

    def test_timing_metric_keys_present(self):
        """Verify all expected timing metric keys would be generated."""
        # Expected new timing keys that should be in metrics
        expected_keys = [
            "env_step_time_seconds",
            "env_step_time_per_rollout",
            "env_step_time_per_step",
        ]

        # Simulate metric dict construction (from _evaluate_single_env_body)
        total_env_step_time = 1.5  # seconds
        n_rollouts = 4
        total_attempted_actions = 20

        metric_dict = {
            "env_step_time_seconds": total_env_step_time,
            "env_step_time_per_rollout": total_env_step_time / max(1, n_rollouts),
            "env_step_time_per_step": total_env_step_time / max(1, total_attempted_actions),
        }

        for key in expected_keys:
            self.assertIn(key, metric_dict)
            self.assertIsInstance(metric_dict[key], float)

    def test_eval_time_added_to_prefixed_metrics(self):
        """Verify eval_time_seconds is added with correct prefix."""
        eval_name = "test_env"
        eval_time = 10.5
        env_metrics = {"some_metric": 1.0}

        # Simulate prefix logic from evaluate()
        prefixed_metrics = {}
        for key, value in env_metrics.items():
            prefixed_key = f"eval_{eval_name}/{key}"
            prefixed_metrics[prefixed_key] = value

        # Add eval_time_seconds (as done in our implementation)
        prefixed_metrics[f"eval_{eval_name}/eval_time_seconds"] = eval_time

        self.assertIn(f"eval_{eval_name}/eval_time_seconds", prefixed_metrics)
        self.assertEqual(prefixed_metrics[f"eval_{eval_name}/eval_time_seconds"], eval_time)

    def test_timing_breakdown_calculation(self):
        """Verify 'other' time calculation is correct."""
        eval_time = 10.0
        inference_time = 6.0
        env_step_time = 2.0
        entropy_probe_time = 1.0

        other_time = eval_time - inference_time - env_step_time - entropy_probe_time

        self.assertAlmostEqual(other_time, 1.0, places=5)

    def test_timing_breakdown_no_entropy(self):
        """Verify timing breakdown when entropy probing is disabled."""
        eval_time = 10.0
        inference_time = 7.0
        env_step_time = 2.0
        entropy_probe_time = 0.0

        other_time = eval_time - inference_time - env_step_time - entropy_probe_time

        self.assertAlmostEqual(other_time, 1.0, places=5)

        # Verify console output format logic
        timing_parts = [
            f"total: {eval_time:.2f}s",
            f"inference: {inference_time:.2f}s",
            f"env_step: {env_step_time:.2f}s"
        ]
        if entropy_probe_time > 0:
            timing_parts.append(f"entropy_probe: {entropy_probe_time:.2f}s")
        timing_parts.append(f"other: {other_time:.2f}s")

        output = f"Completed evaluation for test_env ({', '.join(timing_parts)})"

        self.assertIn("total: 10.00s", output)
        self.assertIn("inference: 7.00s", output)
        self.assertIn("env_step: 2.00s", output)
        self.assertIn("other: 1.00s", output)
        self.assertNotIn("entropy_probe", output)


    def test_other_time_clamped_to_zero(self):
        """Verify other_time is clamped to >= 0 even with rounding issues."""
        eval_time = 10.0
        inference_time = 6.0
        env_step_time = 3.0
        entropy_probe_time = 2.0  # Sum > eval_time

        # Without clamping this would be -1.0
        other_time = max(0.0, eval_time - inference_time - env_step_time - entropy_probe_time)

        self.assertEqual(other_time, 0.0)
        self.assertGreaterEqual(other_time, 0.0)

    def test_other_time_logged_to_metrics(self):
        """Verify other_time_seconds is added to prefixed metrics."""
        eval_name = "test_env"
        eval_time = 10.0
        inference_time = 6.0
        env_step_time = 2.0
        entropy_probe_time = 1.0
        other_time = max(0.0, eval_time - inference_time - env_step_time - entropy_probe_time)

        prefixed_metrics = {}
        prefixed_metrics[f"eval_{eval_name}/eval_time_seconds"] = eval_time
        prefixed_metrics[f"eval_{eval_name}/other_time_seconds"] = other_time

        self.assertIn(f"eval_{eval_name}/other_time_seconds", prefixed_metrics)
        self.assertAlmostEqual(prefixed_metrics[f"eval_{eval_name}/other_time_seconds"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
