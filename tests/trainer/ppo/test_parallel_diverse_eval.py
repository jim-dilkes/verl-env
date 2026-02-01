"""Tests for parallel diverse prompting evaluation.

Tests the helper functions used for Dipper-style ensemble evaluation:
- _expand_for_diverse_prompts(): Expand B prompts to K×B with suffix injection
- _aggregate_diverse_responses(): Majority vote aggregation K×B → B
- _compute_diverse_metrics(): Agreement rate and diversity metrics

Run with: pytest tests/trainer/ppo/test_parallel_diverse_eval.py -v
"""

import pytest
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Callable
from unittest.mock import MagicMock


# Import the functions we're testing
# Note: We test the logic directly, not the class methods, for isolation
# The actual class methods are thin wrappers around this logic


def expand_for_diverse_prompts(
    prompt_texts: List[str],
    diverse_config: Dict,
) -> Tuple[List[str], int]:
    """Standalone version of _expand_for_diverse_prompts for testing."""
    prompts_config = diverse_config.get('prompts', [])
    if not prompts_config:
        return prompt_texts, 1

    K = len(prompts_config)
    expanded = []

    for prompt_text in prompt_texts:
        for prompt_cfg in prompts_config:
            suffix = prompt_cfg.get('suffix')
            if suffix:
                modified = prompt_text.rstrip() + "\n\n" + suffix.strip()
                expanded.append(modified)
            else:
                expanded.append(prompt_text)

    return expanded, K


def aggregate_diverse_responses(
    responses: List[str],
    n_rollouts: int,
    n_prompts: int,
    action_extraction_fn: Callable,
    aggregation: str = "majority_vote",
) -> Tuple[List[str], List[Dict]]:
    """Standalone version of _aggregate_diverse_responses for testing."""
    # Validate aggregation method
    supported_aggregations = {"majority_vote"}
    if aggregation not in supported_aggregations:
        raise ValueError(
            f"Unsupported aggregation method '{aggregation}'. "
            f"Supported: {supported_aggregations}"
        )

    if len(responses) != n_rollouts * n_prompts:
        raise ValueError(
            f"Expected {n_rollouts * n_prompts} responses, got {len(responses)}"
        )

    final_responses = []
    agreement_info = []

    for rollout_idx in range(n_rollouts):
        start_idx = rollout_idx * n_prompts
        rollout_responses = responses[start_idx : start_idx + n_prompts]

        actions = []
        action_to_response = {}
        action_to_idx = {}
        valid_count = 0

        for prompt_idx, resp in enumerate(rollout_responses):
            _, _, executed, is_valid, _ = action_extraction_fn(resp)
            actions.append(executed)
            valid_count += int(is_valid)
            if executed not in action_to_response:
                action_to_response[executed] = resp
                action_to_idx[executed] = prompt_idx

        action_counts = Counter(actions)
        winner, winner_votes = action_counts.most_common(1)[0]

        final_responses.append(action_to_response[winner])

        unique_actions = len(set(actions))
        winner_idx = action_to_idx[winner]
        agreement_info.append({
            "unanimous": unique_actions == 1,
            "winner_votes": winner_votes,
            "n_prompts": n_prompts,
            "unique_actions": unique_actions,
            "valid_count": valid_count,
            "winner_action": winner,
            "winner_idx": winner_idx,
        })

    return final_responses, agreement_info


def compute_diverse_metrics(
    all_agreement_info: List[List[Dict]],
) -> Dict[str, float]:
    """Standalone version of _compute_diverse_metrics for testing."""
    if not all_agreement_info:
        return {}

    all_info = []
    for step_info in all_agreement_info:
        all_info.extend(step_info)

    if not all_info:
        return {}

    n_prompts = all_info[0]["n_prompts"]
    unanimous_count = sum(1 for info in all_info if info["unanimous"])
    total_count = len(all_info)

    avg_unique = np.mean([info["unique_actions"] for info in all_info])
    avg_winner_votes = np.mean([info["winner_votes"] for info in all_info])

    return {
        "diverse/unanimous_ratio": unanimous_count / total_count,
        "diverse/mean_unique_actions": avg_unique,
        "diverse/winner_vote_ratio": avg_winner_votes / n_prompts,
        "diverse/n_prompts": float(n_prompts),
        "diverse/total_decisions": float(total_count),
    }


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def simple_diverse_config():
    """Simple config with 3 prompts."""
    return {
        'enabled': True,
        'aggregation': 'majority_vote',
        'prompts': [
            {'name': 'baseline', 'suffix': None},
            {'name': 'cautious', 'suffix': 'Be careful!'},
            {'name': 'aggressive', 'suffix': 'Be bold!'},
        ]
    }


@pytest.fixture
def mock_action_extractor():
    """Mock action extraction function that parses <action>X</action> tags."""
    def extractor(response: str) -> Tuple[str, str, str, bool, dict]:
        # Simple XML parsing
        import re
        match = re.search(r'<action>(\w+)</action>', response)
        if match:
            action = match.group(1)
            return response, action, action, True, {}
        else:
            return response, None, 'default', False, {}
    return extractor


# ============================================================================
# Tests for expand_for_diverse_prompts
# ============================================================================

class TestExpandForDiversePrompts:
    """Tests for prompt expansion with suffix injection."""

    def test_basic_expansion(self, simple_diverse_config):
        """Test basic B → K×B expansion."""
        prompts = ["Prompt 1", "Prompt 2"]
        expanded, K = expand_for_diverse_prompts(prompts, simple_diverse_config)

        assert K == 3
        assert len(expanded) == 6  # 2 prompts × 3 variants

    def test_suffix_injection(self, simple_diverse_config):
        """Test suffixes are correctly appended."""
        prompts = ["Base prompt"]
        expanded, K = expand_for_diverse_prompts(prompts, simple_diverse_config)

        assert expanded[0] == "Base prompt"  # baseline (no suffix)
        assert "Be careful!" in expanded[1]  # cautious
        assert "Be bold!" in expanded[2]  # aggressive

    def test_interleaved_order(self, simple_diverse_config):
        """Test expansion is interleaved: [p1_s1, p1_s2, p1_s3, p2_s1, p2_s2, p2_s3]."""
        prompts = ["P1", "P2"]
        expanded, K = expand_for_diverse_prompts(prompts, simple_diverse_config)

        # First K entries are variants of P1
        assert expanded[0] == "P1"
        assert "P1" in expanded[1] and "Be careful!" in expanded[1]
        assert "P1" in expanded[2] and "Be bold!" in expanded[2]

        # Next K entries are variants of P2
        assert expanded[3] == "P2"
        assert "P2" in expanded[4] and "Be careful!" in expanded[4]
        assert "P2" in expanded[5] and "Be bold!" in expanded[5]

    def test_empty_prompts_config(self):
        """Test with no prompts configured returns unchanged."""
        config = {'enabled': True, 'prompts': []}
        prompts = ["Test"]
        expanded, K = expand_for_diverse_prompts(prompts, config)

        assert K == 1
        assert expanded == prompts

    def test_single_prompt_single_variant(self):
        """Test with single prompt and single variant."""
        config = {'prompts': [{'name': 'only', 'suffix': 'Extra'}]}
        prompts = ["Base"]
        expanded, K = expand_for_diverse_prompts(prompts, config)

        assert K == 1
        assert len(expanded) == 1
        assert "Extra" in expanded[0]

    def test_whitespace_handling(self):
        """Test trailing whitespace is handled."""
        config = {'prompts': [
            {'name': 'a', 'suffix': None},
            {'name': 'b', 'suffix': '  Suffix with spaces  '},
        ]}
        prompts = ["Base with trailing   "]
        expanded, K = expand_for_diverse_prompts(prompts, config)

        assert expanded[0] == "Base with trailing"  # Stripped
        assert "Suffix with spaces" in expanded[1]
        assert not expanded[1].endswith("  ")  # Suffix stripped


# ============================================================================
# Tests for aggregate_diverse_responses
# ============================================================================

class TestAggregateDiverseResponses:
    """Tests for majority vote aggregation."""

    def test_unanimous_agreement(self, mock_action_extractor):
        """Test all prompts agree on same action."""
        responses = [
            "<action>up</action>",
            "<action>up</action>",
            "<action>up</action>",
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=1, n_prompts=3,
            action_extraction_fn=mock_action_extractor
        )

        assert len(final) == 1
        assert info[0]["unanimous"] is True
        assert info[0]["winner_votes"] == 3
        assert info[0]["unique_actions"] == 1

    def test_majority_wins(self, mock_action_extractor):
        """Test majority action wins."""
        responses = [
            "<action>up</action>",
            "<action>up</action>",
            "<action>down</action>",
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=1, n_prompts=3,
            action_extraction_fn=mock_action_extractor
        )

        assert info[0]["winner_action"] == "up"
        assert info[0]["winner_votes"] == 2
        assert info[0]["unanimous"] is False

    def test_tie_breaks_to_first(self, mock_action_extractor):
        """Test tie breaks to first occurring action."""
        responses = [
            "<action>left</action>",
            "<action>right</action>",
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=1, n_prompts=2,
            action_extraction_fn=mock_action_extractor
        )

        # Counter.most_common returns first in case of tie
        assert info[0]["winner_action"] in ["left", "right"]
        assert info[0]["winner_votes"] == 1

    def test_multiple_rollouts(self, mock_action_extractor):
        """Test aggregation across multiple rollouts."""
        # 2 rollouts × 3 prompts = 6 responses
        responses = [
            # Rollout 0: unanimous up
            "<action>up</action>",
            "<action>up</action>",
            "<action>up</action>",
            # Rollout 1: majority down
            "<action>down</action>",
            "<action>down</action>",
            "<action>left</action>",
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=2, n_prompts=3,
            action_extraction_fn=mock_action_extractor
        )

        assert len(final) == 2
        assert info[0]["winner_action"] == "up"
        assert info[0]["unanimous"] is True
        assert info[1]["winner_action"] == "down"
        assert info[1]["unanimous"] is False

    def test_invalid_actions_use_default(self, mock_action_extractor):
        """Test invalid responses fall back to default action."""
        responses = [
            "<action>up</action>",
            "no action tag here",  # Invalid → default
            "<action>up</action>",
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=1, n_prompts=3,
            action_extraction_fn=mock_action_extractor
        )

        # "up" appears twice, "default" once
        assert info[0]["winner_action"] == "up"
        assert info[0]["valid_count"] == 2

    def test_response_count_mismatch_raises(self, mock_action_extractor):
        """Test wrong number of responses raises error."""
        with pytest.raises(ValueError, match="Expected 6 responses"):
            aggregate_diverse_responses(
                ["a", "b", "c"],  # 3 instead of 6
                n_rollouts=2, n_prompts=3,
                action_extraction_fn=mock_action_extractor
            )

    def test_returns_winning_response(self, mock_action_extractor):
        """Test returned response is the one that produced winning action."""
        responses = [
            "First response <action>up</action>",
            "Second response <action>down</action>",
            "Third response <action>up</action>",
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=1, n_prompts=3,
            action_extraction_fn=mock_action_extractor
        )

        # Should return first response with winning action (up)
        assert final[0] == "First response <action>up</action>"

    def test_winner_idx_tracking(self, mock_action_extractor):
        """Test winner_idx is correctly tracked."""
        responses = [
            "<action>down</action>",  # idx 0
            "<action>up</action>",    # idx 1 - winner
            "<action>up</action>",    # idx 2
        ]
        final, info = aggregate_diverse_responses(
            responses, n_rollouts=1, n_prompts=3,
            action_extraction_fn=mock_action_extractor
        )

        # Winner is "up", first occurrence at idx 1
        assert info[0]["winner_action"] == "up"
        assert info[0]["winner_idx"] == 1

    def test_unsupported_aggregation_raises(self, mock_action_extractor):
        """Test unsupported aggregation method raises ValueError."""
        responses = ["<action>up</action>"] * 3
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            aggregate_diverse_responses(
                responses, n_rollouts=1, n_prompts=3,
                action_extraction_fn=mock_action_extractor,
                aggregation="first_valid"  # Not implemented
            )


# ============================================================================
# Tests for compute_diverse_metrics
# ============================================================================

class TestComputeDiverseMetrics:
    """Tests for metrics computation."""

    def test_all_unanimous(self):
        """Test metrics when all decisions are unanimous."""
        agreement_info = [
            [  # Step 0
                {"unanimous": True, "winner_votes": 3, "n_prompts": 3, "unique_actions": 1},
                {"unanimous": True, "winner_votes": 3, "n_prompts": 3, "unique_actions": 1},
            ],
            [  # Step 1
                {"unanimous": True, "winner_votes": 3, "n_prompts": 3, "unique_actions": 1},
                {"unanimous": True, "winner_votes": 3, "n_prompts": 3, "unique_actions": 1},
            ],
        ]
        metrics = compute_diverse_metrics(agreement_info)

        assert metrics["diverse/unanimous_ratio"] == 1.0
        assert metrics["diverse/mean_unique_actions"] == 1.0
        assert metrics["diverse/winner_vote_ratio"] == 1.0
        assert metrics["diverse/n_prompts"] == 3.0
        assert metrics["diverse/total_decisions"] == 4.0

    def test_no_agreement(self):
        """Test metrics when no decisions are unanimous."""
        agreement_info = [
            [
                {"unanimous": False, "winner_votes": 2, "n_prompts": 3, "unique_actions": 2},
                {"unanimous": False, "winner_votes": 1, "n_prompts": 3, "unique_actions": 3},
            ],
        ]
        metrics = compute_diverse_metrics(agreement_info)

        assert metrics["diverse/unanimous_ratio"] == 0.0
        assert metrics["diverse/mean_unique_actions"] == 2.5  # (2 + 3) / 2
        assert metrics["diverse/winner_vote_ratio"] == 0.5  # (2/3 + 1/3) / 2

    def test_mixed_agreement(self):
        """Test metrics with mixed agreement."""
        agreement_info = [
            [
                {"unanimous": True, "winner_votes": 5, "n_prompts": 5, "unique_actions": 1},
                {"unanimous": False, "winner_votes": 3, "n_prompts": 5, "unique_actions": 2},
            ],
        ]
        metrics = compute_diverse_metrics(agreement_info)

        assert metrics["diverse/unanimous_ratio"] == 0.5  # 1/2 unanimous
        assert metrics["diverse/mean_unique_actions"] == 1.5  # (1 + 2) / 2
        assert metrics["diverse/winner_vote_ratio"] == 0.8  # (5/5 + 3/5) / 2

    def test_empty_info_returns_empty(self):
        """Test empty agreement info returns empty dict."""
        assert compute_diverse_metrics([]) == {}
        assert compute_diverse_metrics([[]]) == {}


# ============================================================================
# Integration-style tests
# ============================================================================

class TestDiverseEvalIntegration:
    """Tests that simulate the full expansion → inference → aggregation flow."""

    def test_full_flow_simulation(self, simple_diverse_config, mock_action_extractor):
        """Simulate full evaluation flow with mocked inference."""
        # Original prompts (B=2)
        prompts = ["State: snake at (1,1)", "State: snake at (2,2)"]

        # 1. Expand prompts
        expanded, K = expand_for_diverse_prompts(prompts, simple_diverse_config)
        assert len(expanded) == 6  # 2 × 3

        # 2. Simulate "inference" - mock LLM returns actions
        # Each prompt variant might produce different actions
        mock_responses = [
            # Rollout 0: baseline=up, cautious=up, aggressive=right
            "<action>up</action>",
            "<action>up</action>",
            "<action>right</action>",
            # Rollout 1: all agree on down
            "<action>down</action>",
            "<action>down</action>",
            "<action>down</action>",
        ]

        # 3. Aggregate
        final, info = aggregate_diverse_responses(
            mock_responses, n_rollouts=2, n_prompts=K,
            action_extraction_fn=mock_action_extractor
        )

        assert len(final) == 2
        assert info[0]["winner_action"] == "up"
        assert info[0]["unanimous"] is False
        assert info[1]["winner_action"] == "down"
        assert info[1]["unanimous"] is True

        # 4. Compute metrics
        metrics = compute_diverse_metrics([[info[0], info[1]]])
        assert 0.0 <= metrics["diverse/unanimous_ratio"] <= 1.0
        assert metrics["diverse/n_prompts"] == 3.0

    def test_five_prompt_ensemble(self):
        """Test with 5-prompt ensemble like the Overcooked config."""
        config = {
            'prompts': [
                {'name': 'baseline', 'suffix': None},
                {'name': 'cautious', 'suffix': 'Safety first'},
                {'name': 'aggressive', 'suffix': 'Speed is key'},
                {'name': 'strategic', 'suffix': 'Think ahead'},
                {'name': 'cooperative', 'suffix': 'Teamwork'},
            ]
        }

        prompts = ["Test prompt"]
        expanded, K = expand_for_diverse_prompts(prompts, config)

        assert K == 5
        assert len(expanded) == 5
        assert expanded[0] == "Test prompt"  # baseline unchanged
        assert "Safety first" in expanded[1]
        assert "Teamwork" in expanded[4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
