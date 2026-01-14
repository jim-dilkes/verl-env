"""Tests for FastSnake action extraction functions.

Tests both single-action (<action>) and multi-action (<decision>) extraction modes.

Run with: pytest tests/envs/test_fastsnake_extraction.py -v
"""

import pytest
from verl.envs.environments.FastSnake.base import FastSnakeLLMAgentsWrapper


class TestExtractActionFromXmlTag:
    """Tests for extract_action_from_xml_tag (single-action mode)."""

    def test_simple_action(self):
        """Basic <action>X</action> extraction."""
        text = "<action>up</action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "up"

    def test_action_with_think_tag(self):
        """Action with preceding think tag."""
        text = "<think>I should go up</think><action>up</action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "up"

    def test_action_uppercase_content(self):
        """Action content should be lowercased."""
        text = "<action>UP</action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "up"

    def test_action_mixed_case(self):
        """Mixed case action content."""
        text = "<action>Left</action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "left"

    def test_action_with_whitespace(self):
        """Whitespace inside tag should be stripped."""
        text = "<action>  down  </action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "down"

    def test_action_with_newlines(self):
        """Newlines inside tag should be stripped."""
        text = "<action>\n  right\n</action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "right"

    def test_no_action_tag(self):
        """No action tag returns None."""
        text = "I will go up"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result is None

    def test_unclosed_action_tag(self):
        """Unclosed tag - extraction still works due to split-based parsing."""
        # Note: This is a quirk of the implementation - it splits on opening tag
        # and takes content before closing tag (or end of string if no closing tag)
        text = "<action>up"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "up"  # Extracts everything after opening tag

    def test_empty_action_tag(self):
        """Empty action tag returns empty string."""
        text = "<action></action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == ""

    def test_multiple_action_tags_takes_first(self):
        """Multiple action tags - takes first one."""
        text = "<action>up</action> then <action>down</action>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "up"

    def test_custom_tag_name(self):
        """Custom tag name parameter."""
        text = "<move>left</move>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text, tag="move")
        assert result == "left"

    def test_decision_tag_not_matched(self):
        """<decision> tag not matched by action extractor."""
        text = "<decision>up</decision>"
        result = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result is None


class TestExtractDecisionFromXml:
    """Tests for extract_decision_from_xml (multi-action mode)."""

    def test_simple_decision(self):
        """Basic <decision>X</decision> extraction."""
        text = "<decision>up</decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "up"

    def test_decision_with_reasoning(self):
        """Decision with multi-action reasoning format."""
        text = """<actions>
<action name=\"up\"><reasoning>Going up is safe</reasoning></action>
<action name=\"down\"><reasoning>Going down hits wall</reasoning></action>
</actions>
<decision>up</decision>"""
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "up"

    def test_decision_uppercase_content(self):
        """Decision content should be lowercased."""
        text = "<decision>LEFT</decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "left"

    def test_decision_mixed_case(self):
        """Mixed case decision content."""
        text = "<decision>Down</decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "down"

    def test_decision_with_whitespace(self):
        """Whitespace inside tag should be stripped."""
        text = "<decision>  right  </decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "right"

    def test_decision_with_newlines(self):
        """Newlines inside tag should be stripped."""
        text = "<decision>\nup\n</decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "up"

    def test_no_decision_tag(self):
        """No decision tag returns None."""
        text = "I choose up"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result is None

    def test_unclosed_decision_tag(self):
        """Unclosed tag - extraction still works due to split-based parsing."""
        # Note: This is a quirk of the implementation - it splits on opening tag
        # and takes content before closing tag (or end of string if no closing tag)
        text = "<decision>up"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "up"  # Extracts everything after opening tag

    def test_empty_decision_tag(self):
        """Empty decision tag returns empty string."""
        text = "<decision></decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == ""

    def test_multiple_decision_tags_takes_first(self):
        """Multiple decision tags - takes first one (note: rewrite_decision_tag replaces last)."""
        text = "<decision>up</decision> then <decision>down</decision>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "up"

    def test_action_tag_not_matched(self):
        """<action> tag not matched by decision extractor."""
        text = "<action>up</action>"
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result is None

    def test_full_multi_action_response(self):
        """Full multi-action response from model."""
        text = """<actions>
<action name=\"up\">Moving up from (6, 5) would take the head to (6, 6). This position is not occupied by an enemy or your body, and it is within bounds. It is a safe move.</action>
<action name=\"down\">Moving down to (6, 4) is safe and within bounds. It is not near any enemy or body segment.</action>
<action name=\"left\">Moving left to (5, 5) is safe and within bounds. This move is a step toward the apple at (5, 1).</action>
<action name=\"right\">Moving right to (7, 5) is safe and within bounds. However, this direction moves away from the apple.</action>
</actions>
<decision>left</decision>"""
        result = FastSnakeLLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "left"


class TestExtractActionInstance:
    """Tests for extract_action_instance method (instance method with mode switching)."""

    @pytest.fixture
    def mock_env(self):
        """Create a minimal mock environment that satisfies wrapper requirements."""
        import gymnasium as gym
        from unittest.mock import MagicMock, patch

        # Create a real gym.Env subclass for the mock
        class MockFastSnakeEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = gym.spaces.Discrete(4)
                self.observation_space = gym.spaces.Dict({})

            def step(self, action):
                return {}, 0, False, False, {}

            def reset(self, seed=None, options=None):
                return {}, {}

            def game_state_text(self):
                return "mock game state"

        return MockFastSnakeEnv()

    def test_multi_action_mode_extracts_decision(self, mock_env):
        """In multi-action mode, extracts from <decision> tag."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        text = "<action>down</action><decision>up</decision>"
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "up"
        assert is_valid is True

    def test_single_action_mode_extracts_action(self, mock_env):
        """In single-action mode, extracts from <action> tag."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=False)

        text = "<action>down</action><decision>up</decision>"
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "down"
        assert is_valid is True

    def test_multi_action_mode_invalid_when_no_decision(self, mock_env):
        """In multi-action mode, invalid if no <decision> tag."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        text = "<action>up</action>"
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "__invalid__"
        assert is_valid is False
        assert executed == "up"  # Falls back to default action

    def test_single_action_mode_invalid_when_no_action(self, mock_env):
        """In single-action mode, invalid if no <action> tag."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=False)

        text = "<decision>up</decision>"
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "__invalid__"
        assert is_valid is False

    def test_valid_action_ratio_metric(self, mock_env):
        """Metrics include valid_action_ratio."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        # Valid action
        _, _, _, _, metrics = wrapper.extract_action_instance("<decision>left</decision>")
        assert metrics["behavior/valid_action_ratio"] == 1.0

        # Invalid action
        _, _, _, _, metrics = wrapper.extract_action_instance("no tags here")
        assert metrics["behavior/valid_action_ratio"] == 0.0

    def test_invalid_action_value_not_in_action_space(self, mock_env):
        """Action value not in action space is invalid."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        text = "<decision>jump</decision>"  # 'jump' not in action space
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "jump"
        assert is_valid is False
        assert executed == "up"  # Falls back to default

    def test_all_valid_actions_return_is_valid_true(self, mock_env):
        """Every valid action should return is_valid=True."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        valid_actions = ["up", "down", "left", "right"]
        for action in valid_actions:
            text = f"<decision>{action}</decision>"
            full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

            assert extracted == action, f"Expected extracted='{action}', got '{extracted}'"
            assert is_valid is True, f"Expected is_valid=True for action '{action}', got {is_valid}"
            assert executed == action, f"Expected executed='{action}', got '{executed}'"

    def test_uppercase_actions_normalized_and_valid(self, mock_env):
        """Uppercase action content should be normalized and valid."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        # Test various case variations
        test_cases = [
            ("UP", "up"),
            ("Down", "down"),
            ("LEFT", "left"),
            ("RiGhT", "right"),
        ]
        for input_action, expected in test_cases:
            text = f"<decision>{input_action}</decision>"
            full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

            assert extracted == expected, f"Expected '{expected}', got '{extracted}'"
            assert is_valid is True, f"Expected is_valid=True for '{input_action}' -> '{expected}'"

    def test_whitespace_stripped_and_valid(self, mock_env):
        """Whitespace should be stripped and action should be valid."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        test_cases = [
            "  up  ",
            "\tdown\t",
            "\n left \n",
            "  right",
        ]
        expected = ["up", "down", "left", "right"]

        for input_action, exp in zip(test_cases, expected):
            text = f"<decision>{input_action}</decision>"
            full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

            assert extracted == exp, f"Expected '{exp}', got '{extracted}'"
            assert is_valid is True, f"Expected is_valid=True for whitespace case"

    def test_language_action_space_contains_expected_actions(self, mock_env):
        """Verify the action space contains exactly the expected actions."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        expected_actions = {"up", "down", "left", "right"}
        actual_actions = set(wrapper.language_action_space)

        assert actual_actions == expected_actions, f"Action space mismatch: {actual_actions} != {expected_actions}"

    def test_extracted_action_in_action_space_implies_valid(self, mock_env):
        """If extracted action is in action space, is_valid must be True."""
        wrapper = FastSnakeLLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        # This tests the logical consistency of the check
        for action in wrapper.language_action_space:
            text = f"<decision>{action}</decision>"
            full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

            # Core invariant: if extracted is in action_space, is_valid must be True
            if extracted in wrapper.language_action_space:
                assert is_valid is True, f"BUG: extracted='{extracted}' is in action_space but is_valid={is_valid}"


def test_production_stack_envwrapper_exposes_extract_action_instance():
    """Regression test: production uses make_env() which wraps env in EnvWrapper.

    VecEnv worker selects extraction via:
        extract_fn = getattr(env, 'extract_action_instance', env.extract_action)

    If EnvWrapper doesn't forward extract_action_instance, multi-action mode silently
    falls back to extract_action (single-action parsing), making <decision> outputs
    look invalid.
    """
    from omegaconf import OmegaConf
    from verl.envs.environments import make_env

    config_dict = {
        "envs": {
            "env_name": "fastsnake",
            "n_rollouts": 1,
            "format_penalty": 0.1,
            "fastsnake_kwargs": {
                "width": 5,
                "height": 5,
                "max_rounds": 10,
                "num_external_snakes": 1,
                "num_random_snakes": 0,
                "num_apples": 0,
                "num_bananas": 0,
                "num_fires": 0,
                "print_visualization": False,
                "print_coordinates": False,
                "print_axes": False,
            },
        },
        "prompt": {"prompt": {"multi_action_reasoning": True, "epsilon": 0.0}},
    }

    config = OmegaConf.create(config_dict)
    assert bool(getattr(config.prompt.prompt, "multi_action_reasoning", False)) is True

    env = make_env("fastsnake", "default", config)

    # Sanity checks: ensure the wrapper stack actually supports instance extraction
    assert hasattr(env, "extract_action_instance"), "Outer EnvWrapper must expose extract_action_instance"
    assert hasattr(env.env, "extract_action_instance"), "Inner FastSnake wrapper must expose extract_action_instance"
    assert bool(getattr(env.env, "multi_action_reasoning", False)) is True

    extract_fn = getattr(env, "extract_action_instance", env.extract_action)
    full, extracted, executed, is_valid, metrics = extract_fn("<decision>up</decision>")

    assert extracted == "up"
    assert executed == "up"
    assert is_valid is True
    assert metrics["behavior/valid_action_ratio"] == 1.0
