"""Tests for BabyAI wrapper interface contract.

Tests the BabyAILLMAgentsWrapper to ensure it conforms to the standard
environment wrapper interface.

Run with: pytest tests/envs/test_babyai_wrapper.py -v
"""

import random
import pytest

from verl.envs.environments.babyai_text.llm_agents_wrapper import BabyAILLMAgentsWrapper
from verl.envs.environments.babyai_text.clean_lang_wrapper import BABYAI_ACTION_SPACE


class TestExtractActionFromXmlTag:
    """Tests for extract_action_from_xml_tag (single-action mode)."""

    def test_simple_action(self):
        """Basic <action>X</action> extraction."""
        text = "<action>go forward</action>"
        result = BabyAILLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "go forward"

    def test_action_uppercase_content(self):
        """Action content should be lowercased."""
        text = "<action>GO FORWARD</action>"
        result = BabyAILLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "go forward"

    def test_action_with_whitespace(self):
        """Whitespace inside tag should be stripped."""
        text = "<action>  turn left  </action>"
        result = BabyAILLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == "turn left"

    def test_no_action_tag(self):
        """No action tag returns None."""
        text = "I will go forward"
        result = BabyAILLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result is None

    def test_empty_action_tag(self):
        """Empty action tag returns empty string."""
        text = "<action></action>"
        result = BabyAILLMAgentsWrapper.extract_action_from_xml_tag(text)
        assert result == ""


class TestExtractDecisionFromXml:
    """Tests for extract_decision_from_xml (multi-action mode)."""

    def test_simple_decision(self):
        """Basic <decision>X</decision> extraction."""
        text = "<decision>turn right</decision>"
        result = BabyAILLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "turn right"

    def test_decision_with_reasoning(self):
        """Decision with multi-action reasoning format."""
        text = """<actions>
<action name="turn left">Can't see anything to the left</action>
<action name="go forward">Path is clear ahead</action>
</actions>
<decision>go forward</decision>"""
        result = BabyAILLMAgentsWrapper.extract_decision_from_xml(text)
        assert result == "go forward"

    def test_no_decision_tag(self):
        """No decision tag returns None."""
        text = "<action>turn left</action>"
        result = BabyAILLMAgentsWrapper.extract_decision_from_xml(text)
        assert result is None


class TestNormalizeAction:
    """Tests for _normalize_action helper."""

    def test_normalize_none(self):
        assert BabyAILLMAgentsWrapper._normalize_action(None) is None

    def test_normalize_variants(self):
        assert BabyAILLMAgentsWrapper._normalize_action("turnleft") == "turn left"
        assert BabyAILLMAgentsWrapper._normalize_action("turnright") == "turn right"
        assert BabyAILLMAgentsWrapper._normalize_action("goforward") == "go forward"
        assert BabyAILLMAgentsWrapper._normalize_action("pickup") == "pick up"

    def test_normalize_underscores(self):
        assert BabyAILLMAgentsWrapper._normalize_action("turn_left") == "turn left"
        assert BabyAILLMAgentsWrapper._normalize_action("go_forward") == "go forward"

    def test_normalize_lowercase(self):
        assert BabyAILLMAgentsWrapper._normalize_action("TURN LEFT") == "turn left"
        assert BabyAILLMAgentsWrapper._normalize_action("Go Forward") == "go forward"


class TestExtractActionClassmethod:
    """Tests for extract_action classmethod (evaluator compatibility)."""

    def test_returns_5_values(self):
        """extract_action must return exactly 5 values."""
        result = BabyAILLMAgentsWrapper.extract_action("<action>go forward</action>")
        assert len(result) == 5

    def test_valid_action(self):
        """Valid action extraction."""
        full, extracted, valid, is_valid, metrics = BabyAILLMAgentsWrapper.extract_action(
            "<action>turn left</action>"
        )
        assert extracted == "turn left"
        assert valid == "turn left"
        assert is_valid is True
        assert metrics["behavior/valid_action_ratio"] == 1.0

    def test_invalid_action_falls_back_to_default(self):
        """Invalid action falls back to default (go forward)."""
        full, extracted, valid, is_valid, metrics = BabyAILLMAgentsWrapper.extract_action(
            "<action>jump</action>"
        )
        assert extracted == "jump"
        assert valid == "go forward"  # default
        assert is_valid is False
        assert metrics["behavior/valid_action_ratio"] == 0.0

    def test_no_tag_falls_back_to_default(self):
        """No tag falls back to default."""
        full, extracted, valid, is_valid, metrics = BabyAILLMAgentsWrapper.extract_action(
            "I will turn left"
        )
        assert extracted is None
        assert valid == "go forward"
        assert is_valid is False

    def test_metrics_include_valid_action_ratio(self):
        """Metrics must include behavior/valid_action_ratio."""
        _, _, _, _, metrics = BabyAILLMAgentsWrapper.extract_action("<action>toggle</action>")
        assert "behavior/valid_action_ratio" in metrics


class TestExtractActionInstance:
    """Tests for extract_action_instance method (instance method with mode switching)."""

    @pytest.fixture
    def mock_env(self):
        """Create a minimal mock environment that satisfies wrapper requirements."""
        import gymnasium as gym

        class MockBabyAIEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = gym.spaces.Discrete(6)
                self.observation_space = gym.spaces.Dict({})
                self.language_action_space = BABYAI_ACTION_SPACE[:]
                self.max_steps = 100
                self._mission = "test mission"

            def step(self, action):
                return {"text": {"long_term_context": "", "short_term_context": ""}, "mission": "test"}, 0, False, False, {}

            def reset(self, seed=None, options=None):
                return {"text": {"long_term_context": "", "short_term_context": ""}, "mission": "test"}, {}

            def get_stats(self):
                return {}

            def get_text_action(self, action):
                if isinstance(action, str):
                    return action
                return self.language_action_space[action]

        return MockBabyAIEnv()

    def test_multi_action_mode_extracts_decision(self, mock_env):
        """In multi-action mode, extracts from <decision> tag."""
        wrapper = BabyAILLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        text = "<action>drop</action><decision>turn left</decision>"
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "turn left"
        assert is_valid is True

    def test_single_action_mode_extracts_action(self, mock_env):
        """In single-action mode, extracts from <action> tag."""
        wrapper = BabyAILLMAgentsWrapper(mock_env, multi_action_reasoning=False)

        text = "<action>drop</action><decision>turn left</decision>"
        full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

        assert extracted == "drop"
        assert is_valid is True

    def test_all_valid_actions(self, mock_env):
        """Every valid action should return is_valid=True."""
        wrapper = BabyAILLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        valid_actions = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]
        for action in valid_actions:
            text = f"<decision>{action}</decision>"
            full, extracted, executed, is_valid, metrics = wrapper.extract_action_instance(text)

            assert extracted == action, f"Expected '{action}', got '{extracted}'"
            assert is_valid is True, f"Expected is_valid=True for '{action}'"
            assert executed == action

    def test_metrics_include_analysis(self, mock_env):
        """Instance method includes additional analysis metrics."""
        wrapper = BabyAILLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        text = "I could turn left but wait maybe go forward <decision>go forward</decision>"
        _, _, _, _, metrics = wrapper.extract_action_instance(text)

        assert "behavior/valid_action_ratio" in metrics
        assert "behavior/plan_length" in metrics
        assert "behavior/backtrack_length" in metrics
        # Don't assert specific values - heuristic word list may change


class TestLanguageActionSpace:
    """Tests for language_action_space property."""

    @pytest.fixture
    def mock_env(self):
        import gymnasium as gym

        class MockBabyAIEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.language_action_space = BABYAI_ACTION_SPACE[:]
                self.max_steps = 100

            def reset(self, **kwargs):
                return {"text": {"long_term_context": "", "short_term_context": ""}, "mission": "test"}, {}

            def step(self, action):
                return {"text": {"long_term_context": "", "short_term_context": ""}, "mission": "test"}, 0, False, False, {}

            def get_stats(self):
                return {}

            def get_text_action(self, action):
                return action if isinstance(action, str) else self.language_action_space[action]

        return MockBabyAIEnv()

    def test_is_list(self, mock_env):
        """language_action_space must be a list."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        assert isinstance(wrapper.language_action_space, list)

    def test_all_strings(self, mock_env):
        """All elements must be strings."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        assert all(isinstance(a, str) for a in wrapper.language_action_space)

    def test_random_choice_works(self, mock_env):
        """random.choice() must work on language_action_space."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        action = random.choice(wrapper.language_action_space)
        assert action in wrapper.language_action_space

    def test_in_membership_works(self, mock_env):
        """'in' membership check must work."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        assert "go forward" in wrapper.language_action_space
        assert "jump" not in wrapper.language_action_space


class TestInstructionPrompt:
    """Tests for get_instruction_prompt method."""

    @pytest.fixture
    def mock_env(self):
        import gymnasium as gym

        class MockBabyAIEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.language_action_space = BABYAI_ACTION_SPACE[:]
                self.max_steps = 100
                self._mission = "go to the red ball"

            def reset(self, **kwargs):
                return {"text": {"long_term_context": "", "short_term_context": ""}, "mission": "go to the red ball"}, {}

            def step(self, action):
                return {"text": {"long_term_context": "", "short_term_context": ""}, "mission": "go to the red ball"}, 0, False, False, {}

            def get_stats(self):
                return {}

            def get_text_action(self, action):
                return action if isinstance(action, str) else self.language_action_space[action]

        return MockBabyAIEnv()

    def test_with_explicit_mission(self, mock_env):
        """get_instruction_prompt with explicit mission kwarg."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        prompt = wrapper.get_instruction_prompt(mission="pick up the key")
        assert "pick up the key" in prompt

    def test_fetches_mission_from_env(self, mock_env):
        """get_instruction_prompt fetches mission from env if not provided."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        wrapper.reset()  # Populates _last_obs
        prompt = wrapper.get_instruction_prompt()
        assert "go to the red ball" in prompt

    def test_default_mission_fallback(self, mock_env):
        """Falls back to default mission if none available."""
        mock_env._mission = None
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        # Don't call reset - no _last_obs
        prompt = wrapper.get_instruction_prompt()
        assert "complete the task" in prompt

    def test_multi_action_prompt_different(self, mock_env):
        """Multi-action mode has different prompt format."""
        wrapper_std = BabyAILLMAgentsWrapper(mock_env, multi_action_reasoning=False)
        wrapper_multi = BabyAILLMAgentsWrapper(mock_env, multi_action_reasoning=True)

        prompt_std = wrapper_std.get_instruction_prompt(mission="test")
        prompt_multi = wrapper_multi.get_instruction_prompt(mission="test")

        assert "<decision>" in prompt_multi
        assert "<decision>" not in prompt_std

    def test_config_override(self, mock_env):
        """Config can override entire instruction prompt."""
        custom_prompt = "Custom prompt: {mission}"
        wrapper = BabyAILLMAgentsWrapper(mock_env, instruction_prompt=custom_prompt)
        prompt = wrapper.get_instruction_prompt(mission="test task")
        assert prompt == "Custom prompt: test task"


class TestRestructureObs:
    """Tests for restructure_obs validation."""

    @pytest.fixture
    def mock_env(self):
        import gymnasium as gym

        class MockBabyAIEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.language_action_space = BABYAI_ACTION_SPACE[:]
                self.max_steps = 100

            def reset(self, **kwargs):
                return {"text": {"long_term_context": "ctx", "short_term_context": ""}, "mission": "test"}, {}

            def step(self, action):
                return {"text": {"long_term_context": "ctx", "short_term_context": ""}, "mission": "test"}, 0, False, False, {}

            def get_stats(self):
                return {}

            def get_text_action(self, action):
                return action if isinstance(action, str) else self.language_action_space[action]

        return MockBabyAIEnv()

    def test_valid_obs_passes(self, mock_env):
        """Valid observation passes validation."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        obs = {"text": {"long_term_context": "ctx", "short_term_context": ""}, "mission": "test"}
        result = wrapper.restructure_obs(obs)
        assert result == obs

    def test_missing_text_raises(self, mock_env):
        """Missing 'text' key raises ValueError."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        with pytest.raises(ValueError, match="missing 'text' key"):
            wrapper.restructure_obs({"mission": "test"})

    def test_missing_context_keys_raises(self, mock_env):
        """Missing context keys raises ValueError."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        with pytest.raises(ValueError, match="missing required keys"):
            wrapper.restructure_obs({"text": {"long_term_context": "ctx"}})


class TestStepResetCycle:
    """Integration tests for step/reset cycle."""

    @pytest.fixture
    def mock_env(self):
        import gymnasium as gym

        class MockBabyAIEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.language_action_space = BABYAI_ACTION_SPACE[:]
                self.max_steps = 100
                self._mission = "test"

            def reset(self, **kwargs):
                return {"text": {"long_term_context": "You see a ball", "short_term_context": ""}, "mission": "get the ball"}, {}

            def step(self, action):
                return {"text": {"long_term_context": "You moved", "short_term_context": "step taken"}, "mission": "get the ball"}, 1.0, False, False, {}

            def get_stats(self):
                return {"mission": self._mission}

            def get_text_action(self, action):
                return action if isinstance(action, str) else self.language_action_space[action]

        return MockBabyAIEnv()

    def test_reset_returns_valid_obs(self, mock_env):
        """Reset returns observation with required keys."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        obs, info = wrapper.reset()

        assert "text" in obs
        assert "long_term_context" in obs["text"]
        assert "short_term_context" in obs["text"]

    def test_step_returns_float_reward(self, mock_env):
        """Step returns float reward."""
        wrapper = BabyAILLMAgentsWrapper(mock_env)
        wrapper.reset()
        action = random.choice(wrapper.language_action_space)
        obs, reward, term, trunc, info = wrapper.step(action, is_valid=True)

        assert isinstance(reward, float)

    def test_step_applies_format_penalty(self, mock_env):
        """Step applies format penalty when is_valid=False."""
        wrapper = BabyAILLMAgentsWrapper(mock_env, format_penalty=0.5)
        wrapper.reset()
        action = random.choice(wrapper.language_action_space)
        _, reward, _, _, _ = wrapper.step(action, is_valid=False)

        assert reward == -0.5

    def test_binary_reward(self, mock_env):
        """Binary reward mode converts positive rewards to 1.0."""
        wrapper = BabyAILLMAgentsWrapper(mock_env, binary_reward=True)
        wrapper.reset()
        action = random.choice(wrapper.language_action_space)
        _, reward, _, _, _ = wrapper.step(action, is_valid=True)

        assert reward == 1.0
