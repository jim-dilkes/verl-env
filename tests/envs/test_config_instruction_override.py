"""Tests for config instruction override behavior in env wrappers.

Verifies that:
1. Config environment_instruction always overrides wrapper default when provided
2. When config doesn't specify instruction, wrapper generates appropriate default
3. Multi-action reasoning mode uses <decision> tag for action extraction
4. Epsilon-greedy exploration works in both envs
"""

import pytest
from types import SimpleNamespace


class DictableNamespace(SimpleNamespace):
    """SimpleNamespace that supports dict() conversion."""

    def __iter__(self):
        return iter(vars(self))

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def items(self):
        return vars(self).items()

    def __getitem__(self, key):
        return getattr(self, key)


class TestFastSnakeConfigOverride:
    """Test FastSnake environment instruction override logic."""

    @pytest.fixture
    def make_env_with_config(self):
        """Factory to create FastSnake env with given config."""
        from verl.envs.environments.FastSnake.fastsnake_env import make_fastsnake_env

        def _make(multi_action=False, epsilon=0.0, env_instruction=None):
            # Build config namespace mimicking hydra config
            prompt_config = SimpleNamespace(
                multi_action_reasoning=multi_action,
                epsilon=epsilon,
            )
            if env_instruction is not None:
                prompt_config.environment_instruction = env_instruction

            # FastSnakeEnv uses width/height, not grid_size
            config = DictableNamespace(
                envs=DictableNamespace(
                    fastsnake_kwargs={"width": 10, "height": 10, "num_external_snakes": 0}
                ),
                prompt=SimpleNamespace(prompt=prompt_config),
            )
            return make_fastsnake_env("fastsnake", None, config)

        return _make

    def test_config_instruction_overrides_in_standard_mode(self, make_env_with_config):
        """Config instruction should override when multi_action_reasoning=False."""
        custom_instruction = "CUSTOM INSTRUCTION FOR TESTING"
        env = make_env_with_config(
            multi_action=False, env_instruction=custom_instruction
        )
        assert env.get_instruction_prompt() == custom_instruction

    def test_config_instruction_overrides_in_multi_action_mode(self, make_env_with_config):
        """Config instruction should override EVEN when multi_action_reasoning=True."""
        custom_instruction = "CUSTOM MULTI-ACTION INSTRUCTION"
        env = make_env_with_config(
            multi_action=True, env_instruction=custom_instruction
        )
        assert env.get_instruction_prompt() == custom_instruction

    def test_wrapper_default_in_standard_mode(self, make_env_with_config):
        """Without config instruction, wrapper should use standard default."""
        env = make_env_with_config(multi_action=False, env_instruction=None)
        prompt = env.get_instruction_prompt()
        # Standard mode uses <action> tag
        assert "<action>" in prompt or "action" in prompt.lower()
        assert "<decision>" not in prompt

    def test_wrapper_default_in_multi_action_mode(self, make_env_with_config):
        """Without config instruction, wrapper should use multi-action default."""
        env = make_env_with_config(multi_action=True, env_instruction=None)
        prompt = env.get_instruction_prompt()
        # Multi-action mode uses <decision> tag
        assert "<decision>" in prompt

    def test_decision_extraction_in_multi_action_mode(self, make_env_with_config):
        """Multi-action mode should parse <decision> tags."""
        env = make_env_with_config(multi_action=True)

        llm_output = """<actions>
<action name="up">Going up seems safe</action>
<action name="down">Down leads to wall</action>
<action name="left">Left has apple</action>
<action name="right">Right is blocked</action>
</actions>
<decision>left</decision>"""

        _, extracted, valid, is_valid, _ = env.extract_action_instance(llm_output)
        assert extracted == "left"
        assert valid == "left"
        assert is_valid

    def test_action_extraction_in_standard_mode(self, make_env_with_config):
        """Standard mode should parse <action> tags."""
        env = make_env_with_config(multi_action=False)

        llm_output = "<plan>Going for apple</plan><action>right</action>"

        _, extracted, valid, is_valid, _ = env.extract_action_instance(llm_output)
        assert extracted == "right"
        assert valid == "right"
        assert is_valid


class TestOvercookedConfigOverride:
    """Test Overcooked environment instruction override logic."""

    @pytest.fixture
    def make_env_with_config(self):
        """Factory to create Overcooked env with given config."""
        from verl.envs.environments.overcooked.overcooked_env import make_overcooked_env

        def _make(multi_action=False, epsilon=0.0, env_instruction=None):
            prompt_config = SimpleNamespace(
                multi_action_reasoning=multi_action,
                epsilon=epsilon,
            )
            if env_instruction is not None:
                prompt_config.environment_instruction = env_instruction

            config = DictableNamespace(
                envs=DictableNamespace(
                    overcooked_kwargs={"layout_name": "cramped_room", "horizon": 50}
                ),
                prompt=SimpleNamespace(prompt=prompt_config),
            )
            return make_overcooked_env("overcooked", None, config)

        return _make

    def test_config_instruction_overrides_in_standard_mode(self, make_env_with_config):
        """Config instruction should override when multi_action_reasoning=False."""
        custom_instruction = "CUSTOM OVERCOOKED INSTRUCTION"
        env = make_env_with_config(
            multi_action=False, env_instruction=custom_instruction
        )
        assert env.get_instruction_prompt() == custom_instruction

    def test_config_instruction_overrides_in_multi_action_mode(self, make_env_with_config):
        """Config instruction should override EVEN when multi_action_reasoning=True."""
        custom_instruction = "CUSTOM MULTI-ACTION OVERCOOKED"
        env = make_env_with_config(
            multi_action=True, env_instruction=custom_instruction
        )
        assert env.get_instruction_prompt() == custom_instruction

    def test_wrapper_default_in_standard_mode(self, make_env_with_config):
        """Without config instruction, wrapper should use standard default."""
        env = make_env_with_config(multi_action=False, env_instruction=None)
        prompt = env.get_instruction_prompt()
        assert "<decision>" not in prompt
        assert "Overcooked" in prompt or "cooking" in prompt.lower()

    def test_wrapper_default_in_multi_action_mode(self, make_env_with_config):
        """Without config instruction, wrapper should use multi-action default."""
        env = make_env_with_config(multi_action=True, env_instruction=None)
        prompt = env.get_instruction_prompt()
        assert "<decision>" in prompt

    def test_decision_extraction_in_multi_action_mode(self, make_env_with_config):
        """Multi-action mode should parse <decision> tags."""
        env = make_env_with_config(multi_action=True)

        llm_output = """<actions>
<action name="right">Moving right to get onion</action>
<action name="down">Down is wall</action>
<action name="left">Left is clear</action>
<action name="up">Up towards pot</action>
<action name="stay">No need to wait</action>
<action name="interact">Can pick up onion</action>
</actions>
<decision>interact</decision>"""

        _, extracted, valid, is_valid, _ = env.extract_action_instance(llm_output)
        assert extracted == "interact"
        assert valid == "interact"
        assert is_valid

    def test_action_extraction_in_standard_mode(self, make_env_with_config):
        """Standard mode should parse <action> tags."""
        env = make_env_with_config(multi_action=False)

        llm_output = "<plan>Pick up onion</plan><action>interact</action>"

        _, extracted, valid, is_valid, _ = env.extract_action_instance(llm_output)
        assert extracted == "interact"
        assert valid == "interact"
        assert is_valid


class TestEpsilonGreedy:
    """Test epsilon-greedy exploration in both envs."""

    def test_fastsnake_epsilon_zero_no_exploration(self):
        """With epsilon=0, should never explore."""
        from verl.envs.environments.FastSnake.fastsnake_env import make_fastsnake_env

        config = DictableNamespace(
            envs=DictableNamespace(
                fastsnake_kwargs={"width": 10, "height": 10, "num_external_snakes": 0}
            ),
            prompt=SimpleNamespace(
                prompt=SimpleNamespace(multi_action_reasoning=False, epsilon=0.0)
            ),
        )
        env = make_fastsnake_env("fastsnake", None, config)

        # Run many times, should never explore
        llm_output = "<action>up</action>"
        for _ in range(100):
            _, _, valid, _, metrics = env.extract_action_instance(llm_output)
            assert valid == "up"
            assert metrics["behavior/epsilon_explored"] == 0.0

    def test_overcooked_epsilon_zero_no_exploration(self):
        """With epsilon=0, should never explore."""
        from verl.envs.environments.overcooked.overcooked_env import make_overcooked_env

        config = DictableNamespace(
            envs=DictableNamespace(
                overcooked_kwargs={"layout_name": "cramped_room", "horizon": 50}
            ),
            prompt=SimpleNamespace(
                prompt=SimpleNamespace(multi_action_reasoning=False, epsilon=0.0)
            ),
        )
        env = make_overcooked_env("overcooked", None, config)

        llm_output = "<action>interact</action>"
        for _ in range(100):
            _, _, valid, _, metrics = env.extract_action_instance(llm_output)
            assert valid == "interact"
            assert metrics["behavior/epsilon_explored"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
