import pytest

import gymnasium as gym

from verl.envs.environments.env_wrapper import EnvWrapper


class _BaseDummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Dict({})

    def step(self, action):
        return {}, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return {}, {}


class _DummySingleActionEnv(_BaseDummyEnv):
    """Minimal env stub used to test strictness behavior.

    We intentionally provide only `extract_action` (single-action parser), while
    advertising `multi_action_reasoning=True`.
    """

    multi_action_reasoning = True

    def extract_action(self, action):
        full_action = str(action)
        return full_action, "__invalid__", "up", False, {"behavior/valid_action_ratio": 0.0}


class _DummySingleActionEnvNoMulti(_BaseDummyEnv):
    multi_action_reasoning = False

    def extract_action(self, action):
        full_action = str(action)
        return full_action, "up", "up", True, {"behavior/valid_action_ratio": 1.0}


def test_envwrapper_refuses_fallback_in_multi_action_mode():
    env = EnvWrapper(_DummySingleActionEnv(), env_name="fastsnake", task_name="default")

    with pytest.raises(RuntimeError, match=r"multi_action_reasoning=True.*no extract_action_instance"):
        env.extract_action_instance("<decision>up</decision>")


def test_envwrapper_allows_fallback_in_single_action_mode():
    env = EnvWrapper(_DummySingleActionEnvNoMulti(), env_name="fastsnake", task_name="default")

    full, extracted, executed, is_valid, metrics = env.extract_action_instance("<action>up</action>")
    assert extracted == "up"
    assert is_valid is True
    assert executed == "up"
    assert metrics["behavior/valid_action_ratio"] == 1.0
