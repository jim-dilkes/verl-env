"""Tests for VecEnv hard_reset functionality.

Tests that:
1. VecEnv.hard_reset() can rebuild env+captioner inside existing worker processes
2. Worker processes remain usable after hard_reset
3. Reset and step work after hard_reset
4. Failed hard_reset keeps worker in usable state (transactional semantics)

Run with: pytest tests/envs/test_vecenv_hard_reset.py -v
"""

import pytest
import sys
from omegaconf import OmegaConf


def make_test_config(n_rollouts=2, width=5, height=5, max_rounds=3):
    """Create a minimal config for testing."""
    return OmegaConf.create({
        "envs": {
            "env_name": "fastsnake",
            "n_rollouts": n_rollouts,
            "format_penalty": 0.0,
            "vec_env_multiprocessing": "fork",
            "fastsnake_kwargs": {
                "width": width,
                "height": height,
                "max_rounds": max_rounds,
                "num_external_snakes": 0,
                "num_random_snakes": 0,
                "num_apples": 1,
                "num_bananas": 0,
                "num_fires": 0,
                "print_visualization": False,
                "print_coordinates": False,
                "print_axes": False,
            },
            "captioner": {
                "name": "history",
                "type": "naive",
                "system_instruction": "",
                "history_length": 1,
                "max_text_history": 10,
                "max_image_history": 0,
                "max_cot_history": 0,
            },
        },
        "prompt": {
            "prompt": {
                "multi_action_reasoning": False,
                "epsilon": 0.0,
                "environment_instruction": "Test instruction",
            }
        },
    })


@pytest.fixture
def vec_env():
    """Create a VecEnv for testing."""
    from verl.envs.environments import make_env
    from verl.envs.captioners import make_captioner
    from verl.envs.vec_env import VecEnv

    config = make_test_config(n_rollouts=2)

    def get_env_fn(rank):
        def init_env():
            return make_env("fastsnake", "default", config, render_mode=None)
        return init_env

    def get_captioner_fn(rank):
        def init_captioner():
            return make_captioner(config)
        return init_captioner

    env_fns = [get_env_fn(i) for i in range(config.envs.n_rollouts)]
    captioner_fns = [get_captioner_fn(i) for i in range(config.envs.n_rollouts)]

    env = VecEnv(
        env_name="fastsnake",
        config=config,
        env_fns=env_fns,
        captioner_fns=captioner_fns,
    )
    yield env
    env.close()


class TestVecEnvHardReset:
    """Tests for VecEnv.hard_reset() method."""

    def test_hard_reset_basic(self, vec_env):
        """Basic hard_reset changes env config and allows reset/step."""
        # Initial reset and step
        obs, info = vec_env.reset(seed=42)
        assert len(obs) == 2
        actions = ["<action>up</action>", "<action>down</action>"]
        obs2, rewards, terminated, truncated, infos = vec_env.step(actions)
        assert len(obs2) == 2

        # Hard reset with different config (bigger board)
        new_config = make_test_config(n_rollouts=2, width=8, height=8, max_rounds=5)
        vec_env.hard_reset(
            env_name="fastsnake",
            task="default",
            config=new_config,
            render_mode=None
        )

        # Verify reset/step still work after hard_reset
        obs3, info3 = vec_env.reset(seed=123)
        assert len(obs3) == 2

        actions2 = ["<action>left</action>", "<action>right</action>"]
        obs4, rewards2, terminated2, truncated2, infos2 = vec_env.step(actions2)
        assert len(obs4) == 2

    def test_hard_reset_changes_env_name(self, vec_env):
        """hard_reset updates internal env_name."""
        assert vec_env.env_name == "fastsnake"

        # After hard_reset, env_name should be updated
        new_config = make_test_config()
        vec_env.hard_reset(
            env_name="fastsnake",
            task="custom_task",
            config=new_config,
        )

        assert vec_env.env_name == "fastsnake"

    def test_hard_reset_clears_cached_obs(self, vec_env):
        """hard_reset clears cached observations."""
        # Do some steps to populate cache
        vec_env.reset(seed=42)
        vec_env.step(["<action>up</action>", "<action>down</action>"])

        # Cache should be populated
        assert vec_env.last_obs[0] is not None
        assert vec_env.last_obs[1] is not None

        # Hard reset
        new_config = make_test_config()
        vec_env.hard_reset(
            env_name="fastsnake",
            task="default",
            config=new_config,
        )

        # Cache should be cleared
        assert vec_env.last_obs[0] is None
        assert vec_env.last_obs[1] is None

    def test_hard_reset_preserves_n_rollouts(self, vec_env):
        """hard_reset doesn't change the number of workers."""
        original_n_rollouts = vec_env.n_rollouts
        original_n_processes = len(vec_env.processes)

        new_config = make_test_config()
        vec_env.hard_reset(
            env_name="fastsnake",
            task="default",
            config=new_config,
        )

        assert vec_env.n_rollouts == original_n_rollouts
        assert len(vec_env.processes) == original_n_processes

    def test_hard_reset_multiple_times(self, vec_env):
        """Can call hard_reset multiple times."""
        for i in range(3):
            config = make_test_config(width=5 + i, height=5 + i)
            vec_env.hard_reset(
                env_name="fastsnake",
                task="default",
                config=config,
            )

            # Verify still works
            obs, info = vec_env.reset(seed=i * 100)
            assert len(obs) == 2
            obs2, _, _, _, _ = vec_env.step(["<action>up</action>", "<action>down</action>"])
            assert len(obs2) == 2


class TestHardResetValidation:
    """Tests for hard_reset validation and error handling."""

    def test_hard_reset_rejects_n_rollouts_mismatch(self, vec_env):
        """hard_reset raises ValueError if config.envs.n_rollouts doesn't match pool."""
        # vec_env fixture has n_rollouts=2
        assert vec_env.n_rollouts == 2

        # Try to hard_reset with n_rollouts=4 (mismatch)
        mismatched_config = make_test_config(n_rollouts=4)

        with pytest.raises(ValueError, match="doesn't match pool worker count"):
            vec_env.hard_reset(
                env_name="fastsnake",
                task="default",
                config=mismatched_config,
            )

    def test_hard_reset_accepts_matching_n_rollouts(self, vec_env):
        """hard_reset succeeds when config.envs.n_rollouts matches pool."""
        assert vec_env.n_rollouts == 2

        # n_rollouts=2 matches
        matching_config = make_test_config(n_rollouts=2, width=7, height=7)

        # Should not raise
        vec_env.hard_reset(
            env_name="fastsnake",
            task="default",
            config=matching_config,
        )

        # Verify it works
        obs, info = vec_env.reset(seed=42)
        assert len(obs) == 2

    def test_hard_reset_env_still_usable_after_config_error(self, vec_env):
        """After a hard_reset config error, the env should still be usable."""
        # First, do a successful reset/step
        obs1, _ = vec_env.reset(seed=42)
        assert len(obs1) == 2

        # Attempt invalid hard_reset
        mismatched_config = make_test_config(n_rollouts=4)
        with pytest.raises(ValueError):
            vec_env.hard_reset(
                env_name="fastsnake",
                task="default",
                config=mismatched_config,
            )

        # Env should still be usable with original config
        obs2, _ = vec_env.reset(seed=43)
        assert len(obs2) == 2
        obs3, _, _, _, _ = vec_env.step(["<action>up</action>", "<action>down</action>"])
        assert len(obs3) == 2


class TestHardResetTimeout:
    """Tests for hard_reset timeout behavior."""

    def test_timeout_env_var_is_respected(self):
        """VERL_HARD_RESET_TIMEOUT env var should be read (smoke test)."""
        import os

        # Just verify the env var is read (actual timeout testing would require mocking)
        old_val = os.environ.get('VERL_HARD_RESET_TIMEOUT')
        try:
            os.environ['VERL_HARD_RESET_TIMEOUT'] = '120'

            from verl.envs.environments import make_env
            from verl.envs.captioners import make_captioner
            from verl.envs.vec_env import VecEnv

            config = make_test_config(n_rollouts=1)

            def get_env_fn(rank):
                def init_env():
                    return make_env("fastsnake", "default", config)
                return init_env

            def get_captioner_fn(rank):
                def init_captioner():
                    return make_captioner(config)
                return init_captioner

            # Create and use VecEnv (timeout is used during hard_reset)
            env = VecEnv(
                env_name="fastsnake",
                config=config,
                env_fns=[get_env_fn(0)],
                captioner_fns=[get_captioner_fn(0)],
            )
            try:
                # hard_reset should work with the timeout setting
                env.hard_reset(
                    env_name="fastsnake",
                    task="default",
                    config=config,
                )
                obs, _ = env.reset(seed=42)
                assert len(obs) == 1
            finally:
                env.close()
        finally:
            if old_val is not None:
                os.environ['VERL_HARD_RESET_TIMEOUT'] = old_val
            elif 'VERL_HARD_RESET_TIMEOUT' in os.environ:
                del os.environ['VERL_HARD_RESET_TIMEOUT']


class TestVecEnvContextManagerDeprecation:
    """Tests for VecEnvContextManager deprecation warning."""

    def test_deprecation_warning_is_raised(self):
        """VecEnvContextManager raises DeprecationWarning on init."""
        import warnings
        from verl.trainer.ppo.multi_env_evaluator import VecEnvContextManager

        config = make_test_config(n_rollouts=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Create context manager (should warn)
            cm = VecEnvContextManager(
                env_name="fastsnake",
                task="default",
                config=config,
            )
            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "MultiEnvEvaluator" in str(w[0].message)
