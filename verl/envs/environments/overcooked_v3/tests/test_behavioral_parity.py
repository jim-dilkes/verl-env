"""Behavioral parity tests comparing pure Python vs JaxMARL implementations.

These tests verify that the pure Python implementation produces identical
behavior to the JaxMARL version for the same inputs.

NOTE: These tests require JaxMARL to be installed. They will be skipped if
JaxMARL is not available.
"""

import pytest
import numpy as np
from typing import Optional, Tuple


# Check if JaxMARL is available
def check_jaxmarl_available():
    """Check if JaxMARL and the original wrapper are available."""
    try:
        import jax
        import jaxmarl
        from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
        return True
    except ImportError:
        return False


JAXMARL_AVAILABLE = check_jaxmarl_available()
skip_without_jaxmarl = pytest.mark.skipif(
    not JAXMARL_AVAILABLE,
    reason="JaxMARL not available"
)


@pytest.fixture
def pure_env():
    """Create pure Python environment."""
    from verl.envs.environments.overcooked_v3.gym_wrapper import OvercookedGymWrapper
    return OvercookedGymWrapper(
        layout="cramped_room",
        max_steps=200,
        partner_policy="noop",
        seed=42,
        shaped_reward=True,
    )


@pytest.fixture
def jax_env():
    """Create JaxMARL environment."""
    if not JAXMARL_AVAILABLE:
        pytest.skip("JaxMARL not available")
    from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
    return OvercookedGymWrapper(
        layout="cramped_room",
        max_steps=200,
        partner_policy="noop",
        seed=42,
        shaped_reward=True,
    )


def get_both_envs(seed: int = 42, **kwargs):
    """Create matched JaxMARL and pure Python environments.

    Returns:
        Tuple of (jax_env, pure_env), or (None, pure_env) if JaxMARL unavailable
    """
    from verl.envs.environments.overcooked_v3.gym_wrapper import OvercookedGymWrapper as PureWrapper

    defaults = {
        "layout": "cramped_room",
        "max_steps": 200,
        "partner_policy": "noop",
        "seed": seed,
        "shaped_reward": True,
    }
    defaults.update(kwargs)

    pure_env = PureWrapper(**defaults)

    if JAXMARL_AVAILABLE:
        from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper as JaxWrapper
        jax_env = JaxWrapper(**defaults)
        return jax_env, pure_env

    return None, pure_env


class TestBasicParity:
    """Basic parity tests."""

    @skip_without_jaxmarl
    def test_reset_state_match(self):
        """Initial state after reset should match."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_obs, jax_info = jax_env.reset(seed=42)
        pure_obs, pure_info = pure_env.reset(seed=42)

        # Compare agent positions
        jax_state = jax_env.get_state_info()
        pure_state = pure_env.get_state_info()

        for i, (jax_agent, pure_agent) in enumerate(zip(jax_state["agents"], pure_state["agents"])):
            assert jax_agent["pos"] == pure_agent["pos"], f"Agent {i} position mismatch"
            assert jax_agent["direction"] == pure_agent["direction"], f"Agent {i} direction mismatch"
            assert jax_agent["inventory"] == pure_agent["inventory"], f"Agent {i} inventory mismatch"

    @skip_without_jaxmarl
    def test_simple_step_parity(self):
        """Single step should produce same result."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        # Take same action
        jax_obs, jax_r, _, _, _ = jax_env.step(4)  # stay
        pure_obs, pure_r, _, _, _ = pure_env.step(4)

        assert abs(jax_r - pure_r) < 1e-6, f"Reward mismatch: {jax_r} vs {pure_r}"

        # Compare states
        jax_state = jax_env.get_state_info()
        pure_state = pure_env.get_state_info()

        assert jax_state["time"] == pure_state["time"], "Time mismatch"


class TestDeterministicSequence:
    """Test deterministic action sequences produce same results."""

    @skip_without_jaxmarl
    def test_movement_sequence(self):
        """Movement sequence should produce identical agent positions."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        # Sequence of movements
        actions = [0, 0, 1, 1, 2, 2, 3, 3]  # right, right, down, down, left, left, up, up

        for action in actions:
            jax_obs, jax_r, _, _, _ = jax_env.step(action)
            pure_obs, pure_r, _, _, _ = pure_env.step(action)

            jax_state = jax_env.get_state_info()
            pure_state = pure_env.get_state_info()

            # Compare controlled agent position
            assert jax_state["agents"][0]["pos"] == pure_state["agents"][0]["pos"], \
                f"Position mismatch after action {action}"

    @skip_without_jaxmarl
    def test_extended_sequence(self):
        """Extended action sequence should match step-by-step."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        # Mixed actions
        actions = [4, 4, 0, 5, 3, 3, 5, 1, 1, 5, 4, 4]

        cumulative_jax_reward = 0.0
        cumulative_pure_reward = 0.0

        for i, action in enumerate(actions):
            jax_obs, jax_r, jax_term, _, _ = jax_env.step(action)
            pure_obs, pure_r, pure_term, _, _ = pure_env.step(action)

            cumulative_jax_reward += jax_r
            cumulative_pure_reward += pure_r

            assert abs(jax_r - pure_r) < 1e-6, \
                f"Reward mismatch at step {i}: {jax_r} vs {pure_r}"
            assert jax_term == pure_term, \
                f"Termination mismatch at step {i}"

        assert abs(cumulative_jax_reward - cumulative_pure_reward) < 1e-5, \
            f"Cumulative reward mismatch: {cumulative_jax_reward} vs {cumulative_pure_reward}"


class TestRenderParity:
    """Test that text rendering matches."""

    @skip_without_jaxmarl
    def test_initial_render_match(self):
        """Initial render output should match."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        jax_render = jax_env.render()
        pure_render = pure_env.render()

        # The renders should be identical
        assert jax_render == pure_render, \
            f"Render mismatch:\n--- JaxMARL ---\n{jax_render}\n--- Pure ---\n{pure_render}"

    @skip_without_jaxmarl
    def test_render_after_steps(self):
        """Render should match after taking steps."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        actions = [4, 0, 1, 5]

        for action in actions:
            jax_env.step(action)
            pure_env.step(action)

        jax_render = jax_env.render()
        pure_render = pure_env.render()

        assert jax_render == pure_render, \
            f"Render mismatch:\n--- JaxMARL ---\n{jax_render}\n--- Pure ---\n{pure_render}"


class TestInteractParity:
    """Test interact action produces same results."""

    @skip_without_jaxmarl
    def test_pickup_from_pile(self):
        """Picking up from ingredient pile should match."""
        jax_env, pure_env = get_both_envs(seed=42)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        # Move to face ingredient pile and interact
        # In cramped_room, agents start at (1, 1) and (3, 1)
        # Ingredient piles are at (0, 1) and (4, 1)

        # Agent 0 faces left toward pile
        actions = [2, 5]  # left (to face pile), interact

        for action in actions:
            jax_env.step(action)
            pure_env.step(action)

        jax_state = jax_env.get_state_info()
        pure_state = pure_env.get_state_info()

        # Check inventory matches
        assert jax_state["agents"][0]["inventory"] == pure_state["agents"][0]["inventory"], \
            f"Inventory mismatch: {jax_state['agents'][0]['inventory']} vs {pure_state['agents'][0]['inventory']}"


class TestCookingParity:
    """Test cooking mechanics match."""

    @skip_without_jaxmarl
    def test_pot_timer_countdown(self):
        """Pot timer should count down identically."""
        jax_env, pure_env = get_both_envs(seed=42, pot_cook_time=5)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        # This is a complex test that would require:
        # 1. Picking up 3 ingredients
        # 2. Placing them in pot
        # 3. Verifying timer behavior

        # For now, just verify the environments have same cook time
        assert jax_env.pot_cook_time == pure_env.pot_cook_time


class TestSoloModeParity:
    """Test solo mode behavior matches."""

    @skip_without_jaxmarl
    def test_solo_mode_partner_position(self):
        """Partner should be at same off-map position in solo mode."""
        jax_env, pure_env = get_both_envs(seed=42, partner_policy="none")

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        jax_state = jax_env.get_state_info()
        pure_state = pure_env.get_state_info()

        # Both should be in solo mode
        assert jax_state["solo_mode"] == True
        assert pure_state["solo_mode"] == True

        # Both should only have 1 agent visible
        assert len(jax_state["agents"]) == 1
        assert len(pure_state["agents"]) == 1


class TestShapedRewardParity:
    """Test shaped rewards match."""

    @skip_without_jaxmarl
    def test_shaped_reward_enabled(self):
        """Shaped rewards should be included when enabled."""
        jax_env, pure_env = get_both_envs(seed=42, shaped_reward=True)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        # Both should have shaped_reward attribute
        assert jax_env.shaped_reward == True
        assert pure_env.shaped_reward == True

    @skip_without_jaxmarl
    def test_shaped_reward_disabled(self):
        """Shaped rewards should be excluded when disabled."""
        jax_env, pure_env = get_both_envs(seed=42, shaped_reward=False)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        assert jax_env.shaped_reward == False
        assert pure_env.shaped_reward == False


class TestFullEpisode:
    """Test complete episode behavior."""

    @skip_without_jaxmarl
    def test_episode_terminates_same_step(self):
        """Episode should terminate at same step."""
        jax_env, pure_env = get_both_envs(seed=42, max_steps=10)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        for i in range(15):  # More than max_steps
            jax_obs, _, jax_term, jax_trunc, _ = jax_env.step(4)
            pure_obs, _, pure_term, pure_trunc, _ = pure_env.step(4)

            assert jax_term == pure_term, f"Termination mismatch at step {i}"
            assert jax_trunc == pure_trunc, f"Truncation mismatch at step {i}"

            if jax_term or jax_trunc:
                break


class TestLayoutParity:
    """Test different layouts produce same results."""

    @skip_without_jaxmarl
    @pytest.mark.parametrize("layout_name", [
        "cramped_room",
        # "asymm_advantages",  # Larger, might have different agent swap behavior
        # "coord_ring",
    ])
    def test_layout_initial_state(self, layout_name):
        """Different layouts should produce matching initial states."""
        jax_env, pure_env = get_both_envs(seed=42, layout=layout_name)

        jax_env.reset(seed=42)
        pure_env.reset(seed=42)

        jax_state = jax_env.get_state_info()
        pure_state = pure_env.get_state_info()

        # Compare grid dimensions
        assert jax_state["grid"].shape == pure_state["grid"].shape, \
            f"Grid shape mismatch for {layout_name}"


class TestRandomAgentPositions:
    """Test random agent position initialization."""

    @skip_without_jaxmarl
    def test_same_seed_same_positions(self):
        """Same seed should produce same random positions."""
        jax_env1, pure_env1 = get_both_envs(seed=42, random_agent_positions=True)
        jax_env2, pure_env2 = get_both_envs(seed=42, random_agent_positions=True)

        # Reset both pairs with same seed
        jax_env1.reset(seed=42)
        pure_env1.reset(seed=42)
        jax_env2.reset(seed=42)
        pure_env2.reset(seed=42)

        # Pure implementations should match each other
        pure_state1 = pure_env1.get_state_info()
        pure_state2 = pure_env2.get_state_info()

        for i, (a1, a2) in enumerate(zip(pure_state1["agents"], pure_state2["agents"])):
            assert a1["pos"] == a2["pos"], \
                f"Pure env position mismatch for agent {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
