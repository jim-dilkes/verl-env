"""Factory function for creating Overcooked environments.

This module provides the same interface as the JaxMARL-based overcooked_env.py
but uses the pure Python implementation.
"""

from .gym_wrapper import OvercookedGymWrapper
from .base import OvercookedLLMAgentsWrapper
from .layouts import BUILTIN_LAYOUTS, CUSTOM_LAYOUTS


def make_overcooked_env(env_name, task, config, render_mode=None):
    """Create an Overcooked environment with LLM agent wrapper.

    This function provides the same interface as the JaxMARL-based version.

    Config options (via envs.overcooked_kwargs):
        layout_name: str - Kitchen layout (default: "cramped_room")
            Built-in: cramped_room, asymm_advantages, coord_ring,
                      forced_coord, counter_circuit
            Custom layouts: cramped_room_mixed (2 onion + 1 tomato recipe)
        horizon: int - Max steps per episode (default: 200)
        partner_policy: str - Partner agent behavior (default: "noop")
            Options: "noop" (stays), "random", "none" (solo mode, partner hidden)
        shaped_reward: bool - Include shaping rewards (default: True)
        seed: int - Random seed (default: 0)
        print_visualization: bool - Show ASCII grid (default: True)
        print_coordinates: bool - Show coordinate text (default: True)
        pot_cook_time: int - Cooking duration in ticks (default: 20)
        random_agent_positions: bool - Randomize agent spawn (default: False)

    Note: Like JaxMARL, this enforces 3 ingredients per recipe.

    Args:
        env_name: Environment name (unused, for compatibility)
        task: Task name (unused, for compatibility)
        config: Configuration object with envs.overcooked_kwargs
        render_mode: Render mode (unused, always "ansi")

    Returns:
        OvercookedLLMAgentsWrapper instance
    """
    # Extract kwargs from config
    overcooked_kwargs = {}
    if hasattr(config, 'envs') and hasattr(config.envs, 'overcooked_kwargs'):
        overcooked_kwargs = dict(config.envs.overcooked_kwargs)

    layout_name = overcooked_kwargs.get("layout_name", "cramped_room")
    max_steps = overcooked_kwargs.get("horizon", 200)
    partner_policy = overcooked_kwargs.get("partner_policy", "noop")
    shaped_reward = overcooked_kwargs.get("shaped_reward", True)
    seed = overcooked_kwargs.get("seed", 0)
    print_visualization = overcooked_kwargs.get("print_visualization", True)
    print_coordinates = overcooked_kwargs.get("print_coordinates", True)
    pot_cook_time = overcooked_kwargs.get("pot_cook_time", None)
    random_agent_positions = overcooked_kwargs.get("random_agent_positions", False)

    # Create base environment
    base_env = OvercookedGymWrapper(
        layout=layout_name,
        max_steps=max_steps,
        partner_policy=partner_policy,
        shaped_reward=shaped_reward,
        seed=seed,
        print_visualization=print_visualization,
        print_coordinates=print_coordinates,
        pot_cook_time=pot_cook_time,
        random_agent_positions=random_agent_positions,
    )

    # Extract env kwargs for wrapper
    env_kwargs = dict(config.envs) if hasattr(config, 'envs') else {}

    # Check for multi-action reasoning mode
    multi_action_reasoning = False
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
        multi_action_reasoning = getattr(config.prompt.prompt, 'multi_action_reasoning', False)

    env_kwargs["multi_action_reasoning"] = multi_action_reasoning

    # Config instruction always overrides wrapper default when provided
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
        environment_instruction = getattr(config.prompt.prompt, 'environment_instruction', None)
        if environment_instruction is not None:
            env_kwargs["instruction_prompt"] = environment_instruction

    # Create LLM wrapper
    env = OvercookedLLMAgentsWrapper(base_env, **env_kwargs)

    return env
