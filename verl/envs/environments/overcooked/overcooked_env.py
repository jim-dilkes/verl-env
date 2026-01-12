from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
from verl.envs.environments.overcooked.base import OvercookedLLMAgentsWrapper


def make_overcooked_env(env_name, task, config, render_mode=None):
    """Create an Overcooked environment with LLM agent wrapper.

    Config options (via envs.overcooked_kwargs):
        layout_name: str - Kitchen layout (default: "cramped_room")
            Options: cramped_room, asymm_advantages, coord_ring,
                    forced_coord, counter_circuit
        horizon: int - Max steps per episode (default: 200)
        partner_policy: str - Partner agent behavior (default: "noop")
            Options: "noop" (stays), "random", "none" (solo mode, partner hidden)
        shaped_reward: bool - Include shaping rewards (default: True)
        seed: int - Random seed (default: 0)
        print_visualization: bool - Show ASCII grid (default: True)
        print_coordinates: bool - Show coordinate text (default: True)
        pot_cook_time: int - Cooking duration in ticks (default: 20)

    Note: JaxMARL enforces 3 ingredients per recipe. This cannot be changed.
    """
    overcooked_kwargs = dict(config.envs.overcooked_kwargs) if hasattr(config.envs, "overcooked_kwargs") else {}

    layout = overcooked_kwargs.get("layout_name", "cramped_room")
    max_steps = overcooked_kwargs.get("horizon", 200)
    partner_policy = overcooked_kwargs.get("partner_policy", "noop")
    shaped_reward = overcooked_kwargs.get("shaped_reward", True)
    seed = overcooked_kwargs.get("seed", 0)
    print_visualization = overcooked_kwargs.get("print_visualization", True)
    print_coordinates = overcooked_kwargs.get("print_coordinates", True)
    pot_cook_time = overcooked_kwargs.get("pot_cook_time", None)

    base_env = OvercookedGymWrapper(
        layout=layout,
        max_steps=max_steps,
        partner_policy=partner_policy,
        shaped_reward=shaped_reward,
        seed=seed,
        print_visualization=print_visualization,
        print_coordinates=print_coordinates,
        pot_cook_time=pot_cook_time,
    )

    env_kwargs = dict(config.envs) if hasattr(config, "envs") else {}

    # Check for multi-action reasoning mode and epsilon
    multi_action_reasoning = False
    epsilon = 0.0
    if hasattr(config, "prompt") and hasattr(config.prompt, "prompt"):
        multi_action_reasoning = getattr(config.prompt.prompt, "multi_action_reasoning", False)
        epsilon = getattr(config.prompt.prompt, "epsilon", 0.0)

    env_kwargs["multi_action_reasoning"] = multi_action_reasoning
    env_kwargs["epsilon"] = epsilon

    # Config instruction always overrides wrapper default when provided
    if hasattr(config, "prompt") and hasattr(config.prompt, "prompt"):
        environment_instruction = getattr(config.prompt.prompt, "environment_instruction", None)
        if environment_instruction is not None:
            env_kwargs["instruction_prompt"] = environment_instruction

    env = OvercookedLLMAgentsWrapper(base_env, **env_kwargs)

    return env
