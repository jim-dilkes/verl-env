from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
from verl.envs.environments.overcooked.base import OvercookedLLMAgentsWrapper


def make_overcooked_env(env_name, task, config, render_mode=None):
    """Create an Overcooked environment with LLM agent wrapper."""

    overcooked_kwargs = dict(config.envs.overcooked_kwargs) if hasattr(config.envs, "overcooked_kwargs") else {}

    layout = overcooked_kwargs.get("layout_name", "cramped_room")
    max_steps = overcooked_kwargs.get("horizon", 200)
    partner_policy = overcooked_kwargs.get("partner_policy", "noop")
    shaped_reward = overcooked_kwargs.get("shaped_reward", True)
    seed = overcooked_kwargs.get("seed", 0)

    base_env = OvercookedGymWrapper(
        layout=layout,
        max_steps=max_steps,
        partner_policy=partner_policy,
        shaped_reward=shaped_reward,
        seed=seed,
    )

    env_kwargs = dict(config.envs) if hasattr(config, "envs") else {}

    if hasattr(config, "prompt") and hasattr(config.prompt, "prompt"):
        environment_instruction = getattr(config.prompt.prompt, "environment_instruction", None)
        if environment_instruction is not None:
            env_kwargs["instruction_prompt"] = environment_instruction

    env = OvercookedLLMAgentsWrapper(base_env, **env_kwargs)

    return env
