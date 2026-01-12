from verl.envs.environments.FastSnake.src.env import FastSnakeEnv
from verl.envs.environments.FastSnake.base import FastSnakeLLMAgentsWrapper


def make_fastsnake_env(env_name, task, config, render_mode=None):
    """Create a FastSnake environment with LLM agent wrapper.

    Args:
        env_name: Name of the environment ('fastsnake')
        task: Task name (unused for FastSnake)
        config: Configuration object with env settings
        render_mode: Optional render mode

    Returns:
        FastSnakeLLMAgentsWrapper wrapping FastSnakeEnv
    """
    fastsnake_kwargs = dict(config.envs.fastsnake_kwargs)

    env = FastSnakeEnv(**fastsnake_kwargs)

    # Prepare kwargs for the wrapper
    env_kwargs = dict(config.envs)

    # Check for multi-action reasoning mode and epsilon first
    multi_action_reasoning = False
    epsilon = 0.0
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
        multi_action_reasoning = getattr(config.prompt.prompt, 'multi_action_reasoning', False)
        epsilon = getattr(config.prompt.prompt, 'epsilon', 0.0)

    env_kwargs['multi_action_reasoning'] = multi_action_reasoning
    env_kwargs['epsilon'] = epsilon

    # Only use environment_instruction from config if NOT using multi-action reasoning
    # When multi_action_reasoning=True, let the wrapper generate its own prompt
    if not multi_action_reasoning:
        if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
            environment_instruction = getattr(config.prompt.prompt, 'environment_instruction', None)
            if environment_instruction is not None:
                env_kwargs['instruction_prompt'] = environment_instruction

    env = FastSnakeLLMAgentsWrapper(env, **env_kwargs)

    return env
