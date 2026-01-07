# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
from gymnasium import spaces

from verl.envs.environments.env_wrapper import EnvWrapper


def make_env(env_name, task, config, render_mode=None):
    """Create an environment instance with the appropriate wrapper based on the environment name.

    Args:
        env_name (str): The name of the environment to create.
        task (str): The specific task within the environment.
        config (dict): Configuration settings for the environment.
        render_mode (str, optional): Rendering mode for the environment. Defaults to None.

    Returns:
        EnvWrapper: A wrapped environment instance.

    Raises:
        ValueError: If the environment name is not recognized.
    """
    # if env_name == "nle":
    #     from verl.envs.environments.nle.nle_env import make_nle_env

    #     base_env = make_nle_env(env_name, task, config, render_mode=render_mode)
    # elif env_name == "minihack":
    #     from verl.evs.environments.minihack.minihack_env import make_minihack_env

    #     base_env = make_minihack_env(env_name, task, config, render_mode=render_mode)
    # elif env_name == "babyai":
    #     from verl.envs.environments.babyai_text.babyai_env import make_babyai_env
    #     base_env = make_babyai_env(env_name, task, config, render_mode=render_mode)
    # elif env_name == "crafter":
    #     from verl.envs.environments.crafter.crafter_env import make_crafter_env

    #     base_env = make_crafter_env(env_name, task, config, render_mode=render_mode)
    # elif env_name == "textworld":
    #     from verl.envs.environments.textworld.textworld_env import make_textworld_env

    #     base_env = make_textworld_env(env_name, task, config, render_mode=render_mode)
    # elif env_name == "babaisai":
    #     from verl.envs.environments.babaisai.babaisai_env import make_babaisai_env
    #     base_env = make_babaisai_env(env_name, task, config, render_mode=render_mode)
        
    # elif env_name == "fastsnake":
    if env_name == "fastsnake":
        from verl.envs.environments.FastSnake.fastsnake_env import make_fastsnake_env
        base_env = make_fastsnake_env(env_name, task, config, render_mode=render_mode)
        
    elif env_name == "frozenlake":
        from verl.envs.environments.FrozenLake.frozen_lake_env import make_frozenlake_env
        base_env = make_frozenlake_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "webshop":
        from verl.envs.environments.webshop.webshop_env import make_webshop_env
        base_env = make_webshop_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "overcooked":
        from verl.envs.environments.overcooked.overcooked_env import make_overcooked_env
        base_env = make_overcooked_env(env_name, task, config, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return EnvWrapper(base_env, env_name, task)


def get_action_extraction_fn(env_name):
    if env_name == "fastsnake":
        from verl.envs.environments.FastSnake.base import FastSnakeLLMAgentsWrapper
        return FastSnakeLLMAgentsWrapper.extract_action
    elif env_name == "overcooked":
        from verl.envs.environments.overcooked.base import OvercookedLLMAgentsWrapper
        return OvercookedLLMAgentsWrapper.extract_action
    else:
        raise ValueError(f"Not accessible action extraction function for {env_name}")


class Strings(spaces.Space):
    """A custom Gym space for managing discrete string-based actions."""

    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def map(self, action):
        return self._dict[action]

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()
