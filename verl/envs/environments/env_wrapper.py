import gymnasium as gym


class EnvWrapper(gym.Wrapper):
    """
    A wrapper class for gym environments to standardize interactions across different environments.
    It provides additional functionalities, such as handling specific observation processing,
    managing action validity, retrieving instruction prompts, and tracking failed action candidates.
    """

    def __init__(self, env, env_name, task_name):
        super().__init__(env)
        self.env_name = env_name
        self.task_name = task_name
        self.failed_candidates = []

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action, is_valid=True):
        obs, reward, terminated, truncated, info = self.env.step(action, is_valid)
        processed_obs = self._process_observation(obs)
        return processed_obs, reward, terminated, truncated, info

    def _process_observation(self, obs):
        if self.env_name in ["nle", "minihack"]:
            obs = obs
        elif self.env_name == "babyai":
            obs = obs
        elif self.env_name == "textworld":
            obs = obs
        elif self.env_name == "babaisai":
            obs = obs
        elif self.env_name == "crafter":
            obs = obs
        elif self.env_name == "fastsnake":
            obs = obs
        elif self.env_name == "frozenlake":
            obs = obs
        elif self.env_name == "webshop":
            obs = obs
        elif self.env_name == "overcooked":
            obs = obs
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

        return obs

    @property
    def actions(self):
        # This property should return the list of available actions
        return self.env.actions if hasattr(self.env, "actions") else list(range(len(self.env.action_space)))

    @property
    def language_action_space(self):
        # Forward to inner environment's language_action_space
        return self.env.language_action_space

    def get_text_action(self, action):
        return self.env.get_text_action(action)

    def get_instruction_prompt(self, instructions=None, info=None):
        if self.env_name == "nle":
            from verl.envs.environments.nle import get_instruction_prompt

            return get_instruction_prompt()
        elif self.env_name == "minihack":
            from verl.envs.environments.minihack import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babyai":
            return self.env.get_instruction_prompt(mission=instructions)
        elif self.env_name == "textworld":
            from verl.envs.environments.textworld import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babaisai":
            from verl.envs.environments.babaisai import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "crafter":
            from verl.envs.environments.crafter import get_instruction_prompt

            return get_instruction_prompt(self.env)
        elif self.env_name in ["fastsnake", "frozenlake", "webshop", "overcooked"]:
            return self.env.get_instruction_prompt()
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

    def check_action_validity(self, candidate_action):
        valid_action = None
        if candidate_action in self.env.language_action_space:
            valid_action = candidate_action
        else:
            valid_action = self.env.default_action
            self.failed_candidates.append(candidate_action)
        return valid_action

    def get_stats(self):
        return self.env.get_stats()
    
    def extract_action(self, action):
        return self.env.extract_action(action)

    def extract_action_instance(self, action):
        """Parse LLM output using instance-specific extraction if available.

        VecEnv's worker prefers `extract_action_instance` when present. Without this
        forwarder, the outer EnvWrapper would hide the inner environment's instance
        method and force the fallback `extract_action` path.
        """
        extract_fn = getattr(self.env, "extract_action_instance", None)
        if extract_fn is None:
            # Fail fast in multi-action mode rather than silently using the wrong parser.
            # The production symptom otherwise is: correct <decision> tags but 0% valid ratio.
            if bool(getattr(self.env, "multi_action_reasoning", False)):
                raise RuntimeError(
                    "multi_action_reasoning=True but env has no extract_action_instance(); "
                    "refusing to fall back to extract_action"
                )
            return self.env.extract_action(action)
        return extract_fn(action)
