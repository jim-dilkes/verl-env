import gymnasium as gym
from verl.envs.environments.FrozenLake import ACTIONS


class FrozenLakeLLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.language_action_space = list(ACTIONS.keys())
        self.format_penalty = kwargs.get('format_penalty', 0.1)
       
        self.instruction_prompt = kwargs.get('instruction_prompt', None)
        if self.instruction_prompt is None:
            self.instruction_prompt = self._default_instruction_prompt()

    def _default_instruction_prompt(self):
        action_strings = ",\n".join(f"\"{action}\": {description}" for action, description in ACTIONS.items())
        instruction = f"""[Instructions]
        You are a helpful assistant. You always respond by wrapping your thoughts in the correct XML tags. Your maximum response length: 200 words (tokens)
You are navigating the surface of a frozen lake. You must reach the goal. 

[Available Actions]
{action_strings}

[Rules]
- If you step on a hole, you will fall through and die. 
- The ice is slippery, so you might accidentally move in an perpendicular direction with every step.
- You can take one action out of up, down, left, or right
    """    
        return instruction

    def get_instruction_prompt(self):
        return self.instruction_prompt

    @property
    def max_steps(self):
        return getattr(self.env, 'max_episode_steps', 100)
        
    @property
    def default_action(self):
        return self.language_action_space[0]
        
    @property
    def actions(self):
        return self.language_action_space
        
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def restructure_obs(self, obs):
        return {'text': {'long_term_context': self.env.env_state_text(), 'short_term_context': ''},
            'state': obs}

    def step(self, action, is_valid=True):
        # Convert text action to integer for the underlying env
        if isinstance(action, str):
            action_int = self.env.STRING_ACTION_MAP.get(action.lower(), 0)
        else:
            action_int = action
            
        obs, reward, terminated, truncated, info = self.env.step(action_int)
        if not is_valid:
            reward = reward - self.format_penalty
        obs = self.restructure_obs(obs)
        return obs, reward * 1.0, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.restructure_obs(obs)
        return obs, info

    def get_text_action(self, action):
        return self.language_action_space[action]

    @staticmethod
    def extract_action_from_xml_tag(text: str, tag: str = "action") -> str:
        """Extract action from XML-style tags like <{tag}>UP</{tag}>."""
        try:
            return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip().lower()
        except (IndexError, AttributeError):
            return None

    def extract_action(self, action):
        full_action = str(action)
        action = FrozenLakeLLMAgentsWrapper.extract_action_from_xml_tag(full_action)

        is_valid = action in self.language_action_space and not action is None
        valid_action = action if is_valid else self.default_action

        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
        }

        extracted_action = action  # action after parsing but before validation
        return full_action, extracted_action, valid_action, is_valid, metrics

    def get_stats(self):
        return {}
