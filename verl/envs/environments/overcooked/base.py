import gymnasium as gym
from verl.envs.environments.overcooked import ACTIONS, ACTION_TO_IDX


class OvercookedLLMAgentsWrapper(gym.Wrapper):
    """LLM-friendly wrapper for Overcooked environment.

    Converts grid observations to text and handles action extraction from LLM output.
    """

    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.language_action_space = list(ACTIONS.keys())
        self.format_penalty = kwargs.get("format_penalty", 0.1)

        self.instruction_prompt = kwargs.get("instruction_prompt", None)
        if self.instruction_prompt is None:
            self.instruction_prompt = self._default_instruction_prompt()

    def _default_instruction_prompt(self):
        action_strings = ",\n".join(
            f'"{action}": {description}' for action, description in ACTIONS.items()
        )

        solo_mode = getattr(self.env, "solo_mode", False)
        cook_time = getattr(self.env, "pot_cook_time", 20)

        if solo_mode:
            game_desc = "You are playing Overcooked solo. You control the only chef in a kitchen."
            partner_rule = ""
        else:
            game_desc = "You are playing Overcooked, a cooperative cooking game. You control one chef in a kitchen."
            partner_rule = "\n- Coordinate with your partner for efficiency"

        cook_rule = f"- Soups need {cook_time} ticks to cook before they can be served"

        return f"""[Instructions]
{game_desc}
Your goal is to cook and deliver soups as fast as possible to earn rewards.

[How to Cook]
1. Pick up ingredients (e.g., onions) from ingredient piles using 'interact'
2. Place 3 ingredients in a pot using 'interact' while facing it
3. Wait for the soup to cook ({cook_time} ticks)
4. Pick up a dish from the dish pile
5. Pick up the cooked soup from the pot (with dish in hand)
6. Deliver the soup to the serving counter using 'interact'

[Available Actions]
{action_strings}

[Rules]
- You can only hold one object at a time
- Each soup requires exactly 3 ingredients
{cook_rule}{partner_rule}
- Delivering a completed soup earns +20 reward"""

    def get_instruction_prompt(self):
        return self.instruction_prompt

    @property
    def max_steps(self):
        return getattr(self.env, "max_steps", 200)

    @property
    def default_action(self):
        return "stay"

    @property
    def actions(self):
        return self.language_action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def restructure_obs(self, obs):
        text_obs = self.env.render()
        return {
            "text": {
                "long_term_context": text_obs,
                "short_term_context": self.env.last_event,
            },
            "state": obs,
        }

    def step(self, action, is_valid=True):
        if isinstance(action, str):
            action_idx = ACTION_TO_IDX.get(action.lower(), ACTION_TO_IDX["stay"])
        else:
            action_idx = action

        obs, reward, terminated, truncated, info = self.env.step(action_idx)

        if not is_valid:
            reward = reward - self.format_penalty

        obs = self.restructure_obs(obs)
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.restructure_obs(obs)
        return obs, info

    def get_text_action(self, action):
        if isinstance(action, int):
            from verl.envs.environments.overcooked import IDX_TO_ACTION
            return IDX_TO_ACTION.get(action, "stay")
        return action

    @staticmethod
    def extract_action_from_xml_tag(text: str, tag: str = "action") -> str:
        try:
            return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip().lower()
        except (IndexError, AttributeError):
            return None

    def extract_action(self, action):
        full_action = str(action)
        extracted_action = OvercookedLLMAgentsWrapper.extract_action_from_xml_tag(full_action)

        is_valid = extracted_action in self.language_action_space and extracted_action is not None
        valid_action = extracted_action if is_valid else self.default_action

        metrics = {
            "behavior/valid_action_ratio": float(is_valid),
        }

        return full_action, extracted_action, valid_action, is_valid, metrics

    def get_stats(self):
        return {}
