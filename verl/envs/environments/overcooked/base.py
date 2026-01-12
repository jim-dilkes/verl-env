import gymnasium as gym
from verl.envs.environments.overcooked import ACTIONS, ACTION_TO_IDX


class OvercookedLLMAgentsWrapper(gym.Wrapper):
    """LLM-friendly wrapper for Overcooked environment.

    Converts grid observations to text and handles action extraction from LLM output.

    Supports two modes:
    1. Standard mode: Model outputs <action>X</action>
    2. Multi-action reasoning mode: Model outputs reasoning for each action,
       then <decision>X</decision>
    """

    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.language_action_space = list(ACTIONS.keys())
        self.format_penalty = kwargs.get("format_penalty", 0.1)

        # Whether to use multi-action reasoning format
        self.multi_action_reasoning = kwargs.get("multi_action_reasoning", False)

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

        if self.multi_action_reasoning:
            return f"""[Instructions]
{game_desc}
Your goal is to cook and deliver soups as fast as possible to earn rewards. Your maximum response length: 400 words.

[How to Cook]
1. Pick up ingredients (e.g., onions) from ingredient piles using 'interact'
2. Place 3 ingredients in a pot using 'interact' while facing it
3. Wait for the soup to cook ({cook_time} ticks)
4. Pick up a dish from the dish pile
5. Pick up the cooked soup from the pot (with dish in hand)
6. Deliver the soup to the serving counter using 'interact'

[Available Actions]
{action_strings}

[Response Format]
You must reason about EACH available action, then make a decision.

First, analyze each action by generating an <action> tag for each:
<actions>
<action name="right">Your reasoning about moving right...</action>
<action name="down">Your reasoning about moving down...</action>
<action name="left">Your reasoning about moving left...</action>
<action name="up">Your reasoning about moving up...</action>
<action name="stay">Your reasoning about staying...</action>
<action name="interact">Your reasoning about interacting...</action>
</actions>

Then, output your final decision:
<decision>your_chosen_action</decision>

[Rules]
- You can only hold one object at a time
- Each soup requires exactly 3 ingredients
{cook_rule}{partner_rule}
- Delivering a completed soup earns +20 reward
- You must output reasoning for ALL six actions before making your decision"""
        else:
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

    @staticmethod
    def extract_decision_from_xml(text: str) -> str:
        """Extract decision from <decision>X</decision> tag."""
        try:
            return text.split("<decision>")[1].split("</decision>")[0].strip().lower()
        except (IndexError, AttributeError):
            return None

    def extract_action(self, action):
        """Parse LLM output (standard mode, no epsilon-greedy).

        Used by evaluator. For training with multi-action/epsilon, use extract_action_instance().
        """
        full_action = str(action)
        extracted_action = OvercookedLLMAgentsWrapper.extract_action_from_xml_tag(full_action)

        is_valid = extracted_action in self.language_action_space and extracted_action is not None
        valid_action = extracted_action if is_valid else self.default_action

        metrics = {
            "behavior/valid_action_ratio": float(is_valid),
        }

        return full_action, extracted_action, valid_action, is_valid, metrics

    def extract_action_instance(self, action):
        """Parse LLM output with instance-specific config (multi-action support).

        This is the instance method version used during training.
        Uses self.multi_action_reasoning for tag selection.
        Epsilon-greedy is handled centrally in vec_env.py.

        Returns:
            full_action: Original LLM output
            extracted_action: Parsed action before validation
            valid_action: Action to execute
            is_valid: Whether extraction succeeded
            metrics: Dict with validity ratio
        """
        full_action = str(action)

        # Parse based on mode
        if self.multi_action_reasoning:
            extracted = self.extract_decision_from_xml(full_action)
        else:
            extracted = self.extract_action_from_xml_tag(full_action)

        if extracted is None:
            extracted = "__invalid__"
        is_valid = extracted in self.language_action_space
        valid_action = extracted if is_valid else self.default_action

        # Epsilon-greedy is handled centrally in vec_env.py
        metrics = {"behavior/valid_action_ratio": is_valid * 1.0}

        return full_action, extracted, valid_action, is_valid, metrics

    def get_stats(self):
        return {}
