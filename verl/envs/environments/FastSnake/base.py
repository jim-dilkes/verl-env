import gymnasium as gym
from verl.envs.environments.FastSnake import ACTIONS


class FastSnakeLLMAgentsWrapper(gym.Wrapper):
    """Wrapper for FastSnake environment that provides LLM-compatible interface.

    Supports two modes:
    1. Standard mode: Model outputs <action>X</action>
    2. Multi-action reasoning mode: Model outputs reasoning for each action,
       then <decision>X</decision>
    """

    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.format_penalty = kwargs.get('format_penalty', 0.1)

        # Whether to use multi-action reasoning format
        self.multi_action_reasoning = kwargs.get('multi_action_reasoning', False)

        self.instruction_prompt = kwargs.get('instruction_prompt', None)
        if self.instruction_prompt is None:
            self.instruction_prompt = self._default_instruction_prompt()

    def _default_instruction_prompt(self):
        action_strings = ",\n".join(f'"{action}": {description}' for action, description in ACTIONS.items())

        if self.multi_action_reasoning:
            instruction = f"""[Instructions]
You are a helpful assistant controlling a snake in a grid-based game. Your maximum response length: 300 words.

[Available Actions]
{action_strings}

[Response Format]
You must reason about EACH available action, then make a decision.

First, analyze each action by generating an <action> tag for each:
<actions>
<action name="up">Your reasoning about why moving up might be good or bad...</action>
<action name="down">Your reasoning about why moving down might be good or bad...</action>
<action name="left">Your reasoning about why moving left might be good or bad...</action>
<action name="right">Your reasoning about why moving right might be good or bad...</action>
</actions>

Then, output your final decision:
<decision>your_chosen_action</decision>

[Rules]
- Eat apples to grow and score points
- Avoid walls, your own body, and enemy snakes
- You must output reasoning for ALL four actions before making your decision
"""
        else:
            instruction = f"""[Instructions]
You are a helpful assistant. You always respond by wrapping your thoughts in the correct XML tags. Your maximum response length: 200 words (tokens)
You are controlling a snake in a multi-player Snake game

[Available Actions]
{action_strings}

[Rules]
- You can move your head one space up, down, left, or right
- If you move onto an apple, you get 1 point and you gain a body segment
- You die if you move into a wall, another snake, or yourself
"""
        return instruction

    def get_instruction_prompt(self):
        return self.instruction_prompt

    @property
    def max_steps(self):
        return getattr(self.env, 'max_rounds', 100)

    @property
    def language_action_space(self):
        return list(ACTIONS.keys())

    @property
    def default_action(self):
        return self.actions[0]

    @property
    def actions(self):
        return self.language_action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def restructure_obs(self, obs):
        return {
            'text': {
                'long_term_context': self.env.game_state_text(),
                'short_term_context': ''
            },
            'state': obs
        }

    def step(self, action, is_valid=True):
        # Convert text action to integer for the underlying env
        if isinstance(action, str):
            action_int = self.env.STRING_ACTION_MAP.get(action.lower(), 0)
        else:
            action_int = action

        obs, reward, terminated, truncated, info = self.env.step(action_int)

        info['action_was_valid'] = is_valid
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

    @staticmethod
    def extract_decision_from_xml(text: str) -> str:
        """Extract decision from <decision>X</decision> tag."""
        try:
            return text.split("<decision>")[1].split("</decision>")[0].strip().lower()
        except (IndexError, AttributeError):
            return None

    @classmethod
    def extract_action(cls, action):
        """Parse LLM output (classmethod for evaluator compatibility).

        This is the classmethod version used by the evaluator. It uses standard
        action parsing (no multi-action reasoning, no epsilon-greedy).

        For training with multi-action/epsilon, use extract_action_instance().

        Returns:
            full_action: Original LLM output
            extracted_action: Parsed action before validation
            valid_action: Action to execute
            is_valid: Whether extraction succeeded
            metrics: Dict with validity ratio
        """
        full_action = str(action)

        # Standard mode: extract from <action>X</action>
        extracted = FastSnakeLLMAgentsWrapper.extract_action_from_xml_tag(full_action)

        if extracted is None:
            extracted = "__invalid__"
        valid_actions = list(ACTIONS.keys())
        is_valid = extracted in valid_actions
        valid_action = extracted if is_valid else valid_actions[0]

        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
        }

        return full_action, extracted, valid_action, is_valid, metrics

    def extract_action_instance(self, action):
        """Parse LLM output with instance-specific config (multi-action, epsilon).

        This is the instance method version used during training.
        Uses self.multi_action_reasoning and self.epsilon.

        Returns:
            full_action: Original LLM output
            extracted_action: Parsed action before validation
            valid_action: Action to execute (may be random if epsilon-greedy triggered)
            is_valid: Whether extraction succeeded
            metrics: Dict with validity ratio and exploration flag
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
