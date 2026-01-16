import gymnasium as gym

from verl.envs.environments.babyai_text.clean_lang_wrapper import BABYAI_ACTION_SPACE

# Action descriptions for instruction prompts
ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object one step in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object one step in front of you",
}

POSSIBLE_ACTIONS = [
    "turn left",
    "turn to the left",
    "turn right",
    "turn to the right",
    "go forward",
    "move forward",
    "pick up",
    "pick it up",
    "drop",
    "toggle",
    "open",
    "turning left",
    "turning right",
    "moving forward",
    "picking up",
    "dropping",
    "toggling",
    "opening",
]

ACTION_VARIANTS = {
    "turnleft": "turn left",
    "turnright": "turn right",
    "goforward": "go forward",
    "pickup": "pick up",
}


class BabyAILLMAgentsWrapper(gym.Wrapper):
    """LLM-friendly wrapper for BabyAI environments.

    Handles action extraction from LLM output and reward modifications.

    Supports two modes:
    1. Standard mode: Model outputs <action>X</action>
    2. Multi-action reasoning mode: Model outputs reasoning for each action,
       then <decision>X</decision>
    """

    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.format_penalty = kwargs.get("format_penalty", 0.0)
        self.binary_reward = kwargs.get("binary_reward", False)
        self.multi_action_reasoning = kwargs.get("multi_action_reasoning", False)

        # Store locally (not via __getattr__ delegation)
        self.language_action_space = list(env.language_action_space)
        self._last_obs = None  # For mission retrieval

        # Instruction prompt: config override or default
        self.instruction_prompt = kwargs.get("instruction_prompt", None)
        if self.instruction_prompt is None:
            self.instruction_prompt = self._default_instruction_prompt()

    @property
    def default_action(self):
        return "go forward"

    @property
    def actions(self):
        return self.language_action_space

    @property
    def max_steps(self):
        return getattr(self.env, "max_steps", 100)

    def _default_instruction_prompt(self):
        action_strings = ",\n".join(
            f'"{action}": {desc}' for action, desc in ACTIONS.items()
        )

        if self.multi_action_reasoning:
            return f"""[Instructions]
You are an agent playing a navigation game. Your maximum response length: 300 words.

[Available Actions]
{action_strings}

[Response Format]
Reason about EACH action, then make a decision.

<actions>
<action name="turn left">Your reasoning...</action>
<action name="turn right">Your reasoning...</action>
<action name="go forward">Your reasoning...</action>
<action name="pick up">Your reasoning...</action>
<action name="drop">Your reasoning...</action>
<action name="toggle">Your reasoning...</action>
</actions>

<decision>your_chosen_action</decision>

[Rules]
- Your goal is to: {{mission}}
- You cannot "go forward" if blocked by wall/object
- Use 'toggle' to interact with objects in front of you"""
        else:
            return f"""[Instructions]
You are an agent playing a navigation game. Your maximum response length: 200 words.

[Available Actions]
{action_strings}

[Rules]
- Your goal is to: {{mission}}
- You cannot "go forward" if blocked by wall/object
- Use 'toggle' to interact with objects in front of you

Output your action in: <action>your_action</action>"""

    def get_instruction_prompt(self, *, mission=None):
        """Return instruction prompt with mission filled in.

        Args:
            mission: Optional mission string. If None, retrieves from env.
        """
        if mission is None:
            # Fallback chain
            if self._last_obs is not None and "mission" in self._last_obs:
                mission = self._last_obs["mission"]
            else:
                mission = getattr(self.env, "_mission", None)
            if mission is None:
                mission = "complete the task"

        return self.instruction_prompt.format(mission=mission)

    def restructure_obs(self, obs):
        """Validate and cache observation.

        CleanLangWrapper already provides the correct format.
        This validates keys and caches for mission retrieval.
        """
        if "text" not in obs:
            raise ValueError("Obs missing 'text' key - check wrapper chain")
        text = obs["text"]
        if "long_term_context" not in text or "short_term_context" not in text:
            raise ValueError("Obs['text'] missing required keys")

        self._last_obs = obs
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.restructure_obs(obs)
        return obs, info

    def step(self, action, is_valid=True):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not is_valid:
            reward = -self.format_penalty
        if self.binary_reward:
            reward = 1.0 if reward > 0 else reward
        obs = self.restructure_obs(obs)
        return obs, float(reward), terminated, truncated, info

    def get_text_action(self, action):
        return self.env.get_text_action(action)

    def get_stats(self):
        if hasattr(self.env, "get_stats"):
            return self.env.get_stats()
        return {}

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

    @staticmethod
    def _normalize_action(extracted):
        """Normalize action string to canonical form."""
        if extracted is None:
            return None
        extracted = extracted.lower().replace("_", " ")
        return ACTION_VARIANTS.get(extracted, extracted)

    @classmethod
    def extract_action(cls, action):
        """Parse LLM output (classmethod for evaluator).

        Standard mode only - no multi-action support.

        Returns:
            full_action: Original LLM output
            extracted_action: Parsed action before validation
            valid_action: Action to execute
            is_valid: Whether extraction succeeded
            metrics: Dict with validity ratio
        """
        full_action = str(action)
        extracted = cls.extract_action_from_xml_tag(full_action)
        extracted = cls._normalize_action(extracted)

        is_valid = extracted in BABYAI_ACTION_SPACE
        valid_action = extracted if is_valid else "go forward"

        metrics = {"behavior/valid_action_ratio": 1.0 if is_valid else 0.0}
        return full_action, extracted, valid_action, is_valid, metrics

    def extract_action_instance(self, action):
        """Parse LLM output (instance method for training).

        Supports multi-action reasoning mode.

        Returns:
            full_action: Original LLM output
            extracted_action: Parsed action before validation
            valid_action: Action to execute
            is_valid: Whether extraction succeeded
            metrics: Dict with validity ratio and analysis
        """
        full_action = str(action)

        if self.multi_action_reasoning:
            extracted = self.extract_decision_from_xml(full_action)
        else:
            extracted = self.extract_action_from_xml_tag(full_action)

        extracted = self._normalize_action(extracted)

        is_valid = extracted in self.language_action_space
        valid_action = extracted if is_valid else self.default_action

        # Additional metrics for analysis
        total_action_occurrences = sum(
            full_action.lower().count(p.lower()) for p in POSSIBLE_ACTIONS
        )
        backtrack_words = ["however", "different", "but", "wait", "won't", "can't", "cannot", "another"]
        total_backtrack = sum(
            full_action.lower().count(w) for w in backtrack_words
        )

        metrics = {
            "behavior/valid_action_ratio": 1.0 if is_valid else 0.0,
            "behavior/plan_length": total_action_occurrences,
            "behavior/backtrack_length": total_backtrack,
        }

        return full_action, extracted, valid_action, is_valid, metrics
