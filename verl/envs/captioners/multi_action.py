from verl.envs.captioners.base import BaseCaptioner


class MultiActionCaptioner(BaseCaptioner):
    """Captioner for multi-action reasoning format.

    Unlike NaiveCaptioner which adds its own response template, this captioner
    relies on the environment's instruction_prompt to define the format.
    This allows the environment to control whether to use standard or
    multi-action reasoning format.

    Supports optional naive_instruction (e.g., /no_think) appended to user message.
    """

    def __init__(self, prompt_builder, env_name=None, naive_instruction=None):
        """Initialize the MultiActionCaptioner.

        Args:
            prompt_builder: The prompt builder instance.
            env_name: Optional environment name.
            naive_instruction: Optional instruction to append to user message (e.g., /no_think).
        """
        super().__init__(prompt_builder)
        self.env_name = env_name
        self.naive_instruction = naive_instruction

    def get_obs(self, obs):
        """Generate prompt from observation.

        The environment's instruction_prompt already contains the response format,
        so we just pass through the observation context. Optionally appends
        naive_instruction (e.g., /no_think) to the last user message.

        Args:
            obs (dict): The current observation in the environment.

        Returns:
            prompt: The prompt for the LLM, formatted as a list of dicts.
        """
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        # Append naive_instruction to last user message if provided
        if self.naive_instruction:
            naive_instruction = self.naive_instruction.strip()
            if naive_instruction and messages and messages[-1].role == "user":
                messages[-1].content += "\n\n" + naive_instruction

        # Convert messages to dict format
        prompt = []
        for message in messages:
            role = message.role
            content = message.content
            prompt.append({"role": role, "content": content})

        return prompt

    def update_action(self, full_response, executed_action):
        """Update prompt builder with the action taken.

        Args:
            full_response: The full LLM response (including reasoning)
            executed_action: The action that was actually executed
        """
        self.prompt_builder.update_reasoning(full_response)
        self.prompt_builder.update_action(executed_action)
