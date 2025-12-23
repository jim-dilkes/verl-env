import copy
import re

from verl.envs.captioners.base import BaseCaptioner


class NaiveCaptioner(BaseCaptioner): 
    """A captioner that generates actions based on observations without complex reasoning."""

    def __init__(self, prompt_builder, env_name=None, naive_instruction=None):
        """Initialize the NaiveCaptioner with a client and prompt builder.
        
        Args:
            prompt_builder: The prompt builder instance.
            env_name: Optional environment name.
            naive_instruction: Optional instruction string. If None, uses hardcoded default.
        """
        super().__init__(prompt_builder)
        self.naive_instruction = naive_instruction

    def get_obs(self, obs):
        """Generate the next action based on the observation.

        Args:
            obs (dict): The current observation in the environment.

        Returns:
            prompt: The prompt for the LLM, formatted as a list of dictionaries with 'role' and 'content'.
        """

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        # Use configurable instruction if provided, otherwise use hardcoded default
        if self.naive_instruction is None:
            naive_instruction = """Respond using ONLY valid XML with <think>...</think>, <plan>...</plan>, and <action>...</action> tags. Stop responding after the</action> tag.
[Response Template]
<think>{Think about the situation - what you should aim to do and what you should avoid doing}</think>
<plan>{Describe the immediate plan you will follow to achieve your goal and avoid bad outcomes. Be explicit about the action you will take: name the action}</plan>
<action>{Your selected action from the list above}</action>
            """.strip()
        else:
            naive_instruction = self.naive_instruction.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction
            
        # TODO: remove the transformation
        prompt = []
        for message in messages:
            role = message.role
            content = message.content
            prompt.append({"role": role, "content": content})

        return prompt
    
    def update_action(self, full_response, executed_action):
        self.prompt_builder.update_reasoning(full_response)
        self.prompt_builder.update_action(executed_action)
