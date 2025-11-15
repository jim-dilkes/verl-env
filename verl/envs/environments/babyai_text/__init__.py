from .clean_lang_wrapper import BabyAITextCleanLangWrapper
from .llm_agents_wrapper import BabyAILLMAgentsWrapper

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object one step in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object one step in front of you",
}


def get_instruction_prompt(env, mission):
    action_strings = ",\n".join(f"\"{action}\": {description}" for action, description in ACTIONS.items())

    instruction_prompt = f"""
[Instructions]
You are a helpful assistant. You always respond by wrapping your thoughts in the correct XML tags. Your maximum response length: 200 words (tokens)
You are an agent playing a simple navigation game. 
If there is a desired object you want to interact with or pickup in front of you, you can use the 'toggle' action to interact with it.

[Available Actions]
{action_strings}

[Rules]
- Your goal is to: {mission}.
- You cannot "go forward" if there is an object or wall in front of you.
- You cannot see the entire map, you may need to explore to find relevant objects.
""".strip()

    return instruction_prompt
