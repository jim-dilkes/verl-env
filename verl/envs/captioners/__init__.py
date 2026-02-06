from .prompt_builder import create_prompt_builder

def make_captioner(config):
    """Create a captioner agent based on the provided configuration.

    Args:
        config: Configuration object containing settings for the agent and client.

    Returns:
        Agent: An instance of the selected agent type, configured with the client and prompt builder.
    """
    prompt_builder = create_prompt_builder(config.envs.captioner)
    env_name = config.envs.env_name
    
    # Get prompt config if available (may be None if not specified)
    naive_instruction = None
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
        naive_instruction = getattr(config.prompt.prompt, 'naive_instruction', None)

    if config.envs.captioner.type == "naive":
        from .naive import NaiveCaptioner
        return NaiveCaptioner(prompt_builder, env_name, naive_instruction=naive_instruction)
    elif config.envs.captioner.type == "cot":
        from .cot import COTCaptioner
        return COTCaptioner(prompt_builder, env_name)
    elif config.envs.captioner.type == "multi_action":
        from .multi_action import MultiActionCaptioner
        return MultiActionCaptioner(prompt_builder, env_name, naive_instruction=naive_instruction)
    else:
        raise ValueError(f"Unknown captioner type: {config.envs.captioner.type}")
