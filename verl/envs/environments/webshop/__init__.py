# Action format descriptions for WebShop environment
ACTION_FORMAT = {
    "search": "search for products using keywords in format: search[keywords]",
    "click": "click on buttons or links in format: click[button_name]",
}


def get_instruction_prompt(env, info=None):
    """Generate instruction prompt for WebShop environment.
    
    Args:
        env: The WebShop environment instance
        info: Optional info dict (not used currently)
    
    Returns:
        str: Instruction prompt for the agent
    """
    # Get the goal/instruction text from the environment
    instruction_text = env.instruction_text if hasattr(env, 'instruction_text') else "Find and purchase the right product."
    
    action_descriptions = "\n".join(f"- {desc}" for desc in ACTION_FORMAT.values())
    
    instruction_prompt = f"""
[Instructions]
You are a helpful assistant shopping on a website. You always respond by wrapping your thoughts in the correct XML tags. Your maximum response length: 200 words (tokens)

[Your Goal]
{instruction_text}

[Available Action Formats]
{action_descriptions}

[Rules]
- If has_search_bar: True, you can use search[keywords] to search for products
- Click on items in the clickables list using click[item_name]
- Product ASINs (like 'b09mpy6s95') are clickable
- 'buy now' to purchase the selected product
- 'back to search' to return to search results
- 'next >' and '< prev' to navigate between pages
""".strip()
    
    return instruction_prompt

