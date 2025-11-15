import sys
from pathlib import Path
from verl.envs.environments.webshop.base import WebShopLLMAgentsWrapper


def make_webshop_env(env_name, task, config, render_mode=None):
    """Create a WebShop environment instance.
    
    Args:
        env_name: Name of the environment ('webshop')
        task: Task name (not used for webshop currently)
        config: Configuration object with environment settings
        render_mode: Rendering mode (not used for webshop)
    
    Returns:
        WebShopLLMAgentsWrapper: Wrapped WebShop environment
    """
    # Add WebShop to path (must be done here to work in Ray workers)
    webshop_path = Path(__file__).parent / "WebShop"
    if str(webshop_path) not in sys.path:
        sys.path.insert(0, str(webshop_path))
    
    # Import here after path is set (to work in Ray workers)
    from web_agent_site.envs.web_agent_text_env_no_flask import WebAgentTextEnv
    from web_agent_site.utils import get_dataset_file_path
    
    # Get WebShop-specific configuration
    webshop_kwargs = dict(config.envs.get('webshop_kwargs', {}))
    num_products = webshop_kwargs.get('num_products', 1000)
    
    # Get the appropriate dataset file path to reduce memory usage
    dataset_file_path = get_dataset_file_path(num_products)
    
    # Create base WebShop environment
    # Common parameters for WebShop:
    # - observation_mode: 'text', 'html', 'text_rich', 'url'
    # - num_products: number of products to use
    # - human_goals: whether to use human goals or synthetic
    # - goals_seed: seed for goal list generation (same across all instances)
    env = WebAgentTextEnv(
        observation_mode=webshop_kwargs.get('observation_mode', 'text'),
        file_path=dataset_file_path,  # Use the appropriate dataset file
        num_products=num_products,
        goals_seed=webshop_kwargs.get('goals_seed', None),
        # human_goals=webshop_kwargs.get('human_goals', 0),
    )
    
    # Wrap with LLM agents wrapper
    env = WebShopLLMAgentsWrapper(env, **config.envs)
    
    return env
