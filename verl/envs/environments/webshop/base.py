import gymnasium as gym
import json


class WebShopLLMAgentsWrapper(gym.Wrapper):
    """Wrapper for WebShop environment to work with LLM agents."""
    
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.env = env
        self.format_penalty = kwargs.get('format_penalty', 0.1)
        self.binary_reward = kwargs.get('binary_reward', False)
        
        # WebShop has dynamic action space, so we don't define a fixed language_action_space
        # Instead, actions are validated against current available actions
        self._current_available_actions = None
        
    @property
    def language_action_space(self):
        """Return current valid actions based on environment state."""
        if self._current_available_actions is None:
            return []
        
        actions = []
        if self._current_available_actions.get('has_search_bar', False):
            actions.append('search')
        actions.extend([f'click[{item}]' for item in self._current_available_actions.get('clickables', [])])
        return actions
    
    @property
    def default_action(self):
        """Return a safe default action."""
        # Try to click 'back to search' as it's commonly available
        if self._current_available_actions:
            clickables = self._current_available_actions.get('clickables', [])
            if 'back to search' in clickables:
                return 'click[back to search]'
            elif clickables:
                return f'click[{clickables[0]}]'
        return 'search[product]'  # Fallback
    
    @property
    def max_steps(self):
        """Return maximum steps for the environment."""
        return getattr(self.env, 'max_steps', 100)
    
    @property
    def actions(self):
        """Return available actions."""
        return self.language_action_space
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
    
    def restructure_obs(self, obs):
        """Convert WebShop observation to standardized format.
        
        Args:
            obs: Raw observation from WebShop (text string)
        
        Returns:
            dict: Standardized observation with text and state
        """
        # Get available actions from environment
        self._current_available_actions = self.env.get_available_actions()
        
        # Format available actions nicely
        actions_str = "Available actions:\n"
        actions_str += f"  has_search_bar: {self._current_available_actions.get('has_search_bar', False)}\n"
        clickables = self._current_available_actions.get('clickables', [])
        if clickables:
            actions_str += f"  clickables: {json.dumps(clickables, indent=4)}"
        else:
            actions_str += "  clickables: []"
        
        # Combine observation with available actions
        combined_obs = f"{obs}\n\n{actions_str}"
        
        return {
            'text': {
                'long_term_context': combined_obs,
                'short_term_context': ''
            },
            'state': obs  # Keep raw observation as state
        }
    
    def reset(self, **kwargs):
        """Reset the environment.
        
        Returns:
            tuple: (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        obs = self.restructure_obs(obs)
        if info is None:
            info = {}
        return obs, info
    
    def step(self, action, is_valid=True):
        """Take a step in the environment.
        
        Args:
            action: Action string (e.g., 'search[keywords]' or 'click[button]')
            is_valid: Whether the action format was valid
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply format penalty if action was invalid
        if not is_valid:
            reward = reward - self.format_penalty
        
        # Apply binary reward if configured
        if self.binary_reward:
            reward = 1.0 if reward > 0 else reward
        
        obs = self.restructure_obs(obs)
        return obs, float(reward), terminated, truncated, info
    
    @staticmethod
    def extract_action_from_xml_tag(text: str, tag: str = "action") -> str:
        """Extract action from XML-style tags like <action>search[keywords]</action>.
        
        Args:
            text: Text containing XML tags
            tag: Tag name to extract from
        
        Returns:
            str: Extracted action or None if not found
        """
        try:
            return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()
        except (IndexError, AttributeError):
            return None
    
    def extract_action(self, action):
        """Extract and validate action from LLM output.
        
        Args:
            action: Raw LLM output string
        
        Returns:
            tuple: (full_action, extracted_action, valid_action, is_valid, metrics)
        """
        full_action = str(action)

        # Extract action from XML tags
        extracted_action = WebShopLLMAgentsWrapper.extract_action_from_xml_tag(full_action)

        if extracted_action is None:
            # No valid XML tag found
            is_valid = False
            valid_action = self.default_action
            metrics = {
                "behavior/valid_action_ratio": 0.0,
            }
            return full_action, extracted_action, valid_action, is_valid, metrics

        # Normalize action (lowercase, strip whitespace)
        extracted_action = extracted_action.strip().lower()

        # Check if it's a valid action format
        is_valid = self._is_valid_action(extracted_action)

        if is_valid:
            valid_action = extracted_action
        else:
            valid_action = self.default_action

        metrics = {
            "behavior/valid_action_ratio": 1.0 if is_valid else 0.0,
        }

        return full_action, extracted_action, valid_action, is_valid, metrics
    
    def _is_valid_action(self, action: str) -> bool:
        """Check if an action is valid given current environment state.
        
        Args:
            action: Action string to validate
        
        Returns:
            bool: True if action is valid
        """
        if self._current_available_actions is None:
            # If we don't have available actions yet, accept any reasonable format
            return action.startswith('search[') or action.startswith('click[')
        
        # Check if it's a search action and search bar is available
        if action.startswith('search[') and action.endswith(']'):
            return self._current_available_actions.get('has_search_bar', False)
        
        # Check if it's a click action on an available clickable
        if action.startswith('click[') and action.endswith(']'):
            # Extract the item name from click[item]
            item_name = action[6:-1].strip().lower()  # Remove 'click[' and ']'
            clickables = self._current_available_actions.get('clickables', [])
            # Check if item is in clickables (case-insensitive)
            return any(item_name == clickable.lower() for clickable in clickables)
        
        return False
    
    def get_stats(self):
        """Get environment statistics.
        
        Returns:
            dict: Statistics dictionary
        """
        stats = {}
        if hasattr(self.env, 'instruction_text'):
            stats['goal'] = self.env.instruction_text
        return stats
    
    def get_instruction_prompt(self):
        """Get instruction prompt for the agent.
        
        Returns:
            str: Instruction prompt
        """
        from verl.envs.environments.webshop import get_instruction_prompt
        return get_instruction_prompt(self.env)

