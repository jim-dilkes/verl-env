from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
import numpy as np
from verl.envs.environments.FrozenLake.base import FrozenLakeLLMAgentsWrapper


class CustomFrozenLakeEnv(FrozenLakeEnv):
    """Custom FrozenLake environment that adds ASCII rendering."""
    
    metadata = {
        'render_modes': ['human', 'ansi', 'ascii', 'rgb_array']
    }
        
    STRING_ACTION_MAP = {
        "left": 0,
        "down": 1,
        "right": 2,
        "up": 3
    }

    def __init__(self, desc=None, map_name="4x4", is_slippery=True, 
                 random_map_size=None, 
                 frozen_probability=None,
                 print_visualization: bool = True,
                 print_coordinates: bool = True,
                 print_axes: bool = False,
                 pad_observation_to_shape: tuple = None,
                 max_rounds=None):
        """
        Initialize the custom FrozenLake environment.
        
        Args:
            desc: Custom map description. If None and map_name is None, a random map will be generated.
            map_name: Name of the map to use. If None and desc is None, a random map will be generated.
            is_slippery: Whether the environment is slippery.
            print_axes: Whether to render coordinate axes.
            print_coordinates: Whether to render coordinates.
            print_visualization: Whether to print visualization.
            random_map_size: Size of the random map to generate. If provided, this will override map_name.
            frozen_probability: Probability of frozen tiles in random map.
            pad_observation_to_shape: Shape to pad observations to.
        """
        if random_map_size is not None:
            frozen_probability = frozen_probability or 0.8
            desc = generate_random_map(size=random_map_size, p=frozen_probability)
            map_name = None

        super().__init__(desc=desc, map_name=map_name, is_slippery=is_slippery, render_mode='ansi')
        self.goal = np.where(self.desc == b'G')
        self.shape = (len(self.desc), len(self.desc[0]))
        self.print_axes = print_axes
        self.print_coordinates = print_coordinates
        self.print_visualization = print_visualization
        self.pad_observation_to_shape = pad_observation_to_shape
        if not (self.print_visualization or self.print_coordinates):
            raise ValueError("Either print_visualization or print_coordinates must be True")

        self.max_rounds = max_rounds
        
    def _snake_observation_tensor_from_state(self):
        """
        Return a tensor of shape (n_channels, self.shape[0], self.shape[1]) with the following channels:
        - 0: Player position (snake head)
        - 1: Empty (snake body)
        - 2: Goal (apples)
        """
        state = self.s

        if self.pad_observation_to_shape is not None:
            if self.pad_observation_to_shape[0] < 3:
                raise ValueError("pad_observation_to_shape must have at least 3 channels")
            if self.pad_observation_to_shape[1] < self.shape[0] or self.pad_observation_to_shape[2] < self.shape[1]:
                raise ValueError("pad_observation_to_shape must be at least as large as the environment shape")
            observation = np.zeros(self.pad_observation_to_shape, dtype=np.int64)   
        else:
            observation = np.zeros((3, self.shape[0], self.shape[1]), dtype=np.int64)
        

        # Get player position
        player_row, player_col = self.state_to_coords(state)
        observation[0, player_row, player_col] = 2

        # Get all holes positions
        holes = np.where(self.desc == b'H')
        for hole_row, hole_col in zip(holes[0], holes[1]):  # Properly pair the coordinates
            observation[1, hole_row, hole_col] = 2
        
        # Get goal position
        goal_row, goal_col = self.goal
        observation[2, goal_row, goal_col] = 1

        # Create a flipped version with normal strides
        observation = observation[:, ::-1, ::-1].copy()
        return observation
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_round = 0
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, done, truncated, info = super().step(action, **kwargs)
        if done and reward == 1:
            info['success'] = True
        else:
            info['success'] = False

        self.current_round += 1
        if not done and not truncated and self.current_round >= self.max_rounds:
            done = False
            truncated = True
        return obs, reward, done, truncated, info

    def state_to_coords(self, state):
        row = state // self.shape[1]
        col = state % self.shape[1]
        return (row, col)

    def state_to_standard_coords(self, state):
        """x axis is flipped in standard coords, and x and y axis are reversed"""
        row, col = self.state_to_coords(state)
        return (col, self.shape[0] - 1 - row)
    
    def env_state_text(self):
        components = []
        if self.print_coordinates:
            components.append(
            f"The board size is {self.shape[1]}x{self.shape[0]}. Normal (X, Y) coordinates are used ranging from.\n"
            f"LEFT decreases X, RIGHT increases X, UP increases Y, and DOWN decreases Y.\n"
            f"Coordinates range from (0, 0) at bottom left to ({self.shape[1]-1}, {self.shape[0]-1}) at top right.\n"
            f"{self._coords_text()}"
            )
        if self.print_visualization:
            components.append(f"{self._render_ansi(axes=self.print_axes)}")

        return "\n".join(components)
     
    def render(self, mode: str = 'ansi'):
        if mode == 'ansi':
            return self._render_ansi(axes=self.print_axes)
        return super().render()

    def _render_ansi(self, axes: bool = False) -> str:
        """Render the environment as ANSI art with optional coordinate axes.
        
        Args:
            axes: Whether to include coordinate axes in the rendering
        """
        # Convert desc to numpy array of characters
        desc = np.array([[c.decode('utf-8') if isinstance(c, bytes) else c for c in row] for row in self.desc])
        
        # Create character mapping array
        char_map = {
            'S': '_',  # Start is empty space
            'F': '_',  # Frozen is empty space
            'H': 'O',  # Hole
            'G': 'G'   # Goal
        }
        
        # Create a vectorized mapping function
        vfunc = np.vectorize(lambda x: char_map[x])
        grid = vfunc(desc)
        
        # Add player position
        player_row, player_col = self.state_to_coords(self.s)
        grid[player_row, player_col] = 'P'
        
        if axes:
            # Create y-axis labels
            y_labels = np.array([str(i) for i in range(self.shape[0]-1, -1, -1)])
            # Create x-axis labels
            x_labels = np.array([' '] + [str(i) for i in range(self.shape[1])])
            
            # Add y-axis labels
            grid = np.column_stack((y_labels, grid))
            # Add x-axis labels
            grid = np.vstack((grid, x_labels))

        key_str = (
            "The meaning of each symbol in the state is:\n"
            "- P: Player\n"
            "- O: Hole\n"
            "- G: Goal\n"
            "- _: Empty space\n"
        )
            
        return key_str + "State:\n" + ('\n'.join([' '.join(row) for row in grid]))

    def _coords_text(self) -> str:
        """Returns a formatted string listing coordinates for each type of object in the environment.
        
        Returns:
            A string containing the coordinates of the player, holes, and goal.
        """
        # Get player position
        player_pos = self.state_to_standard_coords(self.s)
        
        # Find holes and goal
        holes = []
        goal = None
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                char = self.desc[row][col].decode('utf-8') if isinstance(self.desc[row][col], bytes) else self.desc[row][col]
                if char == 'H':
                    holes.append(self.state_to_standard_coords(row * self.shape[1] + col))
                elif char == 'G':
                    goal = self.state_to_standard_coords(row * self.shape[1] + col)
        
        # Format the text
        text = f"Player position: {player_pos}\n"
        text += f"Holes: {', '.join(str(hole) for hole in holes)}\n"
        text += f"Goal: {goal}"
        
        return text


def make_frozenlake_env(env_name, task, config, render_mode=None):
    """Create a FrozenLake environment with LLM agent wrapper."""
    
    frozenlake_kwargs = dict(config.envs.frozenlake_kwargs)
    
    env = CustomFrozenLakeEnv(**frozenlake_kwargs)
    
    # Prepare kwargs for the wrapper, checking prompt config first
    env_kwargs = dict(config.envs)
    
    # Check if prompt config has environment_instruction (takes priority over config.envs.instruction_prompt)
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
        environment_instruction = getattr(config.prompt.prompt, 'environment_instruction', None)
        if environment_instruction is not None:
            env_kwargs['instruction_prompt'] = environment_instruction
    
    env = FrozenLakeLLMAgentsWrapper(env, **env_kwargs)
    
    return env
