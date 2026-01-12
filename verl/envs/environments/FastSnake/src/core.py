# fast_snake/core.py

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque

# Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class FastSnake:
    """Snake game implementation using NumPy arrays.
    
    Args:
        width: Width of the game board
        height: Height of the game board
        max_rounds: Maximum number of rounds to play
        num_apples: Number of apples to spawn
        apple_reward: Reward for eating an apple
        apple_rng: Random number generator for apple positions
        num_bananas: Number of bananas to spawn
        banana_reward: Reward for eating a banana
        banana_rng: Random number generator for banana positions
        num_fires: Number of fires to spawn
        fire_reward: Reward for eating a fire
        fire_rng: Random number generator for fire positions
        hill_direction: Direction of the hill (None for no hill)
        destroy_at_bottom: Whether to destroy snakes at the bottom of the board
        include_absent_objects: Whether to include absent objects in the observation
    """
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 max_rounds: int = 100, 
                 num_apples: int = 1, 
                 apple_reward: int = 1,
                 apple_rng=None, 
                 num_bananas: int = 1, 
                 banana_reward: int = 5,
                 banana_rng=None, 
                 num_fires: int = 2, 
                 fire_reward: int = -1, 
                 fire_rng=None,
                 collision_reward: int = -1,
                 hill_direction: str = None,
                 destroy_at_bottom: bool = False,
                 include_absent_objects: bool = True,
                 no_respawn: bool = False,
                 terminate_on_single_snake: bool = True,):
        # Board representation constants
        self.EMPTY = 100
        self.SNAKE_BODY = 101
        self.APPLE = 102
        self.SNAKE_HEAD = 103
        self.BANANA = 104
        self.FIRE = 105
        
        self.width = width
        self.height = height
        self.max_rounds = max_rounds
        self.num_apples = num_apples if num_apples is not None else 0
        self.include_apples = num_apples > 0
        self.apple_reward = apple_reward if apple_reward is not None else 0
        self.num_bananas = num_bananas if num_bananas is not None else 0
        self.include_bananas = num_bananas > 0
        self.banana_reward = banana_reward if banana_reward is not None else 0
        self.num_fires = num_fires if num_fires is not None else 0
        self.include_fires = num_fires > 0
        self.fire_reward = fire_reward if fire_reward is not None else 0
        self.collision_reward = collision_reward if collision_reward is not None else 0
        self.hill_direction = hill_direction
        self.destroy_at_bottom = destroy_at_bottom
        self.include_absent_objects = include_absent_objects
        self.num_object_types = self.include_apples + self.include_bananas + self.include_fires
        self.no_respawn = no_respawn
        self.terminate_on_single_snake = terminate_on_single_snake
        # Validate hill_direction
        if hill_direction is not None:
            if hill_direction not in ["up", "down", "left", "right"]:
                raise ValueError(f"Invalid hill_direction: {hill_direction}. Must be one of 'up', 'down', 'left', 'right', or None")
            
            # Rolling apples not compatible with fires
            if num_fires is not None and num_fires > 0:
                raise ValueError("Rolling apples (hill_direction) is not compatible with fires. Please set num_fires to 0 or hill_direction to None.")
        
        # Use provided RNGs or create new ones
        self.apple_rng = apple_rng if apple_rng is not None else np.random.RandomState()
        self.banana_rng = banana_rng if banana_rng is not None else np.random.RandomState()
        self.fire_rng = fire_rng if fire_rng is not None else np.random.RandomState()
        
        # Pre-compute movement deltas for efficiency
        self.MOVE_DELTAS = {
            UP: np.array([0, 1]),
            DOWN: np.array([0, -1]),
            LEFT: np.array([-1, 0]),
            RIGHT: np.array([1, 0])
        }
        
        # Map string direction to action constant
        self.DIR_TO_ACTION = {
            "up": UP,
            "down": DOWN,
            "left": LEFT,
            "right": RIGHT
        }
        
        # Initialize game state
        self.reset()
    
    def reset(self):
        """Reset the game state."""
        self.board = np.full((self.height, self.width), self.EMPTY, dtype=np.int8)
        self.round_number = 0
        self.game_over = False
        
        # Initialize snakes dict with numpy arrays for positions
        self.snakes = {}
        self.scores = {}
        
        # Place initial objects
        self.apples = []
        for _ in range(self.num_apples):
            self._place_apple()
            
        self.bananas = []
        for _ in range(self.num_bananas):
            self._place_banana()
            
        self.fires = []
        for _ in range(self.num_fires):
            self._place_fire()
            
        return self.get_state()
    
    def add_snake(self, snake_id: str) -> None:
        """Add a new snake to the game."""
        if snake_id in self.snakes:
            raise ValueError(f"Snake {snake_id} already exists")
            
        # Initialize snake with random position
        pos = self._random_free_cell()
        self.snakes[snake_id] = {
            'positions': deque([pos]),
            'alive': True
        }
        self.scores[snake_id] = 0
        self._update_board()
    
    def _random_free_cell(self) -> Tuple[int, int]:
        """Find a random empty cell efficiently."""
        empty_cells = np.argwhere(self.board == self.EMPTY)
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        
        # Sort empty cells for deterministic selection with same seed
        # This ensures consistent results even if board state changes
        empty_cells = sorted(empty_cells, key=lambda cell: (cell[0], cell[1]))
        
        idx = self.apple_rng.randint(len(empty_cells))
        return tuple(empty_cells[idx])
    
    def _random_free_cell_banana(self) -> Tuple[int, int]:
        """Find a random empty cell for bananas."""
        empty_cells = np.argwhere(self.board == self.EMPTY)
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        
        empty_cells = sorted(empty_cells, key=lambda cell: (cell[0], cell[1]))
        idx = self.banana_rng.randint(len(empty_cells))
        return tuple(empty_cells[idx])
    
    def _random_free_cell_fire(self) -> Tuple[int, int]:
        """Find a random empty cell for fires."""
        empty_cells = np.argwhere(self.board == self.EMPTY)
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        
        empty_cells = sorted(empty_cells, key=lambda cell: (cell[0], cell[1]))
        idx = self.fire_rng.randint(len(empty_cells))
        return tuple(empty_cells[idx])
    
    def _place_apple(self) -> None:
        """Place a new apple on an empty cell, respecting hill direction if set."""
        # Initial game setup - apples can spawn anywhere
        if self.round_number == 0 or self.hill_direction is None:
            try:
                pos = self._random_free_cell()
                self.apples.append(pos)
                y, x = pos
                self.board[y, x] = self.APPLE
            except RuntimeError:
                pass  # No empty cells for apple
            return
            
        # After game start with hill direction, apples spawn at the "upper" edge
        try:
            pos = self._get_apple_spawn_position()
            if pos:
                self.apples.append(pos)
                y, x = pos
                self.board[y, x] = self.APPLE
        except RuntimeError:
            pass  # No valid positions available
    
    def _get_apple_spawn_position(self) -> Optional[Tuple[int, int]]:
        """Get a valid spawn position for an apple based on hill direction."""
        if self.hill_direction is None:
            return self._random_free_cell()
            
        # Define the edge to start checking based on hill direction
        if self.hill_direction == "right":
            # Start from left edge (x=0) and move right if no spaces
            for x in range(self.width):
                empty_positions = []
                for y in range(self.height):
                    if self.board[y, x] == self.EMPTY:
                        empty_positions.append((x, y))
                # If found empty positions in this column, choose a random one
                if empty_positions:
                    idx = self.apple_rng.randint(len(empty_positions))
                    return empty_positions[idx]
                        
        elif self.hill_direction == "left":
            # Start from right edge (x=width-1) and move left if no spaces
            for x in range(self.width - 1, -1, -1):
                empty_positions = []
                for y in range(self.height):
                    if self.board[y, x] == self.EMPTY:
                        empty_positions.append((x, y))
                # If found empty positions in this column, choose a random one
                if empty_positions:
                    idx = self.apple_rng.randint(len(empty_positions))
                    return empty_positions[idx]
                        
        elif self.hill_direction == "up":
            # Start from bottom edge (y=0) and move up if no spaces
            for y in range(self.height):
                empty_positions = []
                for x in range(self.width):
                    if self.board[y, x] == self.EMPTY:
                        empty_positions.append((x, y))
                # If found empty positions in this row, choose a random one
                if empty_positions:
                    idx = self.apple_rng.randint(len(empty_positions))
                    return empty_positions[idx]
                        
        elif self.hill_direction == "down":
            # Start from top edge (y=height-1) and move down if no spaces
            for y in range(self.height - 1, -1, -1):
                empty_positions = []
                for x in range(self.width):
                    if self.board[y, x] == self.EMPTY:
                        empty_positions.append((x, y))
                # If found empty positions in this row, choose a random one
                if empty_positions:
                    idx = self.apple_rng.randint(len(empty_positions))
                    return empty_positions[idx]
        
        # If we've checked all edges and found no space, use any random empty cell
        return self._random_free_cell()
    
    def _place_banana(self) -> None:
        """Place a new banana on an empty cell."""
        try:
            pos = self._random_free_cell_banana()
            self.bananas.append(pos)
            y, x = pos
            self.board[y, x] = self.BANANA
        except RuntimeError:
            pass  # No empty cells for banana
    
    def _place_fire(self) -> None:
        """Place a new fire on an empty cell."""
        try:
            pos = self._random_free_cell_fire()
            self.fires.append(pos)
            y, x = pos
            self.board[y, x] = self.FIRE
        except RuntimeError:
            pass  # No empty cells for fire
    
    def _update_board(self) -> None:
        """Update the board state efficiently."""
        # Clear the board
        self.board.fill(self.EMPTY)
        
        # Place apples
        for ax, ay in self.apples:
            self.board[ay, ax] = self.APPLE
            
        # Place bananas
        for bx, by in self.bananas:
            self.board[by, bx] = self.BANANA
            
        # Place fires
        for fx, fy in self.fires:
            self.board[fy, fx] = self.FIRE
        
        # Place snakes
        for snake_id, snake in self.snakes.items():
            if not snake['alive']:
                continue
                
            # Place body
            for x, y in list(snake['positions'])[1:]:
                self.board[y, x] = self.SNAKE_BODY
                
            # Place head
            positions_list = list(snake['positions'])
            if positions_list: # Check if snake has any positions
                hx, hy = positions_list[0]
                if 0 <= hy < self.height and 0 <= hx < self.width: # Check bounds
                    self.board[hy, hx] = self.SNAKE_HEAD
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, dict]:
        """
        Execute one game step with given actions.
        
        Args:
            actions: Dict mapping snake_id to action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            observations: Dict mapping snake_id to observation array
            rewards: Dict mapping snake_id to reward value
            done: Whether the game is over
            info: Additional information
        """
        if self.game_over:
            raise RuntimeError("Game is already over")
            
        rewards = {sid: 0.0 for sid in self.snakes}
        
        # Track how many apples need to be respawned after all movements
        apples_to_respawn = 0
        
        # ---- PHASE 1: Calculate all new head positions ----
        new_heads = {}
        for snake_id, action in actions.items():
            if not self.snakes[snake_id]['alive']:
                continue
                
            head = np.array(self.snakes[snake_id]['positions'][0])
            new_heads[snake_id] = tuple(head + self.MOVE_DELTAS[action])
        
        # ---- PHASE 2: Check for head-to-head collisions ----
        # Find duplicate head positions (head-to-head collisions)
        head_counts = {}
        for snake_id, new_head in new_heads.items():
            if new_head in head_counts:
                head_counts[new_head].append(snake_id)
            else:
                head_counts[new_head] = [snake_id]
        
        # Mark snakes involved in head-to-head collisions as not alive
        for pos, snake_ids in head_counts.items():
            if len(snake_ids) > 1:
                # More than one snake moving to the same position - collision!
                for snake_id in snake_ids:
                    if self.snakes[snake_id]['alive']:
                        self.snakes[snake_id]['alive'] = False
                        rewards[snake_id] += self.collision_reward
                        
        # ---- PHASE 3: Process individual snake moves and collisions ----
        success = {sid: False for sid in self.snakes}
        for snake_id, new_head in new_heads.items():
            snake = self.snakes[snake_id]
            if not snake['alive']:
                continue  # Skip already dead snakes (from head-to-head collisions)
                
            x, y = new_head
            
            # Check bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                snake['alive'] = False
                rewards[snake_id] += self.collision_reward
                continue
            
            # Check collisions with snake bodies on the current board
            # (This won't catch head-to-head collisions, which we handled above)
            if self.board[y, x] in [self.SNAKE_BODY, self.SNAKE_HEAD]:
                snake['alive'] = False
                rewards[snake_id] += self.collision_reward
                continue
            
            # Move snake
            snake['positions'].appendleft(new_head)
            
            # Check apple
            if new_head in self.apples:
                self.scores[snake_id] += self.apple_reward
                rewards[snake_id] += float(self.apple_reward)
                self.apples.remove(new_head)
                apples_to_respawn += 1  # Track that we need to spawn an apple later
                if self.apple_reward > 0:
                    success[snake_id] = True
                # Don't pop the tail - snake grows when eating an apple
            # Check banana
            elif new_head in self.bananas:
                self.scores[snake_id] += self.banana_reward
                rewards[snake_id] += float(self.banana_reward)
                self.bananas.remove(new_head)
                if not self.no_respawn:
                    self._place_banana()
                if self.banana_reward > 0:
                    success[snake_id] = True
                # Don't pop the tail - snake grows when eating a banana
            # Check fire
            elif new_head in self.fires:
                self.scores[snake_id] += self.fire_reward
                rewards[snake_id] += float(self.fire_reward)
                self.fires.remove(new_head)
                if not self.no_respawn:
                    self._place_fire()
                if self.fire_reward > 0:
                    success[snake_id] = True
                # Don't pop the tail - snake grows when eating a fire
            else:
                # Only pop the tail if we didn't eat anything
                snake['positions'].pop()
        
        # Update the board after all snakes have moved
        self._update_board()
        
        # ---- PHASE 4: Roll apples down the hill (if hill_direction is set) ----
        if self.hill_direction is not None:
            # Store the scores before rolling to calculate rewards
            pre_roll_scores = {sid: score for sid, score in self.scores.items()}
            
            # Roll apples and count how many need to be respawned
            additional_apples_to_respawn = self._roll_apples()
            apples_to_respawn += additional_apples_to_respawn
            
            # Update rewards based on score changes during rolling
            for snake_id in self.snakes:
                if snake_id in pre_roll_scores:
                    # If score increased during rolling, add that to rewards
                    score_change = self.scores[snake_id] - pre_roll_scores[snake_id]
                    if score_change > 0:
                        rewards[snake_id] += float(score_change)
            
            # Update the board after apples have moved
            self._update_board()
        
        # ---- PHASE 5: Respawn all apples that were eaten ----
        # Only do this ground respawn for apples because only apples can be rolled down the hill
        if not self.no_respawn:
            for _ in range(apples_to_respawn):
                self._place_apple()
            
        # Final board update after all changes
        self._update_board()
        
        self.round_number += 1
        
        # Check game over conditions
        alive_snakes = sum(s['alive'] for s in self.snakes.values())
        self.game_over = (alive_snakes <= 1 and self.terminate_on_single_snake) or (alive_snakes == 0 and not self.terminate_on_single_snake) or (self.round_number >= self.max_rounds) 
        
        # Success is a boolean indicating whether any snake ate an item with positive reward
        return (
            self.get_observations(),
            rewards,
            self.game_over,
            {'scores': self.scores.copy(), 'round': self.round_number, 'success': success}
        )
    
    def _roll_apples(self) -> int:
        """
        Roll apples down the hill based on hill_direction.
        Uses multiple passes to ensure proper cascading of apple movements.
        Each apple can only move once per step.
        
        Returns:
            int: Number of apples eaten during rolling that need to be respawned.
        """
        if self.hill_direction is None:
            return 0
            
        # Get movement delta for the hill direction
        delta = self.MOVE_DELTAS[self.DIR_TO_ACTION[self.hill_direction]]
        
        # Track apples eaten during rolling
        apples_removed_count = 0
        
        # Track which original positions have already moved
        # This ensures each apple only moves once per step
        original_positions = {pos: pos for pos in self.apples}
        moved_positions = set()
        
        # Continue rolling until no more movements occur
        movement_occurred = True
        
        while movement_occurred:
            movement_occurred = False
            
            # Process apples in the direction of movement
            apples_to_process = list(self.apples)
            
            if self.hill_direction == "right":
                # Process from right to left (highest x to lowest)
                apples_to_process.sort(key=lambda pos: (-pos[0], pos[1]))
            elif self.hill_direction == "left":
                # Process from left to right (lowest x to highest)
                apples_to_process.sort(key=lambda pos: (pos[0], pos[1]))
            elif self.hill_direction == "up":
                # Process from top to bottom (highest y to lowest)
                apples_to_process.sort(key=lambda pos: (-pos[1], pos[0]))
            elif self.hill_direction == "down":
                # Process from bottom to top (lowest y to highest)
                apples_to_process.sort(key=lambda pos: (pos[1], pos[0]))
                
            # Keep track of new positions
            new_apple_positions = []
            snake_heads = {}
            
            # Get all snake head positions for eating checks
            for snake_id, snake in self.snakes.items():
                if snake['alive'] and snake['positions']:
                    snake_heads[tuple(snake['positions'][0])] = snake_id
                    
            # Track apples to remove (eaten during rolling)
            apples_to_remove = []
            
            for apple_pos in apples_to_process:
                # Get the original position this apple started from
                original_pos = original_positions.get(apple_pos, apple_pos)
                
                # Skip if this original position has already moved in this step
                if original_pos in moved_positions:
                    new_apple_positions.append(apple_pos)
                    continue
                    
                apple_x, apple_y = apple_pos
                new_x = apple_x + int(delta[0])
                new_y = apple_y + int(delta[1])
                new_pos = (new_x, new_y)
                
                # Check if the apple would roll off the board
                if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                    if self.destroy_at_bottom:
                        # If destroy_at_bottom is True, remove the apple and count it as eaten
                        apples_to_remove.append(apple_pos)
                        apples_removed_count += 1
                        movement_occurred = True
                        moved_positions.add(original_pos)
                    else:
                        # Otherwise, keep it in place
                        new_apple_positions.append(apple_pos)
                    continue
                    
                # Check if the apple would roll into a snake's head
                if new_pos in snake_heads:
                    # Snake eats the apple
                    snake_id = snake_heads[new_pos]
                    self.scores[snake_id] += self.apple_reward
                    # Track this apple for removal
                    apples_to_remove.append(apple_pos)
                    apples_removed_count += 1
                    movement_occurred = True  # Count as movement for cascade checks
                    # Mark original position as moved
                    moved_positions.add(original_pos)
                    continue
                    
                # Check collisions using current board state
                board_val = self.board[new_y, new_x]
                collision_objects = [self.SNAKE_BODY, self.APPLE, self.BANANA]
                
                # Check if the apple would roll into a position where another apple is being processed
                collision_with_new_pos = new_pos in new_apple_positions
                
                if board_val in collision_objects or collision_with_new_pos:
                    new_apple_positions.append(apple_pos)  # Stay in place
                    continue
                    
                # Apple can move to the new position
                new_apple_positions.append(new_pos)
                # Update mapping from new position to original position
                original_positions[new_pos] = original_pos
                # Mark that this original position has moved
                moved_positions.add(original_pos)
                movement_occurred = True  # Movement occurred, do another pass
            
            # Remove eaten apples
            for pos in apples_to_remove:
                if pos in self.apples:
                    self.apples.remove(pos)
                    if pos in original_positions:
                        del original_positions[pos]
                    
            # Update all apple positions
            self.apples = [pos for pos in new_apple_positions if pos not in apples_to_remove]
            
            # Update the board to reflect the new apple positions
            self._update_board()
        
        # Return number of apples eaten during rolling
        return apples_removed_count
    
    def get_observations(self, include_absent_objects: bool = None) -> Dict[str, np.ndarray]:
        """Get observations for all snakes."""
        if include_absent_objects is None:
            include_absent_objects = self.include_absent_objects

        return {
            snake_id: self._get_snake_observation(snake_id, include_absent_objects)
            for snake_id in self.snakes
        }
    
    # def _get_board_observation(self, snake_id: str, include_absent_objects: bool = True) -> np.ndarray:
    #     """Get observation of the entire board from a specific snake's perspective.
        
    #     Returns a 5xHxW array where each channel is:
    #     - Channel 0: Current snake's head location (binary)
    #     - Channel 1: All snake bodies (including heads) (binary)
    #     - Channel 2: Apples (binary)
    #     - Channel 3: Bananas (binary)
    #     - Channel 4: Fires (binary)
    #     """
    #     # Initialize observation array with zeros
    #     if not include_absent_objects:
    #         obs = np.zeros((2 + self.num_object_types, self.height, self.width), dtype=np.int8)
    #     else:
    #         obs = np.zeros((5, self.height, self.width), dtype=np.int8)
        
    #     # Channel 0: Current snake's head
    #     if self.snakes[snake_id]['alive']:
    #         head_pos = self.snakes[snake_id]['positions'][0]
    #         obs[0, head_pos[1], head_pos[0]] = 1
        
    #     # Channel 1: All snake bodies (including heads)
    #     # Convert all snake positions to numpy array for vectorized operations
    #     all_positions = []
    #     for snake in self.snakes.values():
    #         if snake['alive']:
    #             all_positions.extend(snake['positions'])
    #     if all_positions:
    #         positions = np.array(all_positions)
    #         obs[1, positions[:, 1], positions[:, 0]] = 1
        
    #     # Channel 2: Apples
    #     if self.apples:
    #         apples = np.array(self.apples)
    #         obs[2, apples[:, 1], apples[:, 0]] = 1
        
    #     # Channel 3: Bananas
    #     if self.bananas:
    #         bananas = np.array(self.bananas)
    #         obs[3, bananas[:, 1], bananas[:, 0]] = 1
        
    #     # Channel 4: Fires
    #     if self.fires:
    #         fires = np.array(self.fires)
    #         obs[4, fires[:, 1], fires[:, 0]] = 1
        
        return obs
    
    def _get_snake_observation(self, snake_id: str, include_absent_objects: bool = True) -> np.ndarray:
        """Get observation from one snake's perspective.
        
        Args:
            snake_id: ID of the snake to get observation for
            include_absent_objects: Whether to include absent objects in the observation
        """
        
        if not include_absent_objects:
            number_of_channels = 2 + self.num_object_types
        else:
            number_of_channels = 5

        obs = np.zeros((number_of_channels, self.height, self.width), dtype=np.int8)
        
        # Channel 0: Current snake
        if self.snakes[snake_id]['alive']:
            positions = self.snakes[snake_id]['positions']
            head = positions[0]
            # Mark head
            obs[0, head[1], head[0]] = 2
            # Mark body
            for x, y in list(positions)[1:]:
                obs[0, y, x] = 1

        # Channel 1: Other snakes
        for other_id, snake in self.snakes.items():
            if other_id != snake_id and snake['alive']:
                positions = snake['positions']
                head = positions[0]
                # Mark head
                obs[1, head[1], head[0]] = 2
                # Mark body
                for x, y in list(positions)[1:]:
                    obs[1, y, x] = 1
        
        objects_channel_index = 2
        # Channel : Apples
        if include_absent_objects or self.num_apples > 0:
            for x, y in self.apples:
                obs[objects_channel_index, y, x] = 1
            self.apple_channel_index = objects_channel_index
            objects_channel_index += 1

        # Channel : Bananas
        if include_absent_objects or self.num_bananas > 0:
            for x, y in self.bananas:
                obs[objects_channel_index, y, x] = 1
            self.banana_channel_index = objects_channel_index
            objects_channel_index += 1
            
        # Channel : Fires
        if include_absent_objects or self.num_fires > 0:
            for x, y in self.fires:
                obs[objects_channel_index, y, x] = 1
            self.fire_channel_index = objects_channel_index
            objects_channel_index += 1
        

        
        return obs
    
    def get_state(self) -> np.ndarray:
        """Get the raw board state."""
        return self.board.copy()
    
    def render_text(self, print_axes: bool = False, use_color_mode: bool = False) -> str:
        """
        Returns a string representation of the board.
        
        If use_color_mode is False (default):
        # = empty space
        A = apple
        B = banana
        F = fire
        T = snake tail
        1,2,3... = snake head (showing player number based on order in self.snakes)
        
        If use_color_mode is True:
        · = empty space (blank)
        🍎 = apple (red)
        🍌 = banana (yellow)
        🔥 = fire (red)
        ■ = snake body (green for player 1, red for others)
        1,2,3... = snake head (colored by player)
        """
        if not use_color_mode:
            # Original visualization mode
            key_str = (
                "The meaning of each symbol in the state is:\n"
                "- 1: Your snake head\n"
                "- 2: Enemy snake head\n"
                "- T: Snake body\n"
            )
            key_str +=  "- A: Apple\n" if self.num_apples > 0 else ""
            key_str +=  "- B: Banana\n" if self.num_bananas > 0 else ""
            key_str +=  "- F: Fire\n" if self.num_fires > 0 else ""
            key_str +=  "- _: Empty space"

            # Create empty board
            board = [['_' for _ in range(self.width)] for _ in range(self.height)]

            # Place apples
            for ax, ay in self.apples:
                if 0 <= ay < self.height and 0 <= ax < self.width:
                    board[ay][ax] = 'A'
                    
            # Place bananas
            for bx, by in self.bananas:
                if 0 <= by < self.height and 0 <= bx < self.width:
                    board[by][bx] = 'B'
                    
            # Place fires
            for fx, fy in self.fires:
                if 0 <= fy < self.height and 0 <= fx < self.width:
                    board[fy][fx] = 'F'

            # Place snakes
            for i, (snake_id, snake_data) in enumerate(self.snakes.items(), start=1):
                if not snake_data['alive']:
                    continue

                positions = snake_data['positions']
                # Place snake body
                for pos_idx, (x, y) in enumerate(positions):
                    if 0 <= y < self.height and 0 <= x < self.width:
                        if pos_idx == 0:  # Head
                            board[y][x] = str(i)  # Use snake number (1, 2, 3...) for head
                        else:  # Body/tail
                            board[y][x] = 'T'
        else:
            # Colored visualization mode
            # ANSI color codes
            RESET = "\033[0m"
            RED = "\033[91m"
            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            MAGENTA = "\033[95m"
            CYAN = "\033[96m"
            BG_GREEN = "\033[42m"
            BG_RED = "\033[41m"

            key_str = (
                "The meaning of each symbol in the state is:\n"
                f"{BG_GREEN}1{RESET}: Your snake head\n"
                f"{GREEN}■{RESET}: Your snake body\n"
                f"{BG_RED}2{RESET}: Enemy snake head\n"
                f"{RED}■{RESET}: Enemy snake body\n"
            )
            key_str += f"{RED}●{RESET}: Apple\n" if self.num_apples > 0 else ""
            key_str += f"{YELLOW}★{RESET}: Banana\n" if self.num_bananas > 0 else ""
            key_str += f"{RED}†{RESET}: Fire\n" if self.num_fires > 0 else ""
            key_str += "·: Empty space"

            # Create empty board
            board = [["·" for _ in range(self.width)] for _ in range(self.height)]

            # Place apples
            for ax, ay in self.apples:
                if 0 <= ay < self.height and 0 <= ax < self.width:
                    board[ay][ax] = f"{RED}●{RESET}"
                    
            # Place bananas
            for bx, by in self.bananas:
                if 0 <= by < self.height and 0 <= bx < self.width:
                    board[by][bx] = f"{YELLOW}★{RESET}"
                    
            # Place fires
            for fx, fy in self.fires:
                if 0 <= fy < self.height and 0 <= fx < self.width:
                    board[fy][fx] = f"{RED}†{RESET}"

            # Place snakes
            for i, (snake_id, snake_data) in enumerate(self.snakes.items(), start=1):
                if not snake_data['alive']:
                    continue

                positions = snake_data['positions']
                # Choose colors based on snake number
                head_bg = BG_GREEN if i == 1 else BG_RED
                body_color = GREEN if i == 1 else RED
                
                # Place snake body
                for pos_idx, (x, y) in enumerate(positions):
                    if 0 <= y < self.height and 0 <= x < self.width:
                        if pos_idx == 0:  # Head
                            board[y][x] = f"{head_bg}{i}{RESET}"  # Use snake number (1, 2, 3...) for head
                        else:  # Body/tail
                            board[y][x] = f"{body_color}■{RESET}"

        # Build the string representation
        result = []
        
        result.append(key_str)
        result.append("State:")
        # Print rows in reverse order (bottom to top)
        for y in range(self.height - 1, -1, -1):
            if print_axes:
                result.append(f"{y:2d} {' '.join(board[y])}")
            else:
                result.append(' '.join(board[y]))
        if print_axes:
            # Add x-axis labels at the bottom
            result.append("   " + " ".join(str(i) for i in range(self.width)))

        return "\n".join(result)