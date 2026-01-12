import json
import numpy as np
import time
import os
import argparse

def load_trajectory(file_path):
    """Load trajectory data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def determine_powerup_types(trajectory_file, num_channels):
    """Determine which channels represent bananas and fires based on filename or channel count."""
    if "banana" in trajectory_file.lower():
        return 3, None  # Channel 3 is bananas, no fires
    elif "fire" in trajectory_file.lower():
        return None, 3  # Channel 3 is fires, no bananas
    elif num_channels >= 5:
        return 3, 4  # Channel 3 is bananas, Channel 4 is fires
    elif num_channels == 4:
        return 3, None  # Default to bananas if no indication
    return None, None

def visualize_state(state, banana_channel=None, fire_channel=None):
    """Convert a state into a visual representation."""
    # Create an empty grid with background tiles
    grid = [['⬜' for _ in range(10)] for _ in range(10)]
    
    # Add snake body (channel 0)
    for i in range(10):
        for j in range(10):
            if state[0][i][j] == 1:
                grid[i][j] = '🟩'  # Snake body
            elif state[0][i][j] == 2:
                grid[i][j] = '🟢'  # Snake head
    
    # Add enemy snake (channel 1)
    for i in range(10):
        for j in range(10):
            if state[1][i][j] == 1:
                grid[i][j] = '🟦'  # Enemy snake body
            elif state[1][i][j] == 2:
                grid[i][j] = '🔵'  # Enemy snake head
    
    # Add food (channel 2)
    for i in range(10):
        for j in range(10):
            if state[2][i][j] == 1:
                grid[i][j] = '🍎'  # Food (apple)
    
    # Add bananas if present
    if banana_channel is not None and len(state) > banana_channel:
        for i in range(10):
            for j in range(10):
                if state[banana_channel][i][j] == 1:
                    grid[i][j] = '🍌'  # Banana
    
    # Add fires if present
    if fire_channel is not None and len(state) > fire_channel:
        for i in range(10):
            for j in range(10):
                if state[fire_channel][i][j] == 1:
                    grid[i][j] = '🔥'  # Fire
    
    return grid

def display_grid(grid, has_bananas, has_fires):
    """Display the grid in the console."""
    print("\n" + "=" * 30)
    print("Legend:")
    print("🟢 - Your snake head")
    print("🟩 - Your snake body")
    print("🔵 - Enemy snake head")
    print("🟦 - Enemy snake body")
    print("🍎 - Apple")
    if has_bananas:
        print("🍌 - Banana")
    if has_fires:
        print("🔥 - Fire")
    print("⬜ - Empty space")
    print("=" * 30)
    for row in grid:
        print("| " + " ".join(row) + " |")
    print("=" * 30)

def replay_trajectory(trajectory_file, delay=0.5):
    """Replay a trajectory with a simple grid visualization."""
    # Load trajectory
    trajectory = load_trajectory(trajectory_file)
    
    # Get states and rewards
    states = trajectory['states']
    rewards = trajectory['rewards']
    actions = trajectory['actions']
    
    # Determine powerup channels
    num_channels = len(states[0]) if states else 0
    banana_channel, fire_channel = determine_powerup_types(trajectory_file, num_channels)
    
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display initial state
    print(f"Step: 0 (Initial State)")
    print("Action: None")
    print("Reward: None")
    print("Total Reward: 0")
    grid = visualize_state(states[0], banana_channel, fire_channel)
    display_grid(grid, banana_channel is not None, fire_channel is not None)
    time.sleep(delay)
    
    # Replay each state transition
    total_reward = 0
    for i, (state, reward, action) in enumerate(zip(states[1:], rewards, actions)):
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display step information
        print(f"Step: {i+1}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        total_reward += reward
        print(f"Total Reward: {total_reward}")
        
        # Display grid
        grid = visualize_state(state, banana_channel, fire_channel)
        display_grid(grid, banana_channel is not None, fire_channel is not None)
        
        # Wait before next state
        time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay a Snake game trajectory')
    parser.add_argument('trajectory_file', type=str, help='Path to the trajectory JSON file')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between steps in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file '{args.trajectory_file}' not found")
        exit(1)
        
    replay_trajectory(args.trajectory_file, delay=args.delay)
