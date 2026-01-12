"""
Test script for rolling apples functionality in the FastSnake game.
"""

from src.env import FastSnakeEnv
from src.core import UP, DOWN, LEFT, RIGHT

def test_rolling_apples():
    """Test that apples roll down the hill in the specified direction."""
    
    # Test with hill_direction="right"
    print("Testing with hill_direction='right'")
    env = FastSnakeEnv(
        width=10,
        height=10,
        num_apples=5,
        apple_reward=1,
        num_bananas=2,
        banana_reward=5,
        num_fires=0,  # No fires when using hill direction
        hill_direction="right"
    )
    
    # Reset the environment to initialize the game
    obs, info = env.reset(seed=42)
    
    # Print initial state
    print("Initial state:")
    print(env.game_state_text())
    
    # Step the environment a few times
    for i in range(5):
        action = UP  # Move upward
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nAfter step {i+1}:")
        print(f"Reward: {reward}")
        print(env.game_state_text())
        
        if done:
            print("Game over!")
            break
    
    # Test with hill_direction="left"
    print("\n\nTesting with hill_direction='left'")
    env = FastSnakeEnv(
        width=10,
        height=10,
        num_apples=5,
        apple_reward=1,
        num_bananas=2,
        banana_reward=5,
        num_fires=0,
        hill_direction="left"
    )
    
    # Reset the environment
    obs, info = env.reset(seed=42)
    
    # Print initial state
    print("Initial state:")
    print(env.game_state_text())
    
    # Step the environment a few times
    for i in range(5):
        action = UP  # Move upward
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nAfter step {i+1}:")
        print(f"Reward: {reward}")
        print(env.game_state_text())
        
        if done:
            print("Game over!")
            break
    
    # Test incompatibility with fires
    print("\n\nTesting incompatibility with fires")
    try:
        env = FastSnakeEnv(
            width=10,
            height=10,
            num_apples=5,
            num_fires=2,
            hill_direction="right"
        )
        print("Failed: Environment was created despite incompatible settings")
    except ValueError as e:
        print(f"Success: Caught expected ValueError: {e}")

if __name__ == "__main__":
    test_rolling_apples() 