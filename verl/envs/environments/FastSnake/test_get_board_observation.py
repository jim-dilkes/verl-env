import numpy as np
import pytest
import sys
import os

# Add the parent directory to the Python path so we can import the core module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .src.core import FastSnake

def verify_channel(obs, channel_idx, expected_positions):
    """Helper function to verify a channel contains exactly the expected positions."""
    # Check that all expected positions are 1
    for pos in expected_positions:
        assert obs[channel_idx, pos[1], pos[0]] == 1
    
    # Check that no other positions are 1
    channel = obs[channel_idx]
    total_ones = np.sum(channel)
    assert total_ones == len(expected_positions)

def test_basic_functionality():
    """Test basic properties of the observation array."""
    game = FastSnake(width=5, height=5, num_apples=0, num_bananas=0, num_fires=0)  # No initial items
    game.add_snake("snake1")
    
    obs = game._get_board_observation("snake1")
    
    # Test shape
    assert obs.shape == (5, 5, 5)  # 5 channels, 5x5 board
    
    # Test binary values
    assert np.all((obs == 0) | (obs == 1))
    
    # Test that only the snake's head and body are marked
    assert np.sum(obs[0]) == 1  # Only head should be marked
    assert np.sum(obs[1]) == 1  # Only body should be marked
    assert np.all(obs[2:] == 0)  # No food or fires

def test_snake_head_channel():
    """Test Channel 0 (current snake's head)."""
    game = FastSnake(width=5, height=5, num_apples=0, num_bananas=0, num_fires=0)  # No initial items
    game.add_snake("snake1")
    game.add_snake("snake2")
    
    # Get initial positions
    snake1_head = game.snakes["snake1"]["positions"][0]
    snake2_head = game.snakes["snake2"]["positions"][0]
    
    # Test snake1's observation
    obs = game._get_board_observation("snake1")
    verify_channel(obs, 0, [snake1_head])
    
    # Test snake2's observation
    obs = game._get_board_observation("snake2")
    verify_channel(obs, 0, [snake2_head])
    
    # Test dead snake
    game.snakes["snake1"]["alive"] = False
    obs = game._get_board_observation("snake1")
    verify_channel(obs, 0, [])

def test_all_snake_bodies_channel():
    """Test Channel 1 (all snake bodies)."""
    game = FastSnake(width=5, height=5, num_apples=0, num_bananas=0, num_fires=0)  # No initial items
    game.add_snake("snake1")
    game.add_snake("snake2")
    
    # Get all positions
    all_positions = []
    for snake in game.snakes.values():
        all_positions.extend(snake["positions"])
    
    # Test observation
    obs = game._get_board_observation("snake1")
    verify_channel(obs, 1, all_positions)
    
    # Test with dead snake
    game.snakes["snake1"]["alive"] = False
    obs = game._get_board_observation("snake1")
    verify_channel(obs, 1, game.snakes["snake2"]["positions"])

def test_food_and_fire_channels():
    """Test Channels 2-4 (apples, bananas, fires)."""
    game = FastSnake(width=5, height=5, num_apples=2, num_bananas=2, num_fires=2)
    game.add_snake("snake1")
    
    # Get positions
    apples = game.apples
    bananas = game.bananas
    fires = game.fires
    
    # Test observation
    obs = game._get_board_observation("snake1")
    
    # Verify each channel
    verify_channel(obs, 2, apples)
    verify_channel(obs, 3, bananas)
    verify_channel(obs, 4, fires)

def test_edge_cases():
    """Test various edge cases."""
    game = FastSnake(width=5, height=5, num_apples=0, num_bananas=0, num_fires=0)  # No initial items
    game.add_snake("snake1")
    
    # Test with no food or fires
    obs = game._get_board_observation("snake1")
    assert np.all(obs[2:] == 0)  # All food/fire channels should be zero
    
    # Test with snake at boundary
    game.snakes["snake1"]["positions"] = [(0, 0)]  # Bottom-left corner
    obs = game._get_board_observation("snake1")
    assert obs[0, 0, 0] == 1  # Head should be marked
    
    # Test with invalid snake_id
    with pytest.raises(KeyError):
        game._get_board_observation("nonexistent_snake")

def test_comprehensive():
    """Test a comprehensive scenario with multiple snakes and items."""
    game = FastSnake(width=10, height=10, num_apples=3, num_bananas=2, num_fires=2)
    game.add_snake("snake1")
    game.add_snake("snake2")
    game.add_snake("snake3")
    
    # Kill one snake
    game.snakes["snake3"]["alive"] = False
    
    # Test observation for each snake
    for snake_id in ["snake1", "snake2"]:
        obs = game._get_board_observation(snake_id)
        
        # Verify shape
        assert obs.shape == (5, 10, 10)
        
        # Verify binary values
        assert np.all((obs == 0) | (obs == 1))
        
        # Verify current snake's head
        head_pos = game.snakes[snake_id]["positions"][0]
        assert obs[0, head_pos[1], head_pos[0]] == 1
        
        # Verify all snake bodies
        all_positions = []
        for snake in game.snakes.values():
            if snake["alive"]:
                all_positions.extend(snake["positions"])
        verify_channel(obs, 1, all_positions)
        
        # Verify food and fires
        verify_channel(obs, 2, game.apples)
        verify_channel(obs, 3, game.bananas)
        verify_channel(obs, 4, game.fires)

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
