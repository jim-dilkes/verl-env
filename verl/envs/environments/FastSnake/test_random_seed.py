#!/usr/bin/env python3

"""
Test script to verify the deterministic behavior of FastSnakeEnv with seeding.
This test creates identical environments with the same seed and checks if
they produce identical states when running with random snakes.
"""

import numpy as np
from src.env import FastSnakeEnv
from src.core import UP, DOWN, LEFT, RIGHT

def print_board(env, env_id=""):
    """Print a text representation of the board state."""
    print(f"Environment {env_id} state:")
    print(env.game.render_text())
    print("\n")

def are_states_identical(env1, env2):
    """Compare two environments to see if they have identical states."""
    # Compare board states
    board1 = env1.game.board
    board2 = env2.game.board
    boards_match = np.array_equal(board1, board2)
    
    # Compare snake positions
    snakes_match = True
    for snake_id in env1.game.snakes:
        if snake_id not in env2.game.snakes:
            snakes_match = False
            break
        # Compare each snake's position lists
        snake1_pos = list(env1.game.snakes[snake_id]['positions'])
        snake2_pos = list(env2.game.snakes[snake_id]['positions'])
        if snake1_pos != snake2_pos:
            snakes_match = False
            break
    
    # Compare apple positions
    apples_match = sorted(env1.game.apples) == sorted(env2.game.apples)
    
    # Compare scores
    scores_match = env1.game.scores == env2.game.scores
    
    return boards_match and snakes_match and apples_match and scores_match

def test_random_snakes_only():
    """Test with only random snakes to verify basic determinism."""
    # Create two environments with the same seed
    SEED = 0
    NUM_STEPS = 10
    
    print("========== TEST 1: RANDOM SNAKES ONLY ==========")
    print(f"Creating two environments, then resetting them with seed {SEED}")
    
    # Two random snakes, no external snakes
    env1 = FastSnakeEnv(width=10, height=10, num_external_snakes=0, 
                       num_random_snakes=2)
    env2 = FastSnakeEnv(width=10, height=10, num_external_snakes=0, 
                       num_random_snakes=2)
    
    # Reset both environments
    obs1, _ = env1.reset(seed=SEED)
    obs2, _ = env2.reset(seed=SEED)
    
    print("Initial states:")
    print_board(env1, "1")
    print_board(env2, "2")
    
    # Verify initial states match
    if are_states_identical(env1, env2):
        print("✅ Initial states are identical")
    else:
        print("❌ Initial states are different!")
    
    # Run both environments for NUM_STEPS steps
    print(f"\nRunning for {NUM_STEPS} steps...")
    
    done1 = done2 = False
    
    for step in range(NUM_STEPS):
        print(f"\nStep {step+1}:")
        
        # Skip if either environment is done
        if done1 or done2:
            print("One or both environments are done, stopping simulation")
            break
            
        # No actions needed since we're using random snakes only
        obs1, reward1, done1, _, info1 = env1.step({})
        obs2, reward2, done2, _, info2 = env2.step({})
        
        print_board(env1, "1")
        print_board(env2, "2")
        
        # Verify states match after each step
        if are_states_identical(env1, env2):
            print(f"✅ States after step {step+1} are identical")
        else:
            print(f"❌ States after step {step+1} are different!")
            
            # Print detailed differences for debugging
            print("Differences:")
            if not np.array_equal(env1.game.board, env2.game.board):
                print("- Board states differ")
            
            for snake_id in env1.game.snakes:
                if snake_id in env2.game.snakes:
                    snake1_pos = list(env1.game.snakes[snake_id]['positions'])
                    snake2_pos = list(env2.game.snakes[snake_id]['positions'])
                    if snake1_pos != snake2_pos:
                        print(f"- Snake {snake_id} positions differ:")
                        print(f"  Env1: {snake1_pos}")
                        print(f"  Env2: {snake2_pos}")
            
            if sorted(env1.game.apples) != sorted(env2.game.apples):
                print(f"- Apple positions differ:")
                print(f"  Env1: {env1.game.apples}")
                print(f"  Env2: {env2.game.apples}")
                
        # Check if both environments end at the same time
        if done1 != done2:
            print(f"❌ Environments have different 'done' states! {done1} vs {done2}")
            
        # Exit if both environments are done
        if done1 and done2:
            print("Both environments finished.")
            break
    
    print("\nFinal comparison:")
    if are_states_identical(env1, env2):
        print("✅ Final states are identical")
    else:
        print("❌ Final states are different!")

def test_with_external_snake_getting_apple():
    """
    Test with one external snake and one random snake.
    Tests if different player actions affect the determinism of random behaviors.
    """
    SEED = 0
    NUM_STEPS = 10
    
    print("\n\n========== TEST 2: EXTERNAL + RANDOM SNAKE (IDENTICAL ACTIONS) ==========")
    print("Creating two environments with the same seed, with identical player actions")
    print("This test specifically focuses on apple collection and subsequent placement")
    
    # One external snake, one random snake
    env1 = FastSnakeEnv(width=10, height=10, num_external_snakes=1, 
                       num_random_snakes=1)
    env2 = FastSnakeEnv(width=10, height=10, num_external_snakes=1, 
                       num_random_snakes=1)
    
    # Reset both environments
    obs1, _ = env1.reset(seed=SEED)
    obs2, _ = env2.reset(seed=SEED)
    
    print("Initial states:")
    print_board(env1, "1")
    print_board(env2, "2")
    
    # Pre-defined same actions for the external snake in both environments
    # Actions designed to collect an apple
    env1_actions = [LEFT, LEFT, LEFT, LEFT, DOWN, RIGHT, UP, RIGHT, UP, RIGHT]
    env2_actions = env1_actions
    
    # Verify initial states match
    if are_states_identical(env1, env2):
        print("✅ Initial states are identical")
    else:
        print("❌ Initial states are different!")
    
    # Run both environments for NUM_STEPS steps with identical player actions
    print(f"\nRunning for {NUM_STEPS} steps with identical player actions...")
    
    done1 = done2 = False
    
    for step in range(NUM_STEPS):
        print(f"\nStep {step+1}:")
        
        # Skip if either environment is done
        if done1 or done2:
            print("One or both environments are done, stopping simulation")
            break
            
        # Same actions for the external snake in each environment
        action1 = env1_actions[step % len(env1_actions)]
        action2 = env2_actions[step % len(env2_actions)]
        
        print(f"Both envs: Player action = {action1}")
        
        # Take step with the identical actions
        obs1, reward1, done1, _, info1 = env1.step(action1)
        obs2, reward2, done2, _, info2 = env2.step(action2)
        
        print_board(env1, "1")
        print_board(env2, "2")
        
        # Print done status
        print(f"Done status: Env1={done1}, Env2={done2}")
        
        # Print rewards
        print(f"Rewards: Env1={reward1}, Env2={reward2}")
        
        # Print alive status for all snakes
        print("\nSnake alive status:")
        print("Env1:")
        for s_id, snake in env1.game.snakes.items():
            print(f"  Snake {s_id}: alive={snake['alive']}, positions={list(snake['positions'])}")
        
        print("Env2:")
        for s_id, snake in env2.game.snakes.items():
            print(f"  Snake {s_id}: alive={snake['alive']}, positions={list(snake['positions'])}")
        
        # Print game over status
        print(f"\nGame over status: Env1={env1.game.game_over}, Env2={env2.game.game_over}")
        print(f"Current round: Env1={env1.game.round_number}, Env2={env2.game.round_number}")
        
        # Check for apple collection and reward
        if reward1 > 0 or reward2 > 0:
            print(f"🍎 Apple collected! Rewards: Env1={reward1}, Env2={reward2}")
        
        # Here we expect the states to be identical because of identical player actions
        print("Expected behavior: States should be identical with same player actions")
        identical = are_states_identical(env1, env2)
        if identical:
            print(f"✅ States after step {step+1} are identical (expected)")
        else:
            print(f"❌ States after step {step+1} are different (unexpected!)")
            
            # Print detailed differences for debugging
            print("Differences:")
            if not np.array_equal(env1.game.board, env2.game.board):
                print("- Board states differ")
            
            for snake_id in env1.game.snakes:
                if snake_id in env2.game.snakes:
                    snake1_pos = list(env1.game.snakes[snake_id]['positions'])
                    snake2_pos = list(env2.game.snakes[snake_id]['positions'])
                    if snake1_pos != snake2_pos:
                        print(f"- Snake {snake_id} positions differ:")
                        print(f"  Env1: {snake1_pos}")
                        print(f"  Env2: {snake2_pos}")
            
            if sorted(env1.game.apples) != sorted(env2.game.apples):
                print(f"- Apple positions differ:")
                print(f"  Env1: {env1.game.apples}")
                print(f"  Env2: {env2.game.apples}")
            
        # Extract random snake IDs (they should be the same in both envs)
        random_snake_id1 = env1.random_snake_ids[0]
        random_snake_id2 = env2.random_snake_ids[0]
        
        # Print positions of the random snakes
        if random_snake_id1 in env1.game.snakes and random_snake_id2 in env2.game.snakes:
            pos1 = list(env1.game.snakes[random_snake_id1]['positions'])
            pos2 = list(env2.game.snakes[random_snake_id2]['positions'])
            print(f"Random snake in Env1: {pos1}")
            print(f"Random snake in Env2: {pos2}")
            
        # Print apple positions
        print(f"Apples in Env1: {env1.game.apples}")
        print(f"Apples in Env2: {env2.game.apples}")
            
        # Check if both environments end at the same time
        if done1 != done2:
            print(f"❌ Environments have different 'done' states! {done1} vs {done2}")
            
        # Exit if both environments are done
        if done1 and done2:
            print("Both environments finished.")
            break
    
    print("\nTest complete!")
    print("This test verifies that with identical actions, the environments remain in sync")

def test_with_external_snake_different_actions():
    """
    Test with one external snake and one random snake.
    Tests if different player actions affect the determinism of random behaviors.
    """
    SEED = 0
    NUM_STEPS = 10
    
    print("\n\n========== TEST 3: EXTERNAL + RANDOM SNAKE (DIFFERENT ACTIONS) ==========")
    print("Creating two environments with the same seed, but with different player actions")
    print("This test examines how random snake behavior is affected by player actions")
    
    # One external snake, one random snake
    env1 = FastSnakeEnv(width=10, height=10, num_external_snakes=1, 
                       num_random_snakes=1)
    env2 = FastSnakeEnv(width=10, height=10, num_external_snakes=1, 
                       num_random_snakes=1)
    
    # Reset both environments
    obs1, _ = env1.reset(seed=SEED)
    obs2, _ = env2.reset(seed=SEED)
    
    print("Initial states:")
    print_board(env1, "1")
    print_board(env2, "2")
    
    # Pre-defined different actions for the external snake in each environment
    # Env1: Player goes LEFT, DOWN, LEFT, LEFT, etc.
    # Env2: Player goes DOWN, DOWN, LEFT, DOWN, etc.
    env1_actions = [LEFT, DOWN, LEFT, LEFT, LEFT, DOWN, LEFT, DOWN, LEFT, DOWN]
    env2_actions = [DOWN, DOWN, LEFT, DOWN, RIGHT, DOWN, LEFT, DOWN, RIGHT, UP]
    
    # Verify initial states match
    if are_states_identical(env1, env2):
        print("✅ Initial states are identical")
    else:
        print("❌ Initial states are different!")
    
    # Run both environments for NUM_STEPS steps with different player actions
    print(f"\nRunning for {NUM_STEPS} steps with different player actions...")
    
    done1 = done2 = False
    
    for step in range(NUM_STEPS):
        print(f"\nStep {step+1}:")
        
        # Skip if either environment is done
        if done1 or done2:
            print("One or both environments are done, stopping simulation")
            break
            
        # Different actions for the external snake in each environment
        action1 = env1_actions[step % len(env1_actions)]
        action2 = env2_actions[step % len(env2_actions)]
        
        print(f"Env1: Player action = {action1}")
        print(f"Env2: Player action = {action2}")
        
        # Take step with the different actions
        obs1, reward1, done1, _, info1 = env1.step(action1)
        obs2, reward2, done2, _, info2 = env2.step(action2)
        
        print_board(env1, "1")
        print_board(env2, "2")
        
        # Print done status
        print(f"Done status: Env1={done1}, Env2={done2}")
        
        # Print rewards
        print(f"Rewards: Env1={reward1}, Env2={reward2}")
        
        # Print alive status for all snakes
        print("\nSnake alive status:")
        print("Env1:")
        for s_id, snake in env1.game.snakes.items():
            print(f"  Snake {s_id}: alive={snake['alive']}, positions={list(snake['positions'])}")
        
        print("Env2:")
        for s_id, snake in env2.game.snakes.items():
            print(f"  Snake {s_id}: alive={snake['alive']}, positions={list(snake['positions'])}")
        
        # Print game over status
        print(f"\nGame over status: Env1={env1.game.game_over}, Env2={env2.game.game_over}")
        print(f"Current round: Env1={env1.game.round_number}, Env2={env2.game.round_number}")
        
        # Here we expect the states to be different because of different player actions
        print("Expected behavior: States should be different due to different player actions")
        identical = are_states_identical(env1, env2)
        if identical:
            print(f"❓ States after step {step+1} are identical (unexpected!)")
        else:
            print(f"✅ States after step {step+1} are different (expected)")
            
        # Extract random snake IDs (they should be the same in both envs)
        random_snake_id1 = env1.random_snake_ids[0]
        random_snake_id2 = env2.random_snake_ids[0]
        
        # Print positions of the random snakes
        if random_snake_id1 in env1.game.snakes and random_snake_id2 in env2.game.snakes:
            pos1 = list(env1.game.snakes[random_snake_id1]['positions'])
            pos2 = list(env2.game.snakes[random_snake_id2]['positions'])
            print(f"Random snake in Env1: {pos1}")
            print(f"Random snake in Env2: {pos2}")
            
        # Print apple positions
        print(f"Apples in Env1: {env1.game.apples}")
        print(f"Apples in Env2: {env2.game.apples}")
            
        # Check if both environments end at the same time
        if done1 != done2:
            print(f"❌ Environments have different 'done' states! {done1} vs {done2}")
            
        # Exit if both environments are done
        if done1 and done2:
            print("Both environments finished.")
            break
    
    print("\nTest complete!")
    print("This test shows how different player actions affect the overall state")
    print("Note: While we expect states to differ, we're interested in whether")
    print("      random snake behaviors are still deterministic given their local state")

def main():
    # Run both test cases
    test_random_snakes_only()
    test_with_external_snake_getting_apple()
    test_with_external_snake_different_actions()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()



