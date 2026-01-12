# interactive_play.py

import sys
import tty
import termios
import argparse
# Adjusted imports for running from modules/SnakeBench directory
from src.env import FastSnakeEnv
from src.core import UP, DOWN, LEFT, RIGHT # Import constants

# --- Default Game Parameters ---
DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 10
DEFAULT_NUM_APPLES = 5
DEFAULT_MAX_ROUNDS = 200
DEFAULT_NUM_RANDOM_SNAKES = 1
DEFAULT_NUM_BANANAS = 0
DEFAULT_BANANA_REWARD = 0
DEFAULT_NUM_FIRES = 0
DEFAULT_FIRE_REWARD = 0
DEFAULT_APPLE_REWARD = 1
DEFAULT_HILL_DIRECTION = None
DEFAULT_USE_COLOR = True  # Enable colored mode by default for interactive play
# ------------------------------

# --- Action Mapping ---
# Map keyboard input (lowercase) to game actions
input_to_action = {
    'w': UP,    # Use W for UP
    's': DOWN,  # Use S for DOWN
    'a': LEFT,  # Use A for LEFT
    'd': RIGHT, # Use D for RIGHT
    'q': 'quit' # Add a quit option
}
# --------------------

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Snake Game')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help=f'Board width (default: {DEFAULT_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help=f'Board height (default: {DEFAULT_HEIGHT})')
    parser.add_argument('--apples', type=int, default=DEFAULT_NUM_APPLES, help=f'Number of apples (default: {DEFAULT_NUM_APPLES})')
    parser.add_argument('--apple-reward', type=int, default=DEFAULT_APPLE_REWARD, help=f'Points for eating an apple (default: {DEFAULT_APPLE_REWARD})')
    parser.add_argument('--rounds', type=int, default=DEFAULT_MAX_ROUNDS, help=f'Maximum rounds (default: {DEFAULT_MAX_ROUNDS})')
    parser.add_argument('--random-snakes', type=int, default=DEFAULT_NUM_RANDOM_SNAKES, help=f'Number of random snakes (default: {DEFAULT_NUM_RANDOM_SNAKES})')
    parser.add_argument('--bananas', type=int, default=DEFAULT_NUM_BANANAS, help=f'Number of bananas (default: {DEFAULT_NUM_BANANAS})')
    parser.add_argument('--banana-reward', type=int, default=DEFAULT_BANANA_REWARD, help=f'Points for eating a banana (default: {DEFAULT_BANANA_REWARD})')
    parser.add_argument('--fires', type=int, default=DEFAULT_NUM_FIRES, help=f'Number of fires (default: {DEFAULT_NUM_FIRES})')
    parser.add_argument('--fire-reward', type=int, default=DEFAULT_FIRE_REWARD, help=f'Points for hitting fire (default: {DEFAULT_FIRE_REWARD}, negative means penalty)')
    parser.add_argument('--hill-direction', type=str, choices=['up', 'down', 'left', 'right', 'none'], default=None, 
                        help='Direction for apples to roll (up, down, left, right, none)')
    parser.add_argument('--destroy-at-bottom', type=bool, default=False, help='Destroy apples at bottom of board (default: False)')
    parser.add_argument('--no-color', action='store_true', help='Disable colored visualization')
    parser.add_argument('--no-respawn', action='store_true', help='Disable respawn of objects (default: False)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for environment initialization (default: None, uses random seed)')
    return parser.parse_args()

# Helper function to get single character input without needing Enter
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def play_game(args):
    # Process hill direction argument
    hill_direction = None if args.hill_direction == 'none' else args.hill_direction
    
    # Check compatibility of hill direction and fires
    if hill_direction is not None and args.fires > 0:
        print("Warning: Hill direction is not compatible with fires.")
        print("Would you like to:")
        print("1. Disable fires")
        print("2. Disable hill direction")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ")
        if choice == '1':
            args.fires = 0
            print("Fires disabled.")
        elif choice == '2':
            hill_direction = None
            print("Hill direction disabled.")
        else:
            print("Exiting game.")
            return
    
    print("Initializing FastSnakeEnv for interactive play...")
    env = FastSnakeEnv(
        width=args.width,
        height=args.height,
        num_apples=args.apples,
        max_rounds=args.rounds,
        num_external_snakes=1, # Must be 1 for this script
        num_random_snakes=args.random_snakes,
        num_bananas=args.bananas,
        banana_reward=args.banana_reward,
        num_fires=args.fires,
        fire_reward=args.fire_reward,
        apple_reward=args.apple_reward,
        hill_direction=hill_direction,
        destroy_at_bottom=args.destroy_at_bottom,
        print_visualization=True,
        print_coordinates=True,
        print_axes=False,
        no_respawn=args.no_respawn
    )

    # Store the color mode preference
    use_color = not args.no_color

    print("Game settings:")
    print(f"  Board: {args.width}x{args.height}")
    print(f"  Apples: {args.apples} ({args.apple_reward} point{'' if args.apple_reward == 1 else 's'} each)")
    print(f"  Bananas: {args.bananas} ({args.banana_reward} points each)")
    print(f"  Fires: {args.fires} ({args.fire_reward} points each)")
    print(f"  Random snakes: {args.random_snakes}")
    print(f"  Hill direction: {hill_direction if hill_direction else 'None (apples stay in place)'}")
    print(f"  Max rounds: {args.rounds}")
    print(f"  Colored visualization: {'enabled' if use_color else 'disabled'}")
    print(f"  Seed: {args.seed if args.seed is not None else 'None (random)'}")

    print("Resetting environment...")
    obs, info = env.reset(seed=args.seed) # Get initial state
    your_snake_id = env.external_snake_ids[0]
    done = False
    total_reward = 0

    print("\nStarting game!")
    print("Controls: W=Up, S=Down, A=Left, D=Right, Q=Quit")

    while not done:
        # 1. Render state
        print("\n" + "="*40)
        try:
            # Override the game state text to use color mode
            if use_color:
                print(env.game.render_text(print_axes=env.print_axes, use_color_mode=True))
                print(f"\nRound: {info.get('round', 0)}")
                print(f"Your Score ({your_snake_id}): {info.get('scores', {}).get(your_snake_id, 0)}")
                print(f"Total Reward: {total_reward:.2f}")
            else:
                print(env.game_state_text())
                current_score = info.get('scores', {}).get(your_snake_id, 0)
                print(f"Round: {info.get('round', 0)}")
                print(f"Your Score ({your_snake_id}): {current_score}")
                print(f"Total Reward: {total_reward:.2f}")
            print("Enter move (w/a/s/d) or q to quit: ", end='', flush=True)
        except Exception as e:
            print(f"Error rendering state: {e}")
            break # Exit if rendering fails

        # 2. Get user input
        action = None
        while action is None:
            try:
                char = getch().lower()
            except Exception as e:
                 print(f"\nError reading input: {e}. Exiting.")
                 return # Exit if input reading fails

            if char in input_to_action:
                move = input_to_action[char]
                if move == 'quit':
                    print("\nQuitting game.")
                    return
                else:
                    action = move
                    print(char) # Echo valid move
            else:
                print(f"\nInvalid input '{char}'. Use w/a/s/d or q.", end='', flush=True)
                # Reprint prompt after handling invalid input without newline
                print("\nEnter move (w/a/s/d) or q to quit: ", end='', flush=True)


        # 3. Step environment
        try:
            # Since num_external_snakes is 1, step expects a single action integer
            # obs, reward, done, _, info = env.step(action) # Standard Gym returns 5 values
            # FastSnakeEnv step returns 5 values: obs, reward, done, False, info
            obs, reward, game_finished, truncated, info = env.step(action)
            done = game_finished # Use the correct done flag

            total_reward += reward
            print(f"Action taken. Reward this step: {reward:.2f}")
            print(f"Success: {info['success']}")
        except Exception as e:
            print(f"\nError during environment step: {e}")
            import traceback
            traceback.print_exc()
            break # Exit if step fails

    # 4. Game over
    print("\n" + "="*40)
    print(" GAME OVER! ".center(40, "="))
    print("="*40)
    try:
        # Print final state
        print("Final State:")
        print(env.game_state_text())
        final_scores = info.get('all_scores', {})
        print("\nFinal Scores:")
        for snake_id, score in final_scores.items():
             print(f"  - {snake_id}: {score}")
        print(f"\nTotal reward for your snake ({your_snake_id}): {total_reward:.2f}")
        print(f"Game lasted {info.get('round', 'N/A')} rounds.")
    except Exception as e:
        print(f"Error printing final state: {e}")

if __name__ == "__main__":
    args = parse_args()
    play_game(args)
