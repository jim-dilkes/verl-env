#!/usr/bin/env python3
"""Interactive play script for BabyAI environment.

Run with:
    python -m verl.envs.environments.babyai_text.interactive_play
    python -m verl.envs.environments.babyai_text.interactive_play --task BabyAI-GoToObj-v0
    python -m verl.envs.environments.babyai_text.interactive_play --list-tasks
"""

import sys
import tty
import termios
import argparse

import gymnasium as gym

try:
    import minigrid
    minigrid.register_minigrid_envs()
except ImportError:
    print("Error: minigrid package not installed.")
    print("Install with: pip install minigrid")
    sys.exit(1)

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

CONTROLS = {
    'a': 'turn left',
    'd': 'turn right',
    'w': 'go forward',
    'p': 'pick up',
    'x': 'drop',
    'e': 'toggle',
    'q': 'quit',
}

# Direction symbols for agent facing
DIRECTION_SYMBOLS = {
    0: '^',  # right
    1: 'v',  # down
    2: '<',  # left (note: minigrid uses different order)
    3: '>',  # up
}


def getch():
    """Get single character input without needing Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive BabyAI Game')
    parser.add_argument('--task', type=str, default='BabyAI-GoToLocal-v0',
                        help='Task name (default: BabyAI-GoToLocal-v0)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override max steps (default: use env default)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--list-tasks', action='store_true',
                        help='List available BabyAI tasks and exit')
    parser.add_argument('--tile-size', type=int, default=1,
                        help='Tile size for ASCII rendering (default: 1)')
    parser.add_argument('--no-highlight', action='store_true',
                        help='Disable ANSI color highlighting')
    return parser.parse_args()


def list_tasks():
    """List all available BabyAI tasks."""
    print("Available BabyAI tasks:")
    tasks = []
    for env_spec in gym.envs.registry:
        if env_spec.startswith("BabyAI-"):
            tasks.append(env_spec)
    for task in sorted(tasks):
        print(f"  - {task}")
    print(f"\nTotal: {len(tasks)} tasks")


def render_grid(env, use_color=True):
    """Render the environment grid as ASCII art."""
    grid = env.unwrapped.grid
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir

    # Direction vectors: 0=right, 1=down, 2=left, 3=up
    dir_symbols = ['\u2192', '\u2193', '\u2190', '\u2191']  # →↓←↑

    # ANSI colors
    if use_color:
        RESET = '\033[0m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        GRAY = '\033[90m'
        BOLD = '\033[1m'
    else:
        RESET = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = GRAY = BOLD = ''

    lines = []
    width, height = grid.width, grid.height

    # Top border
    lines.append(GRAY + '+' + '-' * (width * 2 + 1) + '+' + RESET)

    for j in range(height):
        row = GRAY + '| ' + RESET
        for i in range(width):
            if (i, j) == tuple(agent_pos):
                # Agent position
                symbol = dir_symbols[agent_dir]
                row += BOLD + GREEN + symbol + RESET + ' '
            else:
                cell = grid.get(i, j)
                if cell is None:
                    row += '  '  # Empty floor
                else:
                    obj_type = cell.type
                    color = getattr(cell, 'color', 'grey')

                    # Color mapping
                    color_code = {
                        'red': RED,
                        'green': GREEN,
                        'blue': BLUE,
                        'purple': MAGENTA,
                        'yellow': YELLOW,
                        'grey': GRAY,
                        'gray': GRAY,
                    }.get(color, '')

                    # Object symbols
                    if obj_type == 'wall':
                        symbol = GRAY + '\u2588\u2588' + RESET  # █
                    elif obj_type == 'door':
                        if cell.is_open:
                            symbol = color_code + '\u2591\u2591' + RESET  # ░ open
                        elif cell.is_locked:
                            symbol = color_code + '\u2593\u2593' + RESET  # ▓ locked
                        else:
                            symbol = color_code + '\u2592\u2592' + RESET  # ▒ closed
                    elif obj_type == 'key':
                        symbol = color_code + '\u26BF ' + RESET  # ⚿
                    elif obj_type == 'ball':
                        symbol = color_code + '\u25CF ' + RESET  # ●
                    elif obj_type == 'box':
                        symbol = color_code + '\u25A0 ' + RESET  # ■
                    elif obj_type == 'goal':
                        symbol = YELLOW + '\u2605 ' + RESET  # ★
                    elif obj_type == 'lava':
                        symbol = RED + '\u2591\u2591' + RESET  # ░
                    else:
                        symbol = '? '
                    row += symbol
        row += GRAY + '|' + RESET
        lines.append(row)

    # Bottom border
    lines.append(GRAY + '+' + '-' * (width * 2 + 1) + '+' + RESET)

    return '\n'.join(lines)


def play_game(args):
    print(f"Initializing BabyAI ({args.task})...")

    try:
        env = gym.make(args.task)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nUse --list-tasks to see available tasks.")
        return

    # Override max steps if specified
    if args.max_steps is not None:
        env.unwrapped.max_steps = args.max_steps

    print(f"\nGame settings:")
    print(f"  Task: {args.task}")
    print(f"  Max steps: {env.unwrapped.max_steps}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")

    print("\nControls:")
    print("  W=Go Forward, A=Turn Left, D=Turn Right")
    print("  E=Toggle (open/interact), P=Pick Up, X=Drop")
    print("  Q=Quit")

    seed = args.seed
    obs, info = env.reset(seed=seed)
    mission = obs.get("mission", "Unknown mission")
    done = False
    total_reward = 0
    step_count = 0
    use_color = not args.no_highlight

    print("\n" + "=" * 50)
    print(f"MISSION: {mission}")
    print("=" * 50)

    while not done:
        # Render the grid
        print("\n" + render_grid(env, use_color=use_color))

        # Show text description if available
        if "descriptions" in info:
            print("\nYou see:")
            for desc in info["descriptions"]:
                print(f"  - {desc}")

        print(f"\nStep: {step_count}/{env.unwrapped.max_steps} | Reward: {total_reward:.2f}")
        print("Move (w/a/d/e/p/x) or q to quit: ", end='', flush=True)

        action = None
        while action is None:
            try:
                char = getch().lower()
            except Exception as e:
                print(f"\nError reading input: {e}. Exiting.")
                return

            if char in CONTROLS:
                move = CONTROLS[char]
                if move == 'quit':
                    print("\nQuitting game.")
                    return
                else:
                    action = move
                    print(char)
            elif char == '\x03':  # Ctrl+C
                print("\nInterrupted.")
                return
            else:
                print(f"\nInvalid input '{repr(char)}'. Use w/a/d/e/p/x or q.")
                print("Move: ", end='', flush=True)

        # Convert action to int for raw env
        action_int = BABYAI_ACTION_SPACE.index(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        if reward > 0:
            print(f"\n{'='*20} SUCCESS! Reward: +{reward:.2f} {'='*20}")
        elif reward < 0:
            print(f"\nPenalty: {reward:.2f}")

    print("\n" + "=" * 50)
    if total_reward > 0:
        print(" MISSION COMPLETE! ".center(50, "="))
    else:
        print(" GAME OVER ".center(50, "="))
    print("=" * 50)
    print(f"\nFinal state:")
    print(render_grid(env, use_color=use_color))
    print(f"\nMission: {mission}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps: {step_count}/{env.unwrapped.max_steps}")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tasks:
        list_tasks()
    else:
        play_game(args)
