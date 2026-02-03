#!/usr/bin/env python3
"""Interactive play script for Overcooked environment."""

import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import sys
import tty
import termios
import argparse

from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
from verl.envs.environments.overcooked import ACTION_TO_IDX, IDX_TO_ACTION
from verl.envs.environments.overcooked.custom_layouts import CUSTOM_LAYOUTS

CONTROLS = {
    'w': 'up',
    's': 'down',
    'a': 'left',
    'd': 'right',
    ' ': 'stay',
    'e': 'interact',
    'q': 'quit',
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
    parser = argparse.ArgumentParser(description='Interactive Overcooked Game')
    parser.add_argument('--layout', type=str, default='cramped_room',
                        help='Layout name (default: cramped_room)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode (default: 200)')
    parser.add_argument('--partner', type=str, default='noop',
                        choices=['noop', 'random', 'none'],
                        help='Partner policy: noop, random, or none for solo (default: noop)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--cook-time', type=int, default=None,
                        help='Override cooking time in ticks (default: 20)')
    parser.add_argument('--list-layouts', action='store_true',
                        help='List available layouts and exit')
    parser.add_argument('--no-grid', action='store_true',
                        help='Disable ASCII grid visualization')
    parser.add_argument('--no-coords', action='store_true',
                        help='Disable coordinate text')
    return parser.parse_args()


def list_layouts():
    from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts

    print("Built-in JaxMARL layouts:")
    for name in sorted(overcooked_v2_layouts.keys()):
        print(f"  - {name}")

    print("\nCustom layouts:")
    for name in sorted(CUSTOM_LAYOUTS.keys()):
        layout = CUSTOM_LAYOUTS[name]
        recipes = layout.possible_recipes
        print(f"  - {name} (recipe: {recipes[0] if recipes else 'random'})")


def play_game(args):
    print_viz = not args.no_grid
    print_coords = not args.no_coords

    if not print_viz and not print_coords:
        print("Error: Cannot disable both --no-grid and --no-coords")
        return

    # Check if this is a custom layout
    if args.layout in CUSTOM_LAYOUTS:
        layout = CUSTOM_LAYOUTS[args.layout]
        print(f"Initializing Overcooked (custom: {args.layout})...")
        print(f"Recipe: {layout.possible_recipes[0]}")
    else:
        layout = args.layout
        print(f"Initializing Overcooked ({args.layout})...")

    env = OvercookedGymWrapper(
        layout=layout,
        max_steps=args.max_steps,
        partner_policy=args.partner,
        seed=args.seed,
        shaped_reward=True,
        print_visualization=print_viz,
        print_coordinates=print_coords,
        pot_cook_time=args.cook_time,
    )

    print(f"\nGame settings:")
    print(f"  Layout: {args.layout}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Partner: {args.partner}" + (" (solo mode)" if args.partner == "none" else ""))
    print(f"  Cook time: {env.pot_cook_time} ticks")
    print(f"  Seed: {args.seed}")

    print("\nControls:")
    print("  W=Up, S=Down, A=Left, D=Right")
    print("  E=Interact (pick up/place/use)")
    print("  Space=Stay, Q=Quit")

    obs, info = env.reset()
    done = False
    total_reward = 0

    print("\n" + "=" * 50)
    print("Starting game! Goal: Cook and deliver soups.")
    print("=" * 50)

    while not done:
        print("\n" + env.render())
        print(f"\nTotal reward: {total_reward:.2f}")
        print("Enter move (w/a/s/d/e/space) or q to quit: ", end='', flush=True)

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
                    print(char if char != ' ' else '[space]')
            elif char == '\x03':  # Ctrl+C
                print("\nInterrupted.")
                return
            else:
                print(f"\nInvalid input '{repr(char)}'. Use w/a/s/d/e/space or q.")
                print("Enter move: ", end='', flush=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if reward != 0:
            print(f"Reward: {reward:+.2f}")

    print("\n" + "=" * 50)
    print(" GAME OVER! ".center(50, "="))
    print("=" * 50)
    print(f"\nFinal state:")
    print(env.render())
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Steps: {info.get('step', 'N/A')}/{args.max_steps}")


if __name__ == "__main__":
    args = parse_args()

    if args.list_layouts:
        list_layouts()
    else:
        play_game(args)
