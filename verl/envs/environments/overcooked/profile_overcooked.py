#!/usr/bin/env python3
"""Profile Overcooked environment to identify speed bottlenecks."""

import time
import cProfile
import pstats
from io import StringIO
import numpy as np

from verl.envs.environments.overcooked.jaxmarl_wrapper import OvercookedGymWrapper
from verl.envs.environments.overcooked.base import OvercookedLLMAgentsWrapper


def time_function(func, *args, n_calls=100, **kwargs):
    """Time a function over multiple calls."""
    times = []
    for _ in range(n_calls):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000, result


def profile_overcooked(n_steps=500, layout="cramped_room"):
    """Profile Overcooked environment step performance."""

    print(f"Profiling Overcooked ({layout}) for {n_steps} steps...\n")

    # Create base gym env
    base_env = OvercookedGymWrapper(
        layout=layout,
        max_steps=200,
        partner_policy="noop",
        shaped_reward=True,
        print_visualization=True,
        print_coordinates=True,
    )

    # Create LLM wrapper
    env = OvercookedLLMAgentsWrapper(base_env)

    # Warm up JAX
    print("Warming up JAX...")
    obs, info = env.reset(seed=42)
    for _ in range(10):
        obs, reward, term, trunc, info = env.step("stay")

    # Reset for actual profiling
    obs, info = env.reset(seed=42)

    # Time individual components
    print("\n=== Component Timing (ms) ===\n")

    # Time base env step only
    def base_step():
        return base_env.step(4)  # stay
    mean, std, _ = time_function(base_step, n_calls=100)
    print(f"base_env.step():          {mean:7.3f} ± {std:.3f}")

    # Time render
    def render_only():
        return base_env.render()
    mean, std, _ = time_function(render_only, n_calls=100)
    print(f"base_env.render():        {mean:7.3f} ± {std:.3f}")

    # Time get_state_info
    def get_state_info_only():
        return base_env.get_state_info()
    mean, std, _ = time_function(get_state_info_only, n_calls=100)
    print(f"get_state_info():         {mean:7.3f} ± {std:.3f}")

    # Time _render_coordinates
    info = base_env.get_state_info()
    grid = info["grid"]
    def render_coords_only():
        return base_env._render_coordinates(info, grid)
    mean, std, _ = time_function(render_coords_only, n_calls=100)
    print(f"_render_coordinates():    {mean:7.3f} ± {std:.3f}")

    # Time _render_grid
    def render_grid_only():
        return base_env._render_grid(info, grid)
    mean, std, _ = time_function(render_grid_only, n_calls=100)
    print(f"_render_grid():           {mean:7.3f} ± {std:.3f}")

    # Time _get_pot_info
    def get_pot_info_only():
        return base_env._get_pot_info(grid)
    mean, std, _ = time_function(get_pot_info_only, n_calls=100)
    print(f"_get_pot_info():          {mean:7.3f} ± {std:.3f}")

    # Time _get_static_objects
    def get_static_objects_only():
        return base_env._get_static_objects(grid)
    mean, std, _ = time_function(get_static_objects_only, n_calls=100)
    print(f"_get_static_objects():    {mean:7.3f} ± {std:.3f}")

    # Time restructure_obs (calls render)
    def restructure_obs_only():
        return env.restructure_obs(np.zeros((5, 4, 26)))
    mean, std, _ = time_function(restructure_obs_only, n_calls=100)
    print(f"restructure_obs():        {mean:7.3f} ± {std:.3f}")

    # Time full LLM wrapper step
    def full_step():
        return env.step("stay")
    mean, std, _ = time_function(full_step, n_calls=100)
    print(f"LLMWrapper.step():        {mean:7.3f} ± {std:.3f}")

    # Time extract_action
    sample_output = "<action>stay</action>"
    def extract_action_only():
        return env.extract_action_instance(sample_output)
    mean, std, _ = time_function(extract_action_only, n_calls=100)
    print(f"extract_action():         {mean:7.3f} ± {std:.3f}")

    print("\n=== cProfile Analysis ===\n")

    # Profile full step with cProfile
    pr = cProfile.Profile()
    pr.enable()

    for _ in range(n_steps):
        obs, reward, term, trunc, info = env.step("stay")
        if term or trunc:
            obs, info = env.reset()

    pr.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Summary
    print("\n=== Summary ===")
    print("Key areas to investigate:")
    print("1. render() is called every step via restructure_obs()")
    print("2. _render_coordinates calls _get_pot_info and _get_static_objects")
    print("3. Each of those iterates over the entire grid")
    print("4. Static objects don't change but are recomputed every step!")


if __name__ == "__main__":
    profile_overcooked(n_steps=500)
