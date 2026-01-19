#!/usr/bin/env python3
"""
Test script to compare VecEnv creation times across multiprocessing methods.

Compares: spawn, fork, forkserver
Tests: fastsnake, overcooked, babyai environments

Usage:
    python scripts/test_vecenv_multiprocessing.py
    python scripts/test_vecenv_multiprocessing.py --n-workers 50 --methods spawn forkserver
    python scripts/test_vecenv_multiprocessing.py --envs fastsnake --methods fork forkserver
"""

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure verl is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf


def create_test_config(
    env_name: str,
    n_workers: int,
    mp_method: str,
    task: Optional[str] = None,
) -> OmegaConf:
    """Create a minimal config for VecEnv testing."""

    config = {
        'envs': {
            'n_rollouts': n_workers,
            'episode_length': 8,
            'env_name': env_name,
            'task': task,
            'freeze_completed_episodes': True,
            'duplication_mode': 'none',
            'format_penalty': 0.0,
            'binary_reward': False,
            'vec_env_multiprocessing': mp_method,
            'group_initial_seed': 12345,
            'captioner': {
                'type': 'naive',
                'max_text_history': 0,
                'max_image_history': 0,
                'max_cot_history': 0,
            },
        },
        'prompt': {
            'prompt': {
                'epsilon': 0.0,
                'multi_action_reasoning': False,
                'environment_instruction': 'Test instruction.',
            },
        },
    }

    # Add environment-specific kwargs
    if env_name == 'fastsnake':
        config['envs']['fastsnake_kwargs'] = {
            'width': 10,
            'height': 10,
            'max_rounds': 8,
            'num_apples': 5,
            'death_reward': -1,
            'step_reward': 0,
            'print_visualization': False,
            'print_coordinates': True,
        }
    elif env_name == 'overcooked':
        config['envs']['overcooked_kwargs'] = {
            'layout_name': 'cramped_room',
            'horizon': 15,
            'partner_policy': 'none',
            'shaped_reward': False,
            'pot_cook_time': 5,
            'print_visualization': False,
            'print_coordinates': True,
        }
    elif env_name == 'babyai':
        config['envs']['babyai_kwargs'] = {
            'max_steps': 20,
        }

    return OmegaConf.create(config)


def test_vecenv_lifecycle(
    env_name: str,
    n_workers: int,
    mp_method: str,
    task: Optional[str] = None,
    n_resets: int = 3,
    n_steps: int = 2,
) -> Dict[str, float]:
    """
    Test full VecEnv lifecycle and measure timings.

    Returns dict with:
        - create_time: Time to create VecEnv (spawn workers)
        - reset_time: Average time per reset
        - step_time: Average time per step
        - close_time: Time to close VecEnv
        - total_time: Total time
        - success: Whether test completed without errors
    """
    from verl.envs.vec_env import VecEnv
    from verl.envs.environments import make_env
    from verl.envs.captioners import make_captioner

    config = create_test_config(env_name, n_workers, mp_method, task)

    results = {
        'create_time': 0.0,
        'reset_time': 0.0,
        'step_time': 0.0,
        'close_time': 0.0,
        'total_time': 0.0,
        'success': False,
        'error': None,
    }

    total_start = time.perf_counter()
    vec_env = None

    try:
        # Create env/captioner factory functions
        def get_env_fn(rank):
            def init_env():
                return make_env(env_name, task, config, render_mode=None)
            return init_env

        def get_captioner_fn(rank):
            def init_captioner():
                return make_captioner(config)
            return init_captioner

        env_fns = [get_env_fn(i) for i in range(n_workers)]
        captioner_fns = [get_captioner_fn(i) for i in range(n_workers)]

        # Measure creation time
        create_start = time.perf_counter()
        vec_env = VecEnv(
            env_name=env_name,
            config=config,
            env_fns=env_fns,
            captioner_fns=captioner_fns,
        )
        create_end = time.perf_counter()
        results['create_time'] = create_end - create_start

        # Measure reset time (average over n_resets)
        reset_times = []
        for i in range(n_resets):
            reset_start = time.perf_counter()
            obs, info = vec_env.reset(seed=12345 + i, use_incremental_seeds=True)
            reset_end = time.perf_counter()
            reset_times.append(reset_end - reset_start)
        results['reset_time'] = sum(reset_times) / len(reset_times)

        # Measure step time (average over n_steps)
        # Use valid actions for each env type
        if env_name == 'fastsnake':
            action = "<thinking>Moving up.</thinking>\n<action>up</action>"
        elif env_name == 'overcooked':
            action = "<thinking>Staying.</thinking>\n<action>stay</action>"
        elif env_name == 'babyai':
            action = "<thinking>Moving forward.</thinking>\n<action>forward</action>"
        else:
            action = "<action>stay</action>"

        actions = [action] * n_workers
        step_times = []
        for i in range(n_steps):
            step_start = time.perf_counter()
            obs, rewards, terminated, truncated, infos = vec_env.step(actions)
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
        results['step_time'] = sum(step_times) / len(step_times)

        # Measure close time
        close_start = time.perf_counter()
        vec_env.close()
        vec_env = None
        close_end = time.perf_counter()
        results['close_time'] = close_end - close_start

        results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    finally:
        # Ensure cleanup
        if vec_env is not None:
            try:
                vec_env.close()
            except:
                pass
        gc.collect()

    total_end = time.perf_counter()
    results['total_time'] = total_end - total_start

    return results


def run_comparison(
    envs: List[str],
    methods: List[str],
    n_workers: int,
    n_trials: int = 1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run comparison across environments and methods.

    Returns nested dict: results[env_name][method] = {timing_metrics}
    """
    results = {}

    # Map env to task (babyai needs a task)
    env_tasks = {
        'fastsnake': None,
        'overcooked': None,
        'babyai': 'BabyAI-GoToLocalS5N2-v0',
    }

    total_tests = len(envs) * len(methods) * n_trials
    test_num = 0

    for env_name in envs:
        results[env_name] = {}
        task = env_tasks.get(env_name)

        for method in methods:
            trial_results = []

            for trial in range(n_trials):
                test_num += 1
                print(f"\n[{test_num}/{total_tests}] Testing {env_name} with {method} (trial {trial + 1}/{n_trials})...")

                trial_result = test_vecenv_lifecycle(
                    env_name=env_name,
                    n_workers=n_workers,
                    mp_method=method,
                    task=task,
                )
                trial_results.append(trial_result)

                if trial_result['success']:
                    print(f"  Create: {trial_result['create_time']:.2f}s, "
                          f"Reset: {trial_result['reset_time']:.3f}s, "
                          f"Step: {trial_result['step_time']:.3f}s, "
                          f"Close: {trial_result['close_time']:.2f}s")
                else:
                    print(f"  FAILED: {trial_result['error']}")
                    if 'traceback' in trial_result:
                        print(f"  Traceback:\n{trial_result['traceback']}")

            # Average results across trials
            successful_trials = [r for r in trial_results if r['success']]
            if successful_trials:
                avg_result = {
                    'create_time': sum(r['create_time'] for r in successful_trials) / len(successful_trials),
                    'reset_time': sum(r['reset_time'] for r in successful_trials) / len(successful_trials),
                    'step_time': sum(r['step_time'] for r in successful_trials) / len(successful_trials),
                    'close_time': sum(r['close_time'] for r in successful_trials) / len(successful_trials),
                    'total_time': sum(r['total_time'] for r in successful_trials) / len(successful_trials),
                    'success': True,
                    'n_successful': len(successful_trials),
                    'n_trials': n_trials,
                }
            else:
                avg_result = {
                    'success': False,
                    'error': trial_results[0]['error'] if trial_results else 'No trials run',
                    'n_successful': 0,
                    'n_trials': n_trials,
                }

            results[env_name][method] = avg_result

    return results


def print_summary(results: Dict, n_workers: int):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print(f"VECENV MULTIPROCESSING COMPARISON (n_workers={n_workers})")
    print("=" * 80)

    # Get all methods tested
    all_methods = set()
    for env_results in results.values():
        all_methods.update(env_results.keys())
    methods = sorted(all_methods)

    # Print header
    header = f"{'Environment':<15} | "
    header += " | ".join(f"{m:<12}" for m in methods)
    header += " | Speedup"
    print(header)
    print("-" * len(header))

    # Print results for each env
    for env_name, env_results in results.items():
        row = f"{env_name:<15} | "
        times = []
        for method in methods:
            if method in env_results and env_results[method]['success']:
                create_time = env_results[method]['create_time']
                times.append((method, create_time))
                row += f"{create_time:>10.2f}s | "
            else:
                times.append((method, float('inf')))
                row += f"{'FAILED':>10}s | "

        # Calculate speedup (spawn vs fastest alternative)
        spawn_time = next((t for m, t in times if m == 'spawn'), float('inf'))
        other_times = [(m, t) for m, t in times if m != 'spawn' and t < float('inf')]
        if other_times and spawn_time < float('inf'):
            fastest = min(other_times, key=lambda x: x[1])
            speedup = spawn_time / fastest[1]
            row += f"{fastest[0]}: {speedup:.1f}x"
        else:
            row += "N/A"

        print(row)

    print("=" * 80)

    # Print detailed timing breakdown
    print("\nDETAILED TIMING BREAKDOWN:")
    print("-" * 80)

    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for method, method_results in env_results.items():
            if method_results['success']:
                print(f"  {method}:")
                print(f"    Create: {method_results['create_time']:.3f}s")
                print(f"    Reset:  {method_results['reset_time']:.3f}s (avg)")
                print(f"    Step:   {method_results['step_time']:.3f}s (avg)")
                print(f"    Close:  {method_results['close_time']:.3f}s")
                print(f"    Total:  {method_results['total_time']:.3f}s")
            else:
                print(f"  {method}: FAILED - {method_results.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare VecEnv multiprocessing methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with defaults (4 workers, all envs, all methods)
    python scripts/test_vecenv_multiprocessing.py

    # Test with more workers (simulates cluster workload)
    python scripts/test_vecenv_multiprocessing.py --n-workers 50

    # Test specific methods
    python scripts/test_vecenv_multiprocessing.py --methods spawn forkserver

    # Test specific environment
    python scripts/test_vecenv_multiprocessing.py --envs fastsnake

    # Multiple trials for more accurate timing
    python scripts/test_vecenv_multiprocessing.py --n-trials 3
        """
    )
    parser.add_argument(
        '--n-workers', type=int, default=4,
        help='Number of workers/rollouts (default: 4 for quick test, use 50 for realistic test)'
    )
    parser.add_argument(
        '--methods', nargs='+', default=['spawn', 'fork', 'forkserver'],
        choices=['spawn', 'fork', 'forkserver'],
        help='Multiprocessing methods to test (default: all)'
    )
    parser.add_argument(
        '--envs', nargs='+', default=['fastsnake', 'overcooked'],
        choices=['fastsnake', 'overcooked', 'babyai'],
        help='Environments to test (default: fastsnake, overcooked). babyai requires additional setup.'
    )
    parser.add_argument(
        '--n-trials', type=int, default=1,
        help='Number of trials per configuration (default: 1)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VECENV MULTIPROCESSING COMPARISON TEST")
    print("=" * 80)
    print(f"Workers: {args.n_workers}")
    print(f"Methods: {args.methods}")
    print(f"Environments: {args.envs}")
    print(f"Trials per config: {args.n_trials}")

    results = run_comparison(
        envs=args.envs,
        methods=args.methods,
        n_workers=args.n_workers,
        n_trials=args.n_trials,
    )

    print_summary(results, args.n_workers)

    # Return success if at least one non-spawn method worked
    any_alternative_success = any(
        env_results.get(method, {}).get('success', False)
        for env_results in results.values()
        for method in args.methods
        if method != 'spawn'
    )

    if any_alternative_success:
        print("\nForkserver/fork viable as spawn alternative.")
        return 0
    else:
        print("\nWARNING: No alternative to spawn worked successfully.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
