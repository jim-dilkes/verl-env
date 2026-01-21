#!/usr/bin/env python3
"""Profile VecEnv memory scaling with different worker counts.

This script measures the actual memory cost per VecEnv worker
by creating VecEnvs with increasing worker counts and measuring
the delta.

Run from repo root:
    python scripts/profile_vecenv_scaling.py
"""
import os
import sys
import psutil
import gc
import time

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_children_memory_mb():
    """Get memory of all child processes in MB."""
    process = psutil.Process(os.getpid())
    children = process.children(recursive=True)
    total = 0
    for c in children:
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return total / 1024 / 1024, len(children)


def get_total_memory():
    """Get total memory (parent + children)."""
    parent = get_memory_mb()
    children, n = get_children_memory_mb()
    return parent + children, parent, children, n


def make_test_config(n_rollouts=4):
    """Create minimal config for testing."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        'envs': {
            'env_name': 'fastsnake',
            'n_rollouts': n_rollouts,
            'vec_env_multiprocessing': 'fork',
            'task': 'default',
            'fastsnake_kwargs': {
                'width': 10,
                'height': 10,
                'max_rounds': 8,
                'num_external_snakes': 1,
                'num_random_snakes': 1,
                'death_reward': -1,
            },
            'captioner': {
                'name': 'history',
                'type': 'naive',
                'system_instruction': '',
                'history_length': 1,
                'max_text_history': 0,
                'max_image_history': 0,
                'max_cot_history': 0,
            },
        },
        'prompt': {
            'prompt': {
                'epsilon': 0.0,
            }
        },
    })
    return config


def create_vecenv(n_workers):
    """Create a VecEnv with specified worker count."""
    from verl.envs.vec_env import VecEnv
    from verl.envs.environments import make_env
    from verl.envs.captioners import make_captioner

    config = make_test_config(n_rollouts=n_workers)

    def get_env_fn(rank):
        def init_env():
            return make_env("fastsnake", "default", config, render_mode=None)
        return init_env

    def get_captioner_fn(rank):
        def init_captioner():
            return make_captioner(config)
        return init_captioner

    env = VecEnv(
        env_name="fastsnake",
        config=config,
        env_fns=[get_env_fn(i) for i in range(n_workers)],
        captioner_fns=[get_captioner_fn(i) for i in range(n_workers)],
    )
    return env


def main():
    print("\n" + "=" * 70)
    print("VecEnv Memory Scaling Analysis")
    print("=" * 70)

    # Import and establish baseline
    gc.collect()
    baseline_total, baseline_parent, _, _ = get_total_memory()
    print(f"\nBaseline (before imports): {baseline_total:.1f} MB")

    from verl.envs.vec_env import VecEnv
    gc.collect()

    import_total, import_parent, _, _ = get_total_memory()
    print(f"After imports: {import_total:.1f} MB (+{import_total - baseline_total:.1f} MB)")

    # Test different worker counts
    worker_counts = [4, 8, 16, 32, 50, 100, 200, 400]

    print(f"\n{'Workers':<10} {'Total MB':<12} {'Delta MB':<12} {'Per Worker':<12} {'Children':<10}")
    print("-" * 56)

    envs = []
    prev_total = import_total
    cumulative_workers = 0

    for n_workers in worker_counts:
        # Create VecEnv
        env = create_vecenv(n_workers)
        envs.append(env)

        # Wait for workers to stabilize
        time.sleep(0.5)
        gc.collect()

        # Measure
        total, parent, children, n_children = get_total_memory()
        delta = total - prev_total
        cumulative_workers += n_workers
        per_worker = (total - import_total) / cumulative_workers if cumulative_workers > 0 else 0

        print(f"{n_workers:<10} {total:<12.1f} {delta:<12.1f} {per_worker:<12.2f} {n_children:<10}")
        prev_total = total

    # Summary
    final_total, _, final_children, final_n = get_total_memory()
    total_workers = sum(worker_counts)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total VecEnv workers created: {total_workers}")
    print(f"Final memory: {final_total:.1f} MB")
    print(f"Memory from VecEnvs: {final_total - import_total:.1f} MB")
    print(f"Average per worker: {(final_total - import_total) / total_workers:.2f} MB")
    print(f"Child processes: {final_n}")

    # Compare with cluster config
    print("\n" + "=" * 70)
    print("CLUSTER CONFIG COMPARISON")
    print("=" * 70)

    # Baseline config: 32 train + 600 eval (50*4 + 400)
    baseline_workers = 32 + 50 + 50 + 50 + 50 + 400
    baseline_estimate = baseline_workers * (final_total - import_total) / total_workers

    # Minimal config: 4 train + 4 eval
    minimal_workers = 4 + 4
    minimal_estimate = minimal_workers * (final_total - import_total) / total_workers

    print(f"Baseline config (32 train + 600 eval = {baseline_workers} workers):")
    print(f"  Estimated VecEnv memory: {baseline_estimate:.1f} MB ({baseline_estimate/1024:.1f} GB)")
    print(f"Minimal config (4 train + 4 eval = {minimal_workers} workers):")
    print(f"  Estimated VecEnv memory: {minimal_estimate:.1f} MB")
    print(f"Difference: {baseline_estimate - minimal_estimate:.1f} MB ({(baseline_estimate - minimal_estimate)/1024:.1f} GB)")

    # Cleanup
    print("\n>>> Cleaning up...")
    for env in envs:
        env.close()
    gc.collect()

    cleanup_total, _, _, cleanup_n = get_total_memory()
    print(f"After cleanup: {cleanup_total:.1f} MB, {cleanup_n} children")
    print("=" * 70)


if __name__ == "__main__":
    main()
