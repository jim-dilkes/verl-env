#!/usr/bin/env python3
"""Profile VecEnv memory usage with pooling.

Run from repo root:
    python scripts/profile_vecenv_memory.py
"""
import os
import sys
import psutil
import gc

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
    total = sum(c.memory_info().rss for c in children) / 1024 / 1024
    return total, len(children)


def print_memory(label):
    """Print current memory stats."""
    parent_mb = get_memory_mb()
    children_mb, n_children = get_children_memory_mb()
    total_mb = parent_mb + children_mb

    from verl.envs.vec_env import VecEnv
    stats = VecEnv.get_stats()

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  VecEnv instances: {stats['active_vecenvs']}")
    print(f"  VecEnv workers:   {stats['total_workers']}")
    print(f"  Child processes:  {n_children}")
    print(f"  Parent memory:    {parent_mb:.1f} MB")
    print(f"  Children memory:  {children_mb:.1f} MB")
    print(f"  TOTAL memory:     {total_mb:.1f} MB")
    print(f"{'='*60}\n")
    return total_mb


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


def main():
    print("\n" + "="*60)
    print("VecEnv Memory Profiler")
    print("="*60)

    # Baseline
    gc.collect()
    baseline = print_memory("BASELINE (no VecEnvs)")

    # Import after baseline to see import overhead
    from verl.envs.vec_env import VecEnv
    from verl.envs.environments import make_env
    from verl.envs.captioners import make_captioner

    print_memory("After imports")

    # Create training VecEnv (like trainer does)
    print("\n>>> Creating training VecEnv (4 workers)...")
    train_config = make_test_config(n_rollouts=4)

    def get_env_fn(rank):
        def init_env():
            return make_env("fastsnake", "default", train_config)
        return init_env

    def get_captioner_fn(rank):
        def init_captioner():
            return make_captioner(train_config)
        return init_captioner

    train_env = VecEnv(
        env_name="fastsnake",
        config=train_config,
        env_fns=[get_env_fn(i) for i in range(4)],
        captioner_fns=[get_captioner_fn(i) for i in range(4)],
    )

    after_train = print_memory("After training VecEnv (4 workers)")

    # Simulate prewarm with different worker counts (like eval configs)
    eval_configs = [
        ("eval_small", 4),
        ("eval_medium", 50),
        ("eval_large", 100),
        ("eval_xlarge", 400),
    ]

    eval_envs = []
    for name, n_workers in eval_configs:
        print(f"\n>>> Creating {name} VecEnv ({n_workers} workers)...")
        config = make_test_config(n_rollouts=n_workers)

        env = VecEnv(
            env_name="fastsnake",
            config=config,
            env_fns=[get_env_fn(i) for i in range(n_workers)],
            captioner_fns=[get_captioner_fn(i) for i in range(n_workers)],
        )
        eval_envs.append(env)

        print_memory(f"After {name} ({n_workers} workers)")

    # Summary
    final = print_memory("FINAL (all VecEnvs active)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Baseline:     {baseline:.1f} MB")
    print(f"  After train:  {after_train:.1f} MB (+{after_train - baseline:.1f} MB)")
    print(f"  Final:        {final:.1f} MB (+{final - baseline:.1f} MB from baseline)")
    print(f"  Memory per worker: ~{(final - baseline) / VecEnv.get_stats()['total_workers']:.2f} MB")
    print("="*60)

    # Cleanup
    print("\n>>> Closing all VecEnvs...")
    train_env.close()
    for env in eval_envs:
        env.close()

    gc.collect()
    print_memory("After cleanup")


if __name__ == "__main__":
    main()
