# Adapted from https://github.com/zoeyuchao/mappo/blob/main/onpolicy/envs/env_wrappers.py under the MIT License.
# Original author: yuchao

import logging
import numpy as np
import os
import sys
import multiprocessing
import random
import gc
import traceback

from collections import defaultdict

logger = logging.getLogger(__name__)



def get_process_memory_mb():
    """Get current process memory usage in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert bytes to MB


def _parse_env_flag(name: str, default: bool, override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv:

    def __init__(self, env_name, config, env_fns, captioner_fns, worker_debug: bool | None = None):

        self.env_name = env_name
        self.config = config
        self.n_rollouts = config.envs.n_rollouts
        assert len(env_fns) == self.n_rollouts, "Number of env_fns must match n_rollouts"

        # Extract epsilon for epsilon-greedy exploration (centralized here, not per-env)
        self.epsilon = 0.0
        if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
            self.epsilon = getattr(config.prompt.prompt, 'epsilon', 0.0)
        if self.epsilon > 0:
            logger.info(f"[VecEnv] Epsilon-greedy exploration enabled: epsilon={self.epsilon}")
        
        # Get multiprocessing method from config, default to 'fork' for performance
        # 'fork' is ~100x faster than 'spawn' on NFS (workers inherit imports via COW)
        # Note: VecEnv workers don't use CUDA (envs are CPU-only), so fork is safe here
        mp_method = config.envs.get('vec_env_multiprocessing', 'fork')
        if mp_method not in ['fork', 'spawn', 'forkserver']:
            raise ValueError(f"Invalid vec_env_multiprocessing method: {mp_method}. Must be one of: fork, spawn, forkserver")

        # Policy 1 guard: prevent fork when JAX is already imported in parent process
        # JAX + fork after threads are started can cause deadlocks
        if mp_method == 'fork':
            jax_modules = [m for m in sys.modules if m == 'jax' or m == 'jaxlib' or m == 'jaxmarl'
                          or m.startswith('jax.') or m.startswith('jaxlib.') or m.startswith('jaxmarl.')]
            if jax_modules and not os.environ.get('VERL_ALLOW_UNSAFE_FORK_WITH_JAX', '').strip().lower() in ('1', 'true', 'yes'):
                raise RuntimeError(
                    f"[VecEnv] Cannot use fork multiprocessing when JAX is already imported in parent process. "
                    f"Found JAX modules: {jax_modules[:5]}{'...' if len(jax_modules) > 5 else ''}. "
                    f"This can cause deadlocks. Options:\n"
                    f"  1. Create VecEnv BEFORE importing JAX (use prewarm)\n"
                    f"  2. Use vec_env_multiprocessing=spawn (slower on NFS)\n"
                    f"  3. Set VERL_ALLOW_UNSAFE_FORK_WITH_JAX=1 to override (at your own risk)"
                )

        self.mp_context = multiprocessing.get_context(mp_method)
        self.mp_method = mp_method  # Store for hard_reset to check
        logger.info(f"[VecEnv] Using multiprocessing method: {mp_method}")

        # Controls noisy worker memory prints like "[Worker X] Memory ...".
        self.worker_debug = _parse_env_flag(
            "VERL_VEC_ENV_WORKER_DEBUG",
            default=False,
            override=worker_debug,
        )
        
        self.remotes, self.work_remotes = zip(*[self.mp_context.Pipe() for _ in range(self.n_rollouts)])
        self.processes = []
        for rank, (work_remote, remote, env_fn, captioner_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns, captioner_fns)):
            p = self.mp_context.Process(
                target=worker,
                args=(rank, work_remote, remote, env_name, CloudpickleWrapper(env_fn), CloudpickleWrapper(captioner_fn), self.epsilon, self.worker_debug),
            )
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            self.processes.append(p)
        
        for remote in self.work_remotes:
            remote.close()

        # Flag to mark VecEnv as unusable after partial hard_reset failure
        # Once set, this VecEnv should be closed and removed from any pool
        self._unusable = False

        # Cache for storing last known state of environments (for skip functionality)
        self.last_obs = [None] * self.n_rollouts
        self.last_reward = [0.0] * self.n_rollouts
        self.last_terminated = [False] * self.n_rollouts
        self.last_truncated = [False] * self.n_rollouts
        # Use list comprehension to create independent dicts (not shared references)
        self.last_info = [{"metrics": {}} for _ in range(self.n_rollouts)]
            
    def step(self, actions):
        """
        Returns lists for obs and infos, numpy vectors for reward, terminated, truncated
        Always returns the most recent values of these veriables for each environment, 
           even if the environment was not processed this step (because already completed)
        """
        # Handle skip actions for frozen environments
        skip_action = "__SKIP__"
        active_remotes = []
        active_actions = []
        skip_indices = []
        
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            if action == skip_action:
                skip_indices.append(i)
            else:
                active_remotes.append(remote)
                active_actions.append(action)
        
        # Only send step commands to active environments
        for i, (remote, action) in enumerate(zip(active_remotes, active_actions)):
            try:
                remote.send(('step', action))
            except BrokenPipeError as e:
                logger.error(f"[VecEnv.step] BrokenPipeError when sending to worker {i}: {e}")
                raise RuntimeError("Worker process connection broken during step. Check for errors in worker processes.")
        
        # Collect results from active environments
        active_results = []
        for i, remote in enumerate(active_remotes):
            try:
                result = remote.recv()
                # Check if worker sent an error
                if isinstance(result, tuple) and len(result) > 0 and result[0] == 'error':
                    logger.error(f"[VecEnv.step] Worker {i} sent error: {result[1]}")
                    raise RuntimeError(f"Worker {i} error: {result[1]}")
                
                active_results.append(result)
            except Exception as e:
                logger.error(f"[VecEnv.step] Exception receiving from worker {i}: {e}")
                raise
        
        # Reconstruct full results with cached data for skipped environments
        full_results = []
        active_idx = 0
        for i in range(len(self.remotes)):
            if i in skip_indices:
                # Return cached data for skipped environments
                full_results.append((
                    self.last_obs[i],
                    self.last_reward[i],
                    self.last_terminated[i],
                    self.last_truncated[i],
                    self.last_info[i]
                ))
            else:
                full_results.append(active_results[active_idx])
                active_idx += 1
        
        obs, rews, terminated, truncated, infos = zip(*full_results)

        
        # Update cache with current results
        for i in range(len(self.remotes)):
            self.last_obs[i] = obs[i]
            self.last_reward[i] = rews[i]
            self.last_terminated[i] = terminated[i]
            self.last_truncated[i] = truncated[i]
            self.last_info[i] = infos[i]
        
        return obs, np.stack(rews), np.stack(terminated), np.stack(truncated), infos
    
    def reset(self, seed=None, seed_group_size=None, use_incremental_seeds=False, seeds=None):
        """
        Reset all environments with seeds.

        Args:
            seed: Base seed for resetting environments
            seed_group_size: If provided, group successive sets of this many rollouts to have the same seed.
                           Each group gets seed + group_index. Must divide n_rollouts evenly.
            use_incremental_seeds: If True, each worker gets seed + worker_rank (for evaluation).
                                 If False, all workers get the same seed (for GRPO training).
            seeds: Explicit list of seeds, one per worker. If provided, overrides all other seed logic.
                   Used for batched evaluation where seeds are computed upfront.
        """
        # Explicit seeds list takes priority over all other seed logic
        if seeds is not None:
            if len(seeds) != self.n_rollouts:
                raise ValueError(f"seeds list length ({len(seeds)}) must match n_rollouts ({self.n_rollouts})")
            for i, remote in enumerate(self.remotes):
                try:
                    remote.send(('reset', seeds[i]))
                except Exception as e:
                    raise RuntimeError(f"Error resetting environment {i}: {e}")
        else:
            # Original seed computation logic
            if seed_group_size == self.n_rollouts:
                seed_group_size = None
            if seed_group_size is not None and self.n_rollouts % seed_group_size != 0:
                raise ValueError("n_rollouts must be divisible by seed_group_size")

            for i, remote in enumerate(self.remotes):
                try:
                    if seed_group_size is not None and seed is not None:
                        # Group successive sets of seed_group_size rollouts to have the same seed
                        # This takes priority over use_incremental_seeds to ensure proper grouping
                        group_index = i // seed_group_size
                        worker_seed = seed + group_index
                    elif use_incremental_seeds and seed is not None:
                        # Create unique seed for each worker: base_seed + worker_rank (for evaluation)
                        worker_seed = seed + i
                    else:
                        # Use the same seed for all workers (for GRPO training)
                        worker_seed = seed
                    remote.send(('reset', worker_seed))
                except Exception as e:
                    raise RuntimeError(f"Error resetting environment {i}: {e}")
        
        observations, infos = zip(*[remote.recv() for remote in self.remotes])
        
        # Update cache with reset results
        for i in range(len(self.remotes)):
            self.last_obs[i] = observations[i]
            self.last_reward[i] = 0.0
            self.last_terminated[i] = False
            self.last_truncated[i] = False
            self.last_info[i] = infos[i]
        
        return observations, infos
    
    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        images = [remote.recv() for remote in self.remotes]
        return images
    
    def get_worker_memory_stats(self):
        """Get memory usage statistics from all workers"""
        for remote in self.remotes:
            remote.send(('get_memory', None))
        
        memory_stats = [remote.recv() for remote in self.remotes]
        
        # Calculate statistics
        total_memory = sum(stat['memory_mb'] for stat in memory_stats)
        avg_memory = total_memory / len(memory_stats)
        max_memory = max(stat['memory_mb'] for stat in memory_stats)
        min_memory = min(stat['memory_mb'] for stat in memory_stats)
        
        print(f"\n{'='*60}")
        print(f"Worker Memory Statistics ({len(memory_stats)} workers)")
        print(f"{'='*60}")
        print(f"Total:   {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
        print(f"Average: {avg_memory:.2f} MB per worker")
        print(f"Max:     {max_memory:.2f} MB (worker {memory_stats[np.argmax([s['memory_mb'] for s in memory_stats])]['rank']})")
        print(f"Min:     {min_memory:.2f} MB (worker {memory_stats[np.argmin([s['memory_mb'] for s in memory_stats])]['rank']})")
        print(f"{'='*60}\n")
        
        return memory_stats

    def hard_reset(self, *, env_name: str, task: str, config, render_mode: str | None = None) -> None:
        """
        Rebuild env + captioner inside existing worker processes.

        This allows reusing worker processes for different env configs without
        needing to fork new processes (which can deadlock after threads start).

        After hard_reset, you MUST call reset(seeds=...) before stepping.

        Args:
            env_name: Environment name (e.g., 'snake', 'babyai', 'overcooked')
            task: Task/layout name for the environment
            config: Full config object (will be converted to dict for pickling)
            render_mode: Optional render mode for env

        Raises:
            RuntimeError: If any worker fails to rebuild
            ValueError: If config.envs.n_rollouts doesn't match pool worker count
        """
        import time
        start_time = time.time()

        # Validate n_rollouts matches pool size (workers can't be added/removed)
        config_n_rollouts = config.envs.n_rollouts if hasattr(config, 'envs') else None
        if config_n_rollouts is not None and config_n_rollouts != self.n_rollouts:
            raise ValueError(
                f"[VecEnv.hard_reset] config.envs.n_rollouts ({config_n_rollouts}) doesn't match "
                f"pool worker count ({self.n_rollouts}). Cannot change worker count via hard_reset."
            )

        # Convert config to picklable dict
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(config):
                config_blob = OmegaConf.to_container(config, resolve=True)
            else:
                config_blob = config
        except ImportError:
            config_blob = config

        payload = {
            'env_name': env_name,
            'task': task,
            'render_mode': render_mode,
            'config_blob': config_blob,
        }

        # Send hard_reset to all workers
        for remote in self.remotes:
            try:
                remote.send(('hard_reset', payload))
            except BrokenPipeError as e:
                raise RuntimeError(f"[VecEnv.hard_reset] Worker connection broken: {e}")

        # Wait for all workers to respond with timeout
        # Default 60s per worker; env rebuild can be slow (JAX init, model loading, etc.)
        timeout_per_worker = float(os.environ.get('VERL_HARD_RESET_TIMEOUT', '60'))
        errors = []
        for i, remote in enumerate(self.remotes):
            try:
                # Use poll() with timeout to avoid hanging forever on wedged workers
                if not remote.poll(timeout=timeout_per_worker):
                    errors.append(f"Worker {i}: timeout after {timeout_per_worker}s (may be deadlocked)")
                    continue
                status, result = remote.recv()
                if status == 'error':
                    errors.append(f"Worker {i}: {result}")
            except Exception as e:
                errors.append(f"Worker {i}: recv failed - {e}")

        if errors:
            # CRITICAL: Some workers may have succeeded while others failed.
            # The VecEnv is now in a mixed state and cannot be safely used.
            # Mark it unusable so the pool can evict and close it.
            self._unusable = True
            raise RuntimeError(
                f"[VecEnv.hard_reset] {len(errors)} worker(s) failed. "
                f"VecEnv is now in mixed state and marked unusable. "
                f"Pool should close and evict this entry.\n" + "\n".join(errors)
            )

        # Update internal state
        self.env_name = env_name
        self.config = config

        # Update epsilon from new config
        self.epsilon = 0.0
        if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
            self.epsilon = getattr(config.prompt.prompt, 'epsilon', 0.0)
        elif isinstance(config_blob, dict):
            self.epsilon = config_blob.get('prompt', {}).get('prompt', {}).get('epsilon', 0.0)

        # Clear cached obs/info (will be populated on next reset)
        self.last_obs = [None] * self.n_rollouts
        self.last_reward = [0.0] * self.n_rollouts
        self.last_terminated = [False] * self.n_rollouts
        self.last_truncated = [False] * self.n_rollouts
        # Use list comprehension to create independent dicts (not shared references)
        self.last_info = [{"metrics": {}} for _ in range(self.n_rollouts)]

        elapsed = time.time() - start_time
        logger.info(f"[VecEnv] hard_reset completed in {elapsed:.2f}s for env={env_name}, task={task}")

    @property
    def unusable(self) -> bool:
        """Check if this VecEnv is unusable due to partial hard_reset failure.

        When True, this VecEnv should be closed and removed from any pool.
        """
        return self._unusable

    def close(self):
        # Send close command to all worker processes
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception as e:
                logger.warning(f"[VecEnv.close] Failed to send close command to worker: {e}")
        
        # Wait for all processes to terminate and join them
        for i, process in enumerate(self.processes):
            try:
                # Give each process up to 5 seconds to terminate gracefully
                process.join(timeout=5.0)
                if process.is_alive():
                    logger.warning(f"[VecEnv.close] Worker process {i} did not terminate gracefully, forcing termination")
                    process.terminate()
                    process.join(timeout=2.0)  # Give it 2 more seconds after terminate
                    if process.is_alive():
                        logger.error(f"[VecEnv.close] Worker process {i} still alive after terminate, killing it")
                        process.kill()
                        process.join()
            except Exception as e:
                logger.error(f"[VecEnv.close] Exception while joining worker process {i}: {e}")
        
        # Close all remote connections
        for remote in self.remotes:
            try:
                remote.close()
            except Exception as e:
                logger.warning(f"[VecEnv.close] Failed to close remote connection: {e}")
        
        logger.info(f"[VecEnv] Successfully closed {len(self.processes)} worker processes")
 

    
def worker(rank, remote, parent_remote, env_name, env_fn_wrapper, captioner_fn_wrapper, epsilon=0.0, debug: bool = True):
    random.seed(rank)
    np.random.seed(rank)
    
    parent_remote.close()
    
    # Memory tracking (optional; can be expensive and requires psutil)
    mem_start = None
    if debug:
        mem_start = get_process_memory_mb()
        print(f"[Worker {rank}] Memory at start: {mem_start:.2f} MB", flush=True)
    
    try:
        env = env_fn_wrapper.x()
        if debug:
            mem_after_env = get_process_memory_mb()
            delta = (mem_after_env - mem_start) if mem_start is not None else 0.0
            print(
                f"[Worker {rank}] Memory after env creation: {mem_after_env:.2f} MB "
                f"(delta: +{delta:.2f} MB)",
                flush=True,
            )
    except Exception as e:
        print(f"[ERROR] Worker {rank}: Failed to create environment: {e}")
        raise
    
    try:
        captioner = captioner_fn_wrapper.x()
        if debug:
            mem_after_captioner = get_process_memory_mb()
            # Recompute mem_after_env if we didn't measure it above
            mem_after_env = mem_after_env if 'mem_after_env' in locals() else (mem_start or mem_after_captioner)
            delta = mem_after_captioner - mem_after_env
            print(
                f"[Worker {rank}] Memory after captioner creation: {mem_after_captioner:.2f} MB "
                f"(delta: +{delta:.2f} MB)",
                flush=True,
            )
            print(f"[Worker {rank}] Total memory used: {mem_after_captioner:.2f} MB", flush=True)
    except Exception as e:
        print(f"[ERROR] Worker {rank}: Failed to create captioner: {e}")
        raise
    
    image = None
    reset_count = 0
    action_pct_warning_logged = False

    def env_step(action):
        nonlocal action_pct_warning_logged
        try:
            # Use extract_action_instance if available (for multi-action support).
            # In multi-action mode, do NOT silently fall back: that would parse <action>
            # instead of <decision> and can produce 0% valid-action ratio despite correct tags.
            requires_instance = bool(getattr(env, "multi_action_reasoning", False))
            extract_fn = getattr(env, "extract_action_instance", None)
            if requires_instance and extract_fn is None:
                raise RuntimeError(
                    "Environment has multi_action_reasoning=True but no extract_action_instance(); "
                    "cannot parse multi-action <decision> outputs"
                )
            if extract_fn is None:
                extract_fn = env.extract_action
            full_action, extracted_action, executed_action, is_valid, metrics = extract_fn(action)

            # Epsilon-greedy exploration (centralized here, not per-environment)
            # Only explore if model produced valid output - invalid format is a training signal
            explored = False
            if epsilon > 0 and is_valid and random.random() < epsilon:
                # Guard: epsilon requires a sequence-like action space for random.choice()
                if not isinstance(env.language_action_space, (list, tuple)):
                    raise RuntimeError(
                        f"[Worker {rank}] epsilon={epsilon} but language_action_space is {type(env.language_action_space).__name__}, "
                        "not a list/tuple. Epsilon exploration requires a finite, indexable action space."
                    )
                executed_action = random.choice(env.language_action_space)
                explored = True
            metrics["behavior/epsilon_explored"] = explored * 1.0

            # Track action distribution (indicator variables - mean gives percentage)
            # Generated = what model output (pre-epsilon), Executed = what was actually used
            # Only track for envs with small, static, indexable action spaces
            action_space = env.language_action_space

            # Track invalid parses (always, regardless of action space)
            metrics["action_pct/invalid"] = 0.0 if is_valid else 1.0

            # Check length before materializing to avoid performance trap with large/infinite iterables
            if hasattr(action_space, '__len__') and len(action_space) <= 20:
                action_list = list(action_space) if not isinstance(action_space, (list, tuple)) else action_space
                for action_name in action_list:
                    # Sanitize action name for metric key (replace problematic chars)
                    # Note: collision possible if names differ only by these chars (e.g. "up/left" vs "up_left")
                    # Acceptable for typical small action spaces (snake, babyai, overcooked)
                    safe_name = str(action_name).replace('/', '_').replace('[', '_').replace(']', '_')
                    # Generated: extracted_action if valid parse, else None
                    metrics[f"action_pct/generated/{safe_name}"] = 1.0 if (is_valid and extracted_action == action_name) else 0.0
                    # Executed: final action after epsilon-greedy
                    metrics[f"action_pct/executed/{safe_name}"] = 1.0 if executed_action == action_name else 0.0
            elif not action_pct_warning_logged:
                # Log once per worker - action distribution tracking skipped
                action_pct_warning_logged = True
                print(f"[Worker {rank}] Skipping action_pct/generated and action_pct/executed metrics: "
                      f"action_space has no __len__ or len > 20")

            env_obs, reward, terminated, truncated, info = env.step(executed_action, is_valid)
            info["action_was_valid"] = is_valid
            info["epsilon_explored"] = explored  # For trainer-side re-tokenization
            if executed_action is not None:
                info["executed_action_text"] = executed_action

            image = env_obs.get("image", None)
            instructions = env_obs["mission"]  if env_name == "babyai" else None
            inst_prompt = env.get_instruction_prompt(instructions=instructions, info=info)
            captioner.prompt_builder.update_instruction_prompt(inst_prompt)
            captioner.update_action(full_action, executed_action)
            info["metrics"] = metrics
            return captioner.get_obs(env_obs), reward, terminated, truncated, info, image
        except Exception as e:
            print(f"[ERROR] Worker {rank}: Exception in env_step: {e}")
            import traceback
            print(f"[ERROR] Worker {rank}: Traceback: {traceback.format_exc()}")
            raise

    def env_reset(seed=None):
        nonlocal reset_count
        captioner.reset()
        env_obs, info = env.reset(seed=seed)
        image = env_obs.get("image", None)
        instructions = env_obs["mission"]  if env_name == "babyai" else None
        inst_prompt = env.get_instruction_prompt(instructions=instructions)
        captioner.prompt_builder.update_instruction_prompt(inst_prompt)
        
        # Log memory on first reset
        reset_count += 1
        if reset_count == 1:
            if debug:
                mem_after_reset = get_process_memory_mb()
                print(f"[Worker {rank}] Memory after first reset: {mem_after_reset:.2f} MB", flush=True)
        
        return captioner.get_obs(env_obs), info, image
        
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, terminated, truncated, info, image = env_step(data)
                if terminated or truncated:
                    obs, _, image = env_reset(seed=None)  
                remote.send((obs, reward, terminated, truncated, info))
                
            elif cmd == 'reset':
                seed = data
                obs, info, image = env_reset(seed=seed)
                remote.send((obs, info))
                
            elif cmd == 'render':
                remote.send(image)
                
            elif cmd == 'get_memory':
                mem_current = get_process_memory_mb()
                remote.send({'rank': rank, 'memory_mb': mem_current})

            elif cmd == 'hard_reset':
                # Rebuild env + captioner inside existing worker process
                # TRANSACTIONAL: build new first, then swap (keeps worker usable on failure)
                payload = data
                new_env_name = payload['env_name']
                task = payload['task']
                render_mode = payload['render_mode']
                config_blob = payload['config_blob']

                try:
                    # Import factories lazily (avoid pulling in envs during worker init)
                    from verl.envs.environments import make_env
                    from verl.envs.captioners import make_captioner
                    from omegaconf import OmegaConf

                    # Reconstruct config from blob
                    config = OmegaConf.create(config_blob)

                    # BUILD NEW FIRST (before touching old state)
                    # If this fails, old env/captioner remain usable
                    new_env = make_env(new_env_name, task, config, render_mode=render_mode)
                    new_captioner = make_captioner(config)

                    # SUCCESS - now safe to close old and swap
                    old_env = env
                    old_captioner = captioner

                    # Swap to new
                    env = new_env
                    captioner = new_captioner
                    env_name = new_env_name

                    # Close old env (best effort, don't fail if close errors)
                    if hasattr(old_env, 'close'):
                        try:
                            old_env.close()
                        except Exception as close_err:
                            print(f"[Worker {rank}] Warning: error closing old env: {close_err}")

                    # Close old captioner (best effort, if it has resources)
                    if hasattr(old_captioner, 'close'):
                        try:
                            old_captioner.close()
                        except Exception as close_err:
                            print(f"[Worker {rank}] Warning: error closing old captioner: {close_err}")

                    # Cleanup old references
                    del old_env
                    del old_captioner
                    gc.collect()

                    # Update epsilon from new config
                    epsilon = 0.0
                    if hasattr(config, 'prompt') and hasattr(config.prompt, 'prompt'):
                        epsilon = getattr(config.prompt.prompt, 'epsilon', 0.0)

                    # Reset counters
                    reset_count = 0
                    action_pct_warning_logged = False
                    image = None

                    if debug:
                        mem_after = get_process_memory_mb()
                        print(f"[Worker {rank}] hard_reset completed: env={new_env_name}, task={task}, mem={mem_after:.2f} MB", flush=True)

                    remote.send(('ok', None))

                except Exception as e:
                    # FAILURE - old env/captioner still intact, worker remains usable
                    tb_str = traceback.format_exc()
                    print(f"[ERROR] Worker {rank}: hard_reset failed (old env preserved): {e}\n{tb_str}")
                    remote.send(('error', f"{e}\n{tb_str}"))

            elif cmd == 'close':
                # Close env
                if hasattr(env, 'close'):
                    try:
                        env.close()
                    except Exception as close_err:
                        print(f"[Worker {rank}] Warning: error closing env on shutdown: {close_err}")
                # Close captioner (best effort, matches hard_reset pattern)
                if hasattr(captioner, 'close'):
                    try:
                        captioner.close()
                    except Exception as close_err:
                        print(f"[Worker {rank}] Warning: error closing captioner on shutdown: {close_err}")
                remote.close()
                break
            else:
                print(f"[ERROR] Worker {rank}: Unknown command: '{cmd}'")
                raise NotImplementedError
                
        except Exception as e:
            print(f"[ERROR] Worker {rank}: Exception in command loop: {e}")
            import traceback
            print(f"[ERROR] Worker {rank}: Traceback: {traceback.format_exc()}")
            # Send error back to main process
            try:
                remote.send(('error', str(e)))
            except Exception as send_error:
                print(f"[ERROR] Worker {rank}: Failed to send error to main process: {send_error}")
                pass  # If we can't send the error, just exit
            break

        
  
    
