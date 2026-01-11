# Adapted from https://github.com/zoeyuchao/mappo/blob/main/onpolicy/envs/env_wrappers.py under the MIT License.
# Original author: yuchao

import numpy as np
import os
import torch
import multiprocessing
import random
import psutil

from collections import defaultdict



def get_process_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert bytes to MB


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
    
    def __init__(self, env_name, config, env_fns, captioner_fns):
        
        self.config = config
        self.n_rollouts = config.envs.n_rollouts
        assert len(env_fns) == self.n_rollouts, "Number of env_fns must match n_rollouts"
        
        # Get multiprocessing method from config, default to 'spawn' for safety
        # 'spawn' avoids CUDA segfaults when forking (fork copies parent's CUDA state)
        mp_method = config.envs.get('vec_env_multiprocessing', 'spawn')
        if mp_method not in ['fork', 'spawn', 'forkserver']:
            raise ValueError(f"Invalid vec_env_multiprocessing method: {mp_method}. Must be one of: fork, spawn, forkserver")
        
        self.mp_context = multiprocessing.get_context(mp_method)
        print(f"[VecEnv] Using multiprocessing method: {mp_method}")
        
        self.remotes, self.work_remotes = zip(*[self.mp_context.Pipe() for _ in range(self.n_rollouts)])
        self.processes = []
        for rank, (work_remote, remote, env_fn, captioner_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns, captioner_fns)):
            p = self.mp_context.Process(
                target=worker,
                args=(rank, work_remote, remote, env_name, CloudpickleWrapper(env_fn), CloudpickleWrapper(captioner_fn)),
            )
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            self.processes.append(p)
        
        for remote in self.work_remotes:
            remote.close()
        
        
        # Cache for storing last known state of environments (for skip functionality)
        self.last_obs = [None] * self.n_rollouts
        self.last_reward = [0.0] * self.n_rollouts
        self.last_terminated = [False] * self.n_rollouts
        self.last_truncated = [False] * self.n_rollouts
        self.last_info = [{"metrics": {}}] * self.n_rollouts
            
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
                print(f"[ERROR] VecEnv.step: BrokenPipeError when sending to worker {i}: {e}")
                raise RuntimeError("Worker process connection broken during step. Check for errors in worker processes.")
        
        # Collect results from active environments
        active_results = []
        for i, remote in enumerate(active_remotes):
            try:
                result = remote.recv()
                # Check if worker sent an error
                if isinstance(result, tuple) and len(result) > 0 and result[0] == 'error':
                    print(f"[ERROR] VecEnv.step: Worker {i} sent error: {result[1]}")
                    raise RuntimeError(f"Worker {i} error: {result[1]}")
                
                active_results.append(result)
            except Exception as e:
                print(f"[ERROR] VecEnv.step: Exception receiving from worker {i}: {e}")
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
    
    def reset(self, seed=None, seed_group_size=None, use_incremental_seeds=False):
        """
        Reset all environments with seeds.
        
        Args:
            seed: Base seed for resetting environments
            seed_group_size: If provided, group successive sets of this many rollouts to have the same seed.
                           Each group gets seed + group_index. Must divide n_rollouts evenly.
            use_incremental_seeds: If True, each worker gets seed + worker_rank (for evaluation).
                                 If False, all workers get the same seed (for GRPO training).
        """
        if seed_group_size==self.n_rollouts:
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

    def close(self):
        # Send close command to all worker processes
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception as e:
                print(f"[WARNING] VecEnv.close: Failed to send close command to worker: {e}")
        
        # Wait for all processes to terminate and join them
        for i, process in enumerate(self.processes):
            try:
                # Give each process up to 5 seconds to terminate gracefully
                process.join(timeout=5.0)
                if process.is_alive():
                    print(f"[WARNING] VecEnv.close: Worker process {i} did not terminate gracefully, forcing termination")
                    process.terminate()
                    process.join(timeout=2.0)  # Give it 2 more seconds after terminate
                    if process.is_alive():
                        print(f"[ERROR] VecEnv.close: Worker process {i} still alive after terminate, killing it")
                        process.kill()
                        process.join()
            except Exception as e:
                print(f"[ERROR] VecEnv.close: Exception while joining worker process {i}: {e}")
        
        # Close all remote connections
        for remote in self.remotes:
            try:
                remote.close()
            except Exception as e:
                print(f"[WARNING] VecEnv.close: Failed to close remote connection: {e}")
        
        print(f"[VecEnv] Successfully closed {len(self.processes)} worker processes")
 

    
def worker(rank, remote, parent_remote, env_name, env_fn_wrapper, captioner_fn_wrapper):
    random.seed(rank)
    np.random.seed(rank)
    
    parent_remote.close()
    
    # Memory tracking
    mem_start = get_process_memory_mb()
    print(f"[Worker {rank}] Memory at start: {mem_start:.2f} MB")
    
    try:
        env = env_fn_wrapper.x()
        mem_after_env = get_process_memory_mb()
        print(f"[Worker {rank}] Memory after env creation: {mem_after_env:.2f} MB (delta: +{mem_after_env - mem_start:.2f} MB)")
    except Exception as e:
        print(f"[ERROR] Worker {rank}: Failed to create environment: {e}")
        raise
    
    try:
        captioner = captioner_fn_wrapper.x()
        mem_after_captioner = get_process_memory_mb()
        print(f"[Worker {rank}] Memory after captioner creation: {mem_after_captioner:.2f} MB (delta: +{mem_after_captioner - mem_after_env:.2f} MB)")
        print(f"[Worker {rank}] Total memory used: {mem_after_captioner:.2f} MB")
    except Exception as e:
        print(f"[ERROR] Worker {rank}: Failed to create captioner: {e}")
        raise
    
    image = None
    reset_count = 0
    
    def env_step(action):
        try:
            full_action, extracted_action, executed_action, is_valid, metrics = env.extract_action(action)
            env_obs, reward, terminated, truncated, info = env.step(executed_action, is_valid)
            info["action_was_valid"] = is_valid
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
            mem_after_reset = get_process_memory_mb()
            print(f"[Worker {rank}] Memory after first reset: {mem_after_reset:.2f} MB")
        
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
                
            elif cmd == 'close':
                env.close()
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

        
  
    
