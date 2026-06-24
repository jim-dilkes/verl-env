# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-Environment Evaluator for RayPPOTrainer.

This module provides a flexible evaluation system that can run evaluations
across multiple, distinct environments as specified in the configuration.
"""

import json
import logging
import numpy as np
import time
import gc
import psutil
import os
from collections import Counter

logger = logging.getLogger(__name__)
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Callable, Set
from verl import DataProto
from verl.utils.tracking import ValidationGenerationsLogger

from verl.envs.environments import get_action_extraction_fn
from verl.envs.environments.focus_instructions import (
    has_focus_instructions,
    get_focus_instructions,
    has_ice_instructions,
    get_ice_instructions,
    sample_focus_for_episode,
    inject_focus_into_obs,
)
from verl.envs.vec_env import VecEnv


class VecEnvContextManager:
    """Context manager to ensure proper cleanup of vectorized environments.

    .. deprecated::
        This class is deprecated. Use MultiEnvEvaluator with VecEnv pooling instead,
        which prewarms worker processes early to avoid late-fork deadlocks.
        See MultiEnvEvaluator.prewarm() and hard_reset() for the new pattern.
    """

    def __init__(self, env_name, task, config, render_mode=None):
        import warnings
        warnings.warn(
            "VecEnvContextManager is deprecated. Use MultiEnvEvaluator with VecEnv pooling instead. "
            "The new pattern prewarms worker processes early via prewarm() and reconfigures them "
            "via hard_reset(), avoiding late-fork deadlocks.",
            DeprecationWarning,
            stacklevel=2
        )
        self.env_name = env_name
        self.task = task
        self.config = config
        self.render_mode = render_mode
        self.val_env = None

    def __enter__(self):
        self.val_env = make_vec_env(
            self.env_name,
            self.task,
            self.config,
            render_mode=self.render_mode
        )
        return self.val_env

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.val_env is not None:
            try:
                self.val_env.close()
                logger.debug(f"[VecEnvContextManager] Closed environment {self.env_name}")
            except Exception as e:
                logger.warning(f"[VecEnvContextManager] Failed to close environment {self.env_name}: {e}")
        return False  # Don't suppress exceptions


def make_vec_env(env_name, task, config, render_mode=None):
    """
    Create a vectorized environment.

    This function is copied from ray_trainer.py to ensure the evaluator
    can create new vectorized environments independently.
    """
    from verl.envs.environments import make_env
    from verl.envs.captioners import make_captioner
    from verl.envs.vec_env import VecEnv

    def get_env_fn(rank):
        def init_env():
            env = make_env(env_name, task, config, render_mode=render_mode)
            return env
        return init_env

    def get_captioner_fn(rank):
        def init_captioner():
            return make_captioner(config)
        return init_captioner

    env_fns = [get_env_fn(i) for i in range(config.envs.n_rollouts)]
    captioner_fns = [get_captioner_fn(i) for i in range(config.envs.n_rollouts)]

    env = VecEnv(
        env_name=env_name,
        config=config,
        env_fns=env_fns,
        captioner_fns=captioner_fns,
    )
    return env


class MultiEnvEvaluator:
    """
    Multi-Environment Evaluator for running evaluations across multiple environments.
    
    This class encapsulates the complexity of running evaluations across multiple,
    distinct environments as specified in the configuration. It provides a clean
    interface for the RayPPOTrainer to perform multi-environment evaluation.
    """
    
    def __init__(self, config, tokenizer, actor_rollout_wg, val_reward_fn, eval_config=None, debug: Optional[bool] = None):
        """
        Initialize the MultiEnvEvaluator.
        
        Args:
            config: The main configuration object
            tokenizer: The tokenizer for text processing
            actor_rollout_wg: The shared policy runner used for generation
            val_reward_fn: The validation reward function
            eval_config: The evaluation configuration (if not provided, will try to get from config.evaluation)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.val_reward_fn = val_reward_fn

        # Controls noisy debug prints like "[MultiEnvEvaluator] ...".
        # Default is OFF; can be enabled via env var or constructor arg.
        self._debug = self._parse_env_flag(
            "VERL_MULTIENV_EVALUATOR_DEBUG",
            default=False,
            override=debug,
        )
        
        # Initialize validation generations logger
        self.validation_generations_logger = ValidationGenerationsLogger()
        
        # Extract evaluation environments from config
        if eval_config is not None:
            self.eval_environments = eval_config.environments
        else:
            self.eval_environments = getattr(config.evaluation, 'environments', [])
        
        if not self.eval_environments:
            raise ValueError("No evaluation environments found in evaluation config")
        
        self._dbg_print(f"MultiEnvEvaluator initialized with {len(self.eval_environments)} evaluation environments")
        for i, env_config in enumerate(self.eval_environments):
            self._dbg_print(f"  Environment {i}: {env_config.get('name', f'env_{i}')} ")

        # VecEnv pool for reuse across evaluations (keyed by worker count)
        # This avoids late-fork deadlocks by creating workers early
        self._pool_by_worker_count: Dict[int, VecEnv] = {}

        # Extract pooling config from eval_config (with defaults per spec)
        pooling_config = getattr(eval_config, 'vecenv_pooling', None) if eval_config else None
        if pooling_config is None:
            # Also check config.evaluation.vecenv_pooling
            if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'vecenv_pooling'):
                pooling_config = config.evaluation.vecenv_pooling

        # Config flags with defaults (spec: all default to True)
        self._pool_enabled = True
        self._pool_prewarm = True
        self._pool_fail_if_missing = True

        if pooling_config is not None:
            self._pool_enabled = getattr(pooling_config, 'enabled', True)
            self._pool_prewarm = getattr(pooling_config, 'prewarm', True)
            self._pool_fail_if_missing = getattr(pooling_config, 'fail_if_missing_pool', True)

        logger.info(
            f"[MultiEnvEvaluator] VecEnv pooling config: enabled={self._pool_enabled}, "
            f"prewarm={self._pool_prewarm}, fail_if_missing={self._pool_fail_if_missing}"
        )

    @staticmethod
    def _parse_env_flag(name: str, default: bool, override: Optional[bool] = None) -> bool:
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

    def _dbg_print(self, msg: str) -> None:
        if self._debug:
            print(msg, flush=True)

    def prewarm(self) -> None:
        """
        Pre-create VecEnv worker processes before heavy runtime init.

        This must be called BEFORE Ray/torch distributed/vLLM initialization
        to avoid late-fork deadlocks. Creates one VecEnv pool entry per
        distinct worker count needed by evaluation configs.

        The pool entries use a neutral config (workers will be reconfigured
        via hard_reset before actual evaluation).
        """
        if not self._pool_enabled:
            logger.info("[MultiEnvEvaluator] Pooling disabled, skipping prewarm")
            return

        if not self._pool_prewarm:
            logger.info("[MultiEnvEvaluator] Prewarm disabled, pools will be created lazily (may cause late-fork issues)")
            return

        # Compute all distinct worker counts needed
        worker_counts: Set[int] = set()
        for env_config in self.eval_environments:
            n_rollouts = env_config.get('n_rollouts')
            if n_rollouts is None or n_rollouts <= 0:
                continue
            batch_size = env_config.get('batch_size')
            if batch_size is not None and batch_size > 0 and batch_size < n_rollouts:
                # Batched eval: VecEnv has batch_size workers
                worker_counts.add(batch_size)
            else:
                # Non-batched: VecEnv has n_rollouts workers
                worker_counts.add(n_rollouts)

        if not worker_counts:
            logger.info("[MultiEnvEvaluator] No worker counts to prewarm")
            return

        logger.info(f"[MultiEnvEvaluator] Prewarming VecEnv pools for worker counts: {sorted(worker_counts)}")

        # Use first eval env's config as base for prewarm (will be hard_reset before use)
        base_env_config = self.eval_environments[0]
        base_env_name = base_env_config.get('env_name', self.config.envs.env_name)
        base_task = base_env_config.get('task', getattr(self.config.envs, 'task', None))

        for worker_count in sorted(worker_counts):
            if worker_count in self._pool_by_worker_count:
                logger.debug(f"[MultiEnvEvaluator] Pool for worker_count={worker_count} already exists, skipping")
                continue

            start_time = time.perf_counter()

            # Create temp config with this worker count
            temp_config = self._create_env_config(base_env_config, n_rollouts_override=worker_count)

            # Create VecEnv (this forks worker processes early, before heavy runtime init)
            vec_env = make_vec_env(
                base_env_name,
                base_task,
                temp_config,
                render_mode=None,
            )

            self._pool_by_worker_count[worker_count] = vec_env

            elapsed = time.perf_counter() - start_time
            logger.info(f"[MultiEnvEvaluator] Prewarmed pool for worker_count={worker_count} in {elapsed:.2f}s")

        logger.info(f"[MultiEnvEvaluator] Prewarm complete. Pool sizes: {list(self._pool_by_worker_count.keys())}")

    def close(self) -> None:
        """Close all pooled VecEnvs and release resources."""
        if not self._pool_by_worker_count:
            return

        logger.info(f"[MultiEnvEvaluator] Closing {len(self._pool_by_worker_count)} pooled VecEnv(s)")

        for worker_count, vec_env in list(self._pool_by_worker_count.items()):
            try:
                vec_env.close()
                logger.debug(f"[MultiEnvEvaluator] Closed pool for worker_count={worker_count}")
            except Exception as e:
                logger.error(f"[MultiEnvEvaluator] Failed to close pool for worker_count={worker_count}: {e}")

        self._pool_by_worker_count.clear()
        logger.info("[MultiEnvEvaluator] All pooled VecEnvs closed")

    def _get_pooled_vecenv(self, worker_count: int) -> Optional[VecEnv]:
        """Get a VecEnv from the pool, or None if not available."""
        return self._pool_by_worker_count.get(worker_count)

    def evaluate(self, global_step):
        """
        Run evaluation across all configured environments.
        
        Args:
            global_step: Current global step for logging purposes
            
        Returns:
            dict: Combined metrics from all evaluation environments
        """
        self._dbg_print(f"[MultiEnvEvaluator] Starting evaluation at global_step={global_step}")
        self._dbg_print(f"[MultiEnvEvaluator] Number of environments to evaluate: {len(self.eval_environments)}")
        
        # Log initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        self._dbg_print(f"[MultiEnvEvaluator] Initial memory usage: {initial_memory:.1f} MB")
        
        all_metrics = {}
        
        for env_idx, env_config in enumerate(self.eval_environments):
            eval_name = env_config.get('name', f'env_{env_idx}')
            self._dbg_print(f"Evaluating environment: {eval_name}")

            start_time = time.perf_counter()

            # Run evaluation (VecEnv creation now happens inside _evaluate_single_env_body)
            try:
                env_metrics, episode_data = self._evaluate_single_env(env_config, eval_name)
                self._maybe_log_episode_generation(episode_data, eval_name, global_step)

                end_time = time.perf_counter()
                eval_time = end_time - start_time

                prefixed_metrics = {}
                for key, value in env_metrics.items():
                    prefixed_key = f"eval_{eval_name}/{key}"
                    prefixed_metrics[prefixed_key] = value

                # Extract timing components for metrics and console
                inference_time = env_metrics.get("inference_time_seconds", 0.0)
                env_step_time = env_metrics.get("env_step_time_seconds", 0.0)
                entropy_probe_time = env_metrics.get("action_entropy_probe_time_seconds", 0.0)
                vecenv_create_time = env_metrics.get("vecenv_create_time_seconds", 0.0)
                other_time = max(0.0, eval_time - inference_time - env_step_time - entropy_probe_time - vecenv_create_time)

                # Add timing metrics (total wall time includes episode logging)
                prefixed_metrics[f"eval_{eval_name}/eval_time_seconds"] = eval_time
                prefixed_metrics[f"eval_{eval_name}/other_time_seconds"] = other_time

                all_metrics.update(prefixed_metrics)
                self._dbg_print(f"[MultiEnvEvaluator] Added {len(prefixed_metrics)} metrics for {eval_name}")
                self._dbg_print(f"[MultiEnvEvaluator] Sample metrics for {eval_name}: {list(prefixed_metrics.keys())[:5]}")

                # Print timing breakdown to console
                timing_parts = [f"total: {eval_time:.2f}s", f"inference: {inference_time:.2f}s", f"env_step: {env_step_time:.2f}s"]
                if entropy_probe_time > 0:
                    timing_parts.append(f"entropy_probe: {entropy_probe_time:.2f}s")
                timing_parts.append(f"vecenv_create: {vecenv_create_time:.2f}s")
                timing_parts.append(f"other: {other_time:.2f}s")
                print(f"Completed evaluation for {eval_name} ({', '.join(timing_parts)})")

            except Exception as e:
                logger.error(f"[MultiEnvEvaluator] Failed to evaluate environment {eval_name}: {e}")
                import traceback
                logger.error(f"[MultiEnvEvaluator] Traceback: {traceback.format_exc()}")
                # Don't raise — continue evaluating remaining environments.
                # Failed env metrics are simply missing from the results.
                # The trainer-level try/except provides the final safety net.

            # Always run gc after each environment eval, even on failure,
            # to free memory before the next environment evaluation
            gc.collect()
            
            # Log memory usage after each environment
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            self._dbg_print(
                f"[MultiEnvEvaluator] Memory usage after {eval_name}: {current_memory:.1f} MB "
                f"(delta: {current_memory - initial_memory:+.1f} MB)"
            )
        
        # Final memory cleanup and logging
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        self._dbg_print(
            f"[MultiEnvEvaluator] Final memory usage: {final_memory:.1f} MB "
            f"(total delta: {final_memory - initial_memory:+.1f} MB)"
        )
        
        self._dbg_print(f"[MultiEnvEvaluator] Evaluation completed. Total metrics collected: {len(all_metrics)}")
        self._dbg_print(f"[MultiEnvEvaluator] All metric keys: {list(all_metrics.keys())}")
        self._dbg_print(f"[MultiEnvEvaluator] Sample metric values: {dict(list(all_metrics.items())[:3])}")
        
        return all_metrics
    
    def _create_env_config(self, env_config, n_rollouts_override: Optional[int] = None):
        """
        Create a temporary config object for a specific environment.

        Args:
            env_config: Environment-specific configuration
            n_rollouts_override: If provided, use this instead of env_config['n_rollouts'].
                               Used for batched evaluation where each batch has fewer rollouts.

        Returns:
            OmegaConf object: Temporary config for the environment
        """
        from omegaconf import OmegaConf, open_dict

        n_rollouts = n_rollouts_override if n_rollouts_override is not None else env_config['n_rollouts']

        self._dbg_print(f"[MultiEnvEvaluator] Creating config for environment: {env_config.get('name', 'unknown')}")
        self._dbg_print(f"[MultiEnvEvaluator] Original env_config: {env_config}")
        self._dbg_print(f"[MultiEnvEvaluator] Training config n_rollouts: {self.config.envs.n_rollouts}")
        self._dbg_print(f"[MultiEnvEvaluator] Evaluation n_rollouts: {n_rollouts} (override={n_rollouts_override})")
        
        # Create a proper copy using OmegaConf methods to avoid struct mode issues
        # Convert to container, modify, then recreate OmegaConf object
        temp_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        
        # Use open_dict to allow modifications even if struct mode was enabled
        with open_dict(temp_config):
            # Override environment-specific settings
            temp_config.envs.n_rollouts = n_rollouts
            temp_config.envs.episode_length = env_config['episode_length']
            temp_config.envs.env_name = env_config['env_name']
            temp_config.envs.task = env_config.get('task', None)  # Set the task from env_config
            temp_config.envs.freeze_completed_episodes = env_config.get('freeze_completed_episodes', False)
            temp_config.envs.duplication_mode = env_config.get('duplication_mode', 'none')
            temp_config.envs.format_penalty = env_config.get('format_penalty', 0.0)
            temp_config.envs.binary_reward = env_config.get('binary_reward', False)
            
            # Handle instruction_prompt from evaluation config
            # All env factories read from prompt.prompt.environment_instruction
            if 'instruction_prompt' in env_config:
                temp_config.prompt.prompt.environment_instruction = env_config['instruction_prompt']
                self._dbg_print(
                    f"[MultiEnvEvaluator] Set environment_instruction from eval config (length: {len(env_config['instruction_prompt']) if env_config['instruction_prompt'] else 0} chars)"
                )
            
            self._dbg_print(
                f"[MultiEnvEvaluator] After basic overrides - n_rollouts: {temp_config.envs.n_rollouts}, task: {temp_config.envs.task}, env_name: {temp_config.envs.env_name}"
            )
            
            # Handle captioner configuration
            if 'captioner' in env_config:
                self._dbg_print(f"[MultiEnvEvaluator] Original captioner config: {self.config.envs.captioner}")
                self._dbg_print(f"[MultiEnvEvaluator] Environment captioner config: {env_config['captioner']}")
                
                # Merge captioner config with defaults to ensure all required fields are present
                captioner_config = OmegaConf.to_container(self.config.envs.captioner)
                captioner_config.update(env_config['captioner'])
                temp_config.envs.captioner = captioner_config
                
                self._dbg_print(f"[MultiEnvEvaluator] Final captioner config: {temp_config.envs.captioner}")
            
            # Handle environment-specific kwargs
            env_name = env_config['env_name']
            if f'{env_name}_kwargs' in env_config:
                self._dbg_print(f"[MultiEnvEvaluator] Setting {env_name}_kwargs: {env_config[f'{env_name}_kwargs']}")
                temp_config.envs[f'{env_name}_kwargs'] = env_config[f'{env_name}_kwargs']
            
            # Set initial seed if specified
            if 'initial_seed' in env_config:
                temp_config.envs.group_initial_seed = env_config['initial_seed']
                self._dbg_print(f"[MultiEnvEvaluator] Set initial_seed: {env_config['initial_seed']}")

            # Handle prompt.prompt overrides for evaluation
            if hasattr(temp_config, 'prompt') and hasattr(temp_config.prompt, 'prompt'):
                original_epsilon = getattr(temp_config.prompt.prompt, 'epsilon', 0.0)
                original_multi_action = getattr(temp_config.prompt.prompt, 'multi_action_reasoning', False)

                # Check for inherit_training_multiaction (inherits both epsilon and multi_action_reasoning)
                # Also support legacy inherit_training_epsilon for backwards compatibility
                inherit_training = env_config.get('inherit_training_multiaction', False) or env_config.get('inherit_training_epsilon', False)

                # Epsilon handling:
                # - If 'epsilon' explicitly set in eval config, use that value
                # - If inherit_training_multiaction: true, keep training epsilon (for entropy/exploration metrics)
                # - Otherwise, force epsilon=0 for evaluation
                if 'epsilon' in env_config:
                    temp_config.prompt.prompt.epsilon = env_config['epsilon']
                    self._dbg_print(f"[MultiEnvEvaluator] Epsilon explicitly set in eval config: {env_config['epsilon']}")
                elif inherit_training:
                    # Keep original training epsilon for exploration measurement
                    self._dbg_print(f"[MultiEnvEvaluator] Inheriting training epsilon: {original_epsilon}")
                else:
                    temp_config.prompt.prompt.epsilon = 0.0
                    if original_epsilon > 0:
                        self._dbg_print(f"[MultiEnvEvaluator] Forcing epsilon=0 for evaluation (training had epsilon={original_epsilon})")

                # multi_action_reasoning handling:
                # - If explicitly set in eval config, use that value
                # - If inherit_training_multiaction: true, keep training value
                # - Otherwise, force to False for evaluation
                if 'multi_action_reasoning' in env_config:
                    temp_config.prompt.prompt.multi_action_reasoning = env_config['multi_action_reasoning']
                    self._dbg_print(f"[MultiEnvEvaluator] multi_action_reasoning set from eval config: {env_config['multi_action_reasoning']}")
                elif inherit_training:
                    self._dbg_print(f"[MultiEnvEvaluator] Inheriting training multi_action_reasoning: {original_multi_action}")
                else:
                    temp_config.prompt.prompt.multi_action_reasoning = False
                    if original_multi_action:
                        self._dbg_print(f"[MultiEnvEvaluator] Forcing multi_action_reasoning=False for evaluation (training had multi_action_reasoning={original_multi_action})")
        
        self._dbg_print(f"[MultiEnvEvaluator] Final temp_config.envs.n_rollouts: {temp_config.envs.n_rollouts}")
        self._dbg_print(f"[MultiEnvEvaluator] Final temp_config.envs keys: {list(temp_config.envs.keys())}")
        return temp_config
    
    def _get_generation_config(self, env_config):
        """
        Get generation configuration for an environment.
        
        If generation parameters are specified in env_config, use those.
        Otherwise, fall back to val_kwargs from the main config.
        
        Args:
            env_config: Environment-specific configuration
            
        Returns:
            dict: Generation parameters to set in meta_info
        """
        gen_config = {}
        
        # Check if generation config is specified in env_config
        if 'generation' in env_config:
            gen_config_from_env = env_config['generation']
            self._dbg_print(f"[MultiEnvEvaluator] Using generation config from env_config: {gen_config_from_env}")
            
            # Map generation parameters to meta_info keys
            # Note: vLLM supports min_p but not top_a
            if 'temperature' in gen_config_from_env:
                gen_config['temperature'] = gen_config_from_env['temperature']
            if 'top_p' in gen_config_from_env:
                gen_config['top_p'] = gen_config_from_env['top_p']
            if 'top_k' in gen_config_from_env:
                gen_config['top_k'] = gen_config_from_env['top_k']
            if 'min_p' in gen_config_from_env:
                gen_config['min_p'] = gen_config_from_env['min_p']
            if 'do_sample' in gen_config_from_env:
                gen_config['do_sample'] = gen_config_from_env['do_sample']
            if 'n' in gen_config_from_env:
                gen_config['n'] = gen_config_from_env['n']
        else:
            # Fall back to val_kwargs from main config
            if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout'):
                rollout_config = self.config.actor_rollout_ref.rollout
                if hasattr(rollout_config, 'val_kwargs'):
                    val_kwargs = rollout_config.val_kwargs
                    self._dbg_print(f"[MultiEnvEvaluator] Using generation config from val_kwargs: {val_kwargs}")
                    
                    if hasattr(val_kwargs, 'temperature'):
                        gen_config['temperature'] = val_kwargs.temperature
                    if hasattr(val_kwargs, 'top_p'):
                        gen_config['top_p'] = val_kwargs.top_p
                    if hasattr(val_kwargs, 'top_k'):
                        gen_config['top_k'] = val_kwargs.top_k
                    if hasattr(val_kwargs, 'min_p'):
                        gen_config['min_p'] = val_kwargs.min_p
                    if hasattr(val_kwargs, 'do_sample'):
                        gen_config['do_sample'] = val_kwargs.do_sample
                    if hasattr(val_kwargs, 'n'):
                        gen_config['n'] = val_kwargs.n
        
        # Always set validate flag
        gen_config['validate'] = True
        
        self._dbg_print(f"[MultiEnvEvaluator] Final generation config: {gen_config}")
        return gen_config
    
    def _maybe_log_episode_generation(self, episode_data, env_name, global_step):
        """
        Log a full episode generation to the configured logger.
        
        Args:
            episode_data: Dictionary containing episode information (inputs, outputs, total_score)
            env_name: Name of the environment
            global_step: Current global step for logging
        """
        # Check if logging is enabled (similar to ray_trainer logic)
        log_generations = getattr(self.config.trainer, 'log_val_generations', 0)
        if log_generations == 0:
            return
        
        if episode_data is None:
            self._dbg_print(f"[MultiEnvEvaluator] No episode data to log for {env_name}")
            return
        
        # Format the episode data for logging
        formatted_input = self._format_episode(episode_data['inputs'])
        formatted_output = self._format_episode(episode_data['outputs'])
        total_score = episode_data['total_score']
        total_reward = episode_data.get('total_reward')
        max_length_steps = episode_data.get('max_length_steps', [])

        score_log_value = "N/A" if total_score is None else total_score
        reward_log_value = "N/A" if total_reward is None else total_reward
        max_length_steps_log_value = str(max_length_steps)  # [] if none, else list of step indices

        if total_score is None:
            self._dbg_print(f"[MultiEnvEvaluator] Episode score unavailable for {env_name}; logging 'N/A'.")

        # Create sample tuple (input, output, score, reward, max_length_steps)
        sample = (formatted_input, formatted_output, score_log_value, reward_log_value, max_length_steps_log_value)
        
        # Log to each configured logger
        logger_backends = getattr(self.config.trainer, 'logger', ['console'])
        self.validation_generations_logger.log(logger_backends, [sample], global_step, table_name=f"eval_{env_name}_gen/generations")
        
        self._dbg_print(f"[MultiEnvEvaluator] Logged episode generation for {env_name} at step {global_step}")
    
    def _format_episode(self, ep_content_list):
        """
        Format episode inputs into a readable string.
        
        Args:
            inputs: List of input observations for each step
            
        Returns:
            str: Formatted input string
        """
        if not ep_content_list:
            return "No inputs recorded"
        
        formatted_steps = []
        for i, text in enumerate(ep_content_list):
            formatted_steps.append(f"---\nStep {i+1}\n---\n{text}")
        
        return "\n\n---\n\n".join(formatted_steps)

    def _extract_from_info(self, infos: List[Dict], key: str, as_array: bool = False, array_dtype: Optional[np.dtype] = None, default: Any = None) -> Optional[List[Any]]:
        """
        Extract a list from info dictionary. Default value is used if the key is not found in the info.
        """
        if infos is None:
            raise ValueError(f"[MultiEnvEvaluator] infos is None")
        
        if not isinstance(infos, (list, tuple)):
            raise ValueError(f"[MultiEnvEvaluator] infos is not a list or tuple")

        return_list = []
        for info in infos:
            if not isinstance(info, Mapping):
                raise ValueError(f"[MultiEnvEvaluator] info instance is not a dictionary")
            if key not in info:
                if default is None:
                    raise ValueError(f"[MultiEnvEvaluator] {key} not found in info")
                else:
                    return_list.append(default)
                    continue
            return_list.append(info[key])
        if as_array:
            return np.asarray(return_list, dtype=array_dtype)
        else:
            return return_list
    
    def _evaluate_single_env(self, env_config, eval_name):
        """Run evaluation while preserving tokenizer state."""
        original_padding_side = getattr(self.tokenizer, "padding_side", None)
        self.tokenizer.padding_side = "left"
        try:
            return self._evaluate_single_env_body(env_config, eval_name)
        finally:
            if original_padding_side is not None:
                self.tokenizer.padding_side = original_padding_side

    def _evaluate_single_env_body(self, env_config, eval_name):
        """
        Run evaluation for a single environment with optional batching.

        Supports batched evaluation to prevent OOM: if batch_size is set in env_config,
        rollouts are processed in batches with VecEnv recreated per batch.

        Args:
            env_config: Environment-specific configuration
            eval_name: Name of the environment for logging

        Returns:
            tuple: (dict of metrics, dict episode_data)
        """
        max_seq_len = self.config.data.max_prompt_length
        n_rollouts = env_config.get('n_rollouts')
        if n_rollouts is None or n_rollouts <= 0:
            raise ValueError(f"[MultiEnvEvaluator] {eval_name}: n_rollouts must be > 0")

        batch_size = env_config.get('batch_size')
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"[MultiEnvEvaluator] {eval_name}: batch_size must be > 0 if provided")
        if batch_size is None or batch_size >= n_rollouts:
            batch_size = n_rollouts  # No batching needed

        initial_seed = env_config.get('initial_seed', None)
        if initial_seed is not None:
            initial_seed = int(initial_seed)
        seed_group_size = env_config.get('seed_group_size', n_rollouts)
        if seed_group_size is None:
            # Treat explicit None as the default (single group).
            seed_group_size = n_rollouts
        seed_group_size = int(seed_group_size)
        eval_env_name = env_config.get('env_name', None)
        if eval_env_name is None:
            raise ValueError(f"[MultiEnvEvaluator] {eval_name}: env_name is not specified")

        # Action entropy configuration
        action_entropy_cfg = env_config.get('action_entropy') or {}
        entropy_enabled = bool(action_entropy_cfg.get('enabled', False))
        entropy_cfg = None
        entropy_measure_steps: Sequence[int] = []

        if entropy_enabled:
            entropy_cfg = self._validate_action_entropy_config(action_entropy_cfg)
            entropy_measure_steps = self._compute_entropy_measurement_steps(
                entropy_cfg,
                env_config['episode_length'],
                initial_seed,
            )
            print(
                f"[MultiEnvEvaluator] Action entropy enabled for {eval_name} "
                f"with measurement steps: {sorted(entropy_measure_steps)}"
            )

        exclusive_entropy_metrics = bool(entropy_cfg['exclusive_metric']) if entropy_enabled else False
        track_standard_metrics = not (entropy_enabled and exclusive_entropy_metrics)

        # Get generation config for this environment (once per environment)
        self._current_gen_config = self._get_generation_config(env_config)

        # ICE focus injection (opt-in per eval env via inherit_ice)
        ice_config = getattr(self.config.prompt.prompt, 'ice', None) if hasattr(self.config, 'prompt') and hasattr(self.config.prompt, 'prompt') else None
        eval_ice_source = getattr(ice_config, 'source', 'specific') if ice_config else 'specific'
        eval_ice_enabled = (
            env_config.get('inherit_ice', False)
            and ice_config is not None
            and getattr(ice_config, 'enabled', False)
            and has_ice_instructions(eval_env_name, eval_ice_source)
        )
        if eval_ice_enabled:
            eval_ice_template = getattr(ice_config, 'template', '') if eval_ice_source == 'specific' else '{STEP_TEXT}'
            eval_ice_instructions = get_ice_instructions(eval_env_name, eval_ice_source)
            eval_ice_no_supp = getattr(ice_config, 'no_supplement_prob', None)
            if eval_ice_no_supp is None:
                raise ValueError(
                    "ice.no_supplement_prob must be set explicitly when ice.enabled=true. "
                    "Recommended: 0.125 (12.5% clean rollouts)."
                )
            eval_ice_no_supp = float(eval_ice_no_supp)
            # Per-eval-env override: ice_proportion = fraction WITH focus
            eval_ice_proportion = env_config.get('ice_proportion', None)
            if eval_ice_proportion is not None:
                eval_ice_no_supp = 1.0 - float(eval_ice_proportion)
            self._dbg_print(f"[MultiEnvEvaluator] ICE focus enabled for {eval_name} (no_supp={eval_ice_no_supp:.3f})")
        elif env_config.get('inherit_ice', False):
            logger.debug(
                "[MultiEnvEvaluator] inherit_ice=true for %s but ICE not active "
                "(enabled=%s, has_instructions=%s)",
                eval_name,
                ice_config is not None and getattr(ice_config, 'enabled', False),
                has_ice_instructions(eval_env_name, eval_ice_source) if ice_config else False,
            )

        # Validate seed_group_size
        if seed_group_size <= 0:
            raise ValueError(f"[MultiEnvEvaluator] {eval_name}: seed_group_size must be > 0")
        if n_rollouts % seed_group_size != 0:
            raise ValueError(f"n_rollouts must be divisible by seed_group_size")
        n_groups = n_rollouts // seed_group_size

        # Compute full seed sequence upfront for determinism
        all_seeds = self._compute_seed_sequence(initial_seed, n_rollouts, seed_group_size)

        # =========== ACCUMULATORS (global across all batches) ===========
        # Per-rollout arrays (will be concatenated across batches)
        all_rew_of_traj = []
        all_len_of_traj = []
        all_score_of_traj = []
        all_pos_rew_of_traj = []

        # Per-group state-action tracking (global indexing)
        group_state_action_texts_valid = [[] for _ in range(n_groups)]
        group_state_action_texts_all = [[] for _ in range(n_groups)]

        # Scalars accumulated across batches
        total_tokens_generated = 0
        total_inference_time = 0.0
        total_env_step_time = 0.0
        total_attempted_actions = 0
        total_valid_actions = 0

        # Episode tracking (first rollout only, from first batch)
        episode_inputs = []
        episode_outputs = []
        episode_total_score = None
        episode_total_reward = None
        episode_max_length_steps = []
        episode_tracked = False

        # Entropy accumulators
        entropy_measurements: List[float] = []
        entropy_unique_executed_actions_per_unique_text: List[float] = []
        entropy_unique_valid_actions_per_unique_valid_text: List[float] = []
        entropy_action_counter: Counter = Counter()
        entropy_unique_texts: List[int] = []
        entropy_unique_valid_texts: List[int] = []
        entropy_unique_executed_actions: List[int] = []
        entropy_unique_valid_actions: List[int] = []
        entropy_probe_time = 0.0

        # Track per-rollout output token counts for the *last generated step*.
        # This keeps toks_out_mean/std invariant to batch_size (concat batches -> n_rollouts).
        response_n_tokens_last_step: List[Optional[int]] = [None] * n_rollouts

        n_batches = (n_rollouts + batch_size - 1) // batch_size
        if n_batches > 1:
            print(f"[MultiEnvEvaluator] {eval_name}: Running {n_batches} batches (batch_size={batch_size}, n_rollouts={n_rollouts})")
            # VecEnv is reused across batches, so batch_size must evenly divide n_rollouts
            # (otherwise last batch would have fewer seeds than workers)
            if n_rollouts % batch_size != 0:
                raise ValueError(
                    f"[MultiEnvEvaluator] {eval_name}: n_rollouts ({n_rollouts}) must be evenly divisible "
                    f"by batch_size ({batch_size}) when using batched evaluation. "
                    f"Adjust batch_size to a divisor of n_rollouts."
                )

        # Timing accumulators for debugging batched evaluation performance
        total_vecenv_create_time = 0.0
        total_vecenv_reset_time = 0.0
        total_vecenv_close_time = 0.0
        total_gc_time = 0.0
        total_state_action_accum_time = 0.0
        total_tokenizer_time = 0.0

        # =========== GET OR CREATE VECENV ===========
        # Try to use pooled VecEnv first (avoids late-fork deadlock).
        # If pooled, use hard_reset to reconfigure for this eval env.
        temp_config = self._create_env_config(env_config, n_rollouts_override=batch_size)

        vecenv_create_start = time.perf_counter()
        vec_envs = self._get_pooled_vecenv(batch_size)
        used_pooled_vecenv = vec_envs is not None

        if used_pooled_vecenv:
            # Reconfigure pooled VecEnv for this evaluation
            self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: Using pooled VecEnv (worker_count={batch_size})")

            # Warn if eval env_name differs from training env_name (spec constraint)
            training_env_name = self.config.envs.env_name
            eval_env_name_for_reset = env_config['env_name']
            if eval_env_name_for_reset != training_env_name:
                logger.warning(
                    f"[MultiEnvEvaluator] {eval_name}: eval env_name '{eval_env_name_for_reset}' differs from "
                    f"training env_name '{training_env_name}'. Cross-env evaluation during training is not fully "
                    f"supported and may produce unexpected behavior."
                )

            try:
                vec_envs.hard_reset(
                    env_name=env_config['env_name'],
                    task=temp_config.envs.task,
                    config=temp_config,
                    render_mode=None
                )
            except RuntimeError as e:
                # hard_reset failed - VecEnv may be in mixed state
                if vec_envs.unusable:
                    # Evict from pool and close to prevent reuse of corrupted state
                    logger.error(
                        f"[MultiEnvEvaluator] {eval_name}: hard_reset failed, VecEnv marked unusable. "
                        f"Evicting from pool and closing."
                    )
                    del self._pool_by_worker_count[batch_size]
                    try:
                        vec_envs.close()
                    except Exception as close_err:
                        logger.warning(f"[MultiEnvEvaluator] {eval_name}: Failed to close unusable VecEnv: {close_err}")
                raise
        else:
            # No pooled VecEnv available
            if self._pool_fail_if_missing:
                # FAIL FAST: Do not late-fork. This would reintroduce the deadlock risk.
                raise RuntimeError(
                    f"[MultiEnvEvaluator] {eval_name}: No pooled VecEnv for worker_count={batch_size}. "
                    f"This would require late process creation which can deadlock. "
                    f"Ensure prewarm() was called before init_workers() and that batch_size ({batch_size}) "
                    f"matches one of the prewarmed worker counts: {list(self._pool_by_worker_count.keys())}. "
                    f"Check eval config n_rollouts/batch_size values. "
                    f"Set evaluation.vecenv_pooling.fail_if_missing_pool=false to allow late creation (risky)."
                )
            else:
                # FALLBACK: Late creation allowed (user explicitly opted in via config)
                logger.warning(
                    f"[MultiEnvEvaluator] {eval_name}: Creating VecEnv late (worker_count={batch_size}). "
                    f"This may cause deadlocks if fork is used after threads are started. "
                    f"Consider prewarming all needed worker counts."
                )
                vec_envs = make_vec_env(
                    env_config['env_name'],
                    temp_config.envs.task,
                    temp_config,
                    render_mode=None
                )
                # Don't add to pool - this is a one-off late creation
                used_pooled_vecenv = False
        vecenv_create_end = time.perf_counter()
        total_vecenv_create_time = vecenv_create_end - vecenv_create_start

        try:
            # =========== BATCH LOOP ===========
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_rollouts)
                batch_n = batch_end - batch_start
                batch_seeds = all_seeds[batch_start:batch_end]

                if n_batches > 1:
                    self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: Starting batch {batch_idx + 1}/{n_batches} (rollouts {batch_start}-{batch_end - 1})")

                # Reset with explicit seeds for this batch
                reset_start = time.perf_counter()
                obs_vec, info_vec = vec_envs.reset(seeds=batch_seeds)
                reset_end = time.perf_counter()
                total_vecenv_reset_time += (reset_end - reset_start)

                # Raw game state texts for deterministic state-action dedup
                current_game_state_texts = self._extract_from_info(info_vec, "game_state_text")

                # ICE: sample focus instructions for this batch
                if eval_ice_enabled:
                    batch_focus = sample_focus_for_episode(
                        batch_n, eval_ice_instructions, eval_ice_no_supp
                    )

                # Per-batch state
                pending_entropy_steps = set(entropy_measure_steps) if entropy_enabled else set()
                ever_terminated = np.zeros(batch_n, dtype=bool)  # Persistent: True once terminated, never resets
                active_rollouts = np.ones(batch_n, dtype=bool) if entropy_enabled else None
                end_of_traj = None
                rew_of_traj = 0.
                score_of_traj = None
                len_of_traj = 0.
                pos_rew_of_traj = None

                # Track per-rollout token counts: updated while active, frozen on termination
                # Use -1 as sentinel for "not yet set" (valid token counts are >= 0)
                batch_frozen_toks = np.full(batch_n, -1, dtype=np.int64)

                # Episode loop for this batch
                for step_idx in range(env_config['episode_length']):
                    tokenizer_start = time.perf_counter()
                    if eval_ice_enabled:
                        obs_for_gen = inject_focus_into_obs(obs_vec, batch_focus, eval_ice_template)
                    else:
                        obs_for_gen = obs_vec
                    val_input_obs_text = self.tokenizer.apply_chat_template(
                        obs_for_gen, tokenize=False, add_generation_prompt=True
                    )

                    # Entropy probing (if enabled and this step is a measurement step)
                    if entropy_enabled and step_idx in pending_entropy_steps:
                        if 'multi_action_reasoning' in env_config:
                            multi_action = env_config['multi_action_reasoning']
                        elif hasattr(self.config, 'prompt') and hasattr(self.config.prompt, 'prompt'):
                            multi_action = getattr(self.config.prompt.prompt, 'multi_action_reasoning', False)
                        else:
                            multi_action = False
                        action_extraction_fn = get_action_extraction_fn(eval_env_name, multi_action=multi_action)
                        (
                            entropies,
                            step_action_counts,
                            probe_time,
                            unique_texts_count,
                            unique_valid_texts_count,
                            unique_executed_actions_count,
                            unique_valid_actions_count,
                        ) = self._probe_action_entropy(
                            val_input_obs_text,
                            entropy_cfg,
                            active_rollouts,
                            max_seq_len,
                            action_extraction_fn=action_extraction_fn,
                        )
                        entropy_measurements.extend(entropies)
                        entropy_action_counter.update(step_action_counts)
                        entropy_probe_time += probe_time
                        if unique_texts_count > 0:
                            entropy_unique_executed_actions_per_unique_text.append(
                                float(unique_executed_actions_count) / float(unique_texts_count)
                            )
                        if unique_valid_texts_count > 0:
                            entropy_unique_valid_actions_per_unique_valid_text.append(
                                float(unique_valid_actions_count) / float(unique_valid_texts_count)
                            )
                        entropy_unique_texts.append(unique_texts_count)
                        entropy_unique_valid_texts.append(unique_valid_texts_count)
                        entropy_unique_executed_actions.append(unique_executed_actions_count)
                        entropy_unique_valid_actions.append(unique_valid_actions_count)
                        pending_entropy_steps.remove(step_idx)

                        if not track_standard_metrics and not pending_entropy_steps:
                            break

                    val_input_obs = self.tokenizer(
                        val_input_obs_text,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=max_seq_len
                    )
                    tokenizer_end = time.perf_counter()
                    total_tokenizer_time += (tokenizer_end - tokenizer_start)

                    input_ids = val_input_obs['input_ids']
                    attention_mask = val_input_obs['attention_mask']
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)

                    val_obs_data = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids,
                    }
                    val_gen_batch = DataProto.from_dict(tensors=val_obs_data)
                    val_gen_batch.meta_info["step"] = None

                    for key, value in self._current_gen_config.items():
                        val_gen_batch.meta_info[key] = value

                    inference_start = time.perf_counter()
                    val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
                    inference_end = time.perf_counter()
                    total_inference_time += (inference_end - inference_start)

                    response_ids = val_gen_batch_output.batch['responses']
                    full_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                    # Track first rollout for logging (only from first batch)
                    if track_standard_metrics and not episode_tracked and batch_idx == 0:
                        episode_inputs.append(val_input_obs_text[0])
                        episode_outputs.append(full_responses[0])
                        # Check if input was truncated (hit max_seq_len)
                        if attention_mask[0].sum().item() >= max_seq_len:
                            episode_max_length_steps.append(step_idx)

                    response_n_tokens = (response_ids != self.tokenizer.pad_token_id).sum(dim=-1)

                    # Only count tokens for active rollouts (not yet terminated)
                    active_this_step = ~ever_terminated
                    response_n_tokens_np = response_n_tokens.cpu().numpy()
                    total_tokens_generated += int((response_n_tokens_np * active_this_step).sum())

                    # Update frozen toks for active rollouts (will be frozen when they terminate)
                    batch_frozen_toks[active_this_step] = response_n_tokens_np[active_this_step]

                    try:
                        env_step_start = time.perf_counter()
                        obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = vec_envs.step(full_responses)
                        env_step_end = time.perf_counter()
                        total_env_step_time += (env_step_end - env_step_start)
                    except Exception as e:
                        logger.error(f"[MultiEnvEvaluator] Exception in val_env.step: {e}")
                        import traceback
                        logger.error(f"[MultiEnvEvaluator] Traceback: {traceback.format_exc()}")
                        raise

                    was_valid_list = self._extract_from_info(info_vec, "action_was_valid")
                    executed_actions = self._extract_from_info(info_vec, "executed_action_text", default='__ended_traj__')

                    if len(was_valid_list) != batch_n:
                        raise ValueError(f"[MultiEnvEvaluator] {eval_name}: len(was_valid_list) != batch_n")
                    if len(executed_actions) != batch_n:
                        raise ValueError(f"[MultiEnvEvaluator] {eval_name}: len(executed_actions) != batch_n")

                    use_end_of_traj = end_of_traj if end_of_traj is not None else np.zeros(batch_n, dtype=bool)

                    self._dbg_print(
                        f"[MultiEnvEvaluator] {eval_name}: batch {batch_idx} step {step_idx} use_end_of_traj={use_end_of_traj.tolist()}"
                    )

                    total_attempted_actions_this_step = batch_n - use_end_of_traj.sum()
                    total_attempted_actions += total_attempted_actions_this_step
                    total_valid_actions_this_step = sum(1 for v, e in zip(was_valid_list, use_end_of_traj) if v and not e)
                    total_valid_actions += total_valid_actions_this_step

                    # Accumulate state-action texts with GLOBAL indexing (with timing)
                    # Uses raw game state text (deterministic) instead of chat-template obs
                    # (which includes stochastic reasoning history from max_cot_history)
                    state_action_accum_start = time.perf_counter()
                    for local_idx, (executed_action, was_valid_action, rollout_already_ended) in enumerate(
                        zip(executed_actions, was_valid_list, use_end_of_traj)
                    ):
                        global_idx = batch_start + local_idx
                        group_idx = global_idx // seed_group_size
                        state_text = current_game_state_texts[local_idx]
                        if not state_text:
                            raise ValueError(
                                f"[MultiEnvEvaluator] {eval_name}: game_state_text empty for rollout {global_idx}. "
                                f"Ensure env provides text obs via env_obs['text']['long_term_context']."
                            )
                        if was_valid_action and not rollout_already_ended:
                            group_state_action_texts_valid[group_idx].append(f"{state_text} {executed_action}")
                        if not rollout_already_ended:
                            group_state_action_texts_all[group_idx].append(f"{state_text} {executed_action}")
                    state_action_accum_end = time.perf_counter()
                    total_state_action_accum_time += (state_action_accum_end - state_action_accum_start)

                    # Update game state texts for next step (step returns NEW state)
                    current_game_state_texts = self._extract_from_info(info_vec, "game_state_text")

                    # Update persistent termination tracking (never resets after auto-reset)
                    done_mask = np.logical_or(terminated_vec, truncated_vec)
                    ever_terminated = np.logical_or(ever_terminated, done_mask)

                    # Update entropy probing mask from persistent tracking
                    if entropy_enabled and active_rollouts is not None:
                        active_rollouts = np.logical_not(ever_terminated)

                    if track_standard_metrics:
                        try:
                            score_values = self._extract_from_info(info_vec, "score", as_array=True, array_dtype=np.float64)
                        except ValueError:
                            score_values = None

                        if end_of_traj is None:
                            end_of_traj = np.logical_or(terminated_vec, truncated_vec)
                            rew_of_traj = reward_vec
                            len_of_traj = np.ones_like(reward_vec)
                            pos_rew_of_traj = (np.array(reward_vec) > 0.) * 1.
                            if score_values is not None:
                                score_of_traj = score_values
                        else:
                            done = np.logical_or(terminated_vec, truncated_vec)
                            prev_end_of_traj = end_of_traj.copy()
                            active_mask = (~prev_end_of_traj).astype(np.float32)
                            rew_of_traj += reward_vec * active_mask
                            len_of_traj += active_mask
                            end_of_traj = np.logical_or(prev_end_of_traj, done)
                            pos_rew_of_traj = np.logical_or(pos_rew_of_traj, (np.array(reward_vec) > 0.) * 1.)
                            if score_values is not None:
                                if score_of_traj is None:
                                    score_of_traj = score_values
                                else:
                                    # Update scores for rollouts that were active at the start of this step,
                                    # including rollouts that become done on this step.
                                    score_of_traj = np.where(~prev_end_of_traj, score_values, score_of_traj)

                        # Episode tracking (first batch, first rollout)
                        if not episode_tracked and batch_idx == 0:
                            if score_of_traj is not None:
                                episode_total_score = float(score_of_traj[0])
                            if terminated_vec[0] or truncated_vec[0] or step_idx == env_config['episode_length'] - 1:
                                # rew_of_traj may be scalar 0. on step 0 or array on later steps
                                if isinstance(rew_of_traj, np.ndarray):
                                    episode_total_reward = float(rew_of_traj[0])
                                else:
                                    episode_total_reward = float(rew_of_traj)
                                episode_tracked = True

                        if end_of_traj.all():
                            break

                # End of episode loop for this batch
                # Accumulate per-rollout results
                if track_standard_metrics and end_of_traj is not None:
                    all_rew_of_traj.extend(np.array(rew_of_traj, dtype=np.float64).tolist())
                    all_len_of_traj.extend(np.array(len_of_traj, dtype=np.float64).tolist())
                    all_pos_rew_of_traj.extend(np.array(pos_rew_of_traj, dtype=np.float64).tolist())
                    if score_of_traj is not None:
                        all_score_of_traj.extend(np.array(score_of_traj, dtype=np.float64).tolist())

                # Capture frozen token counts (frozen at termination, or last step if never terminated)
                # -1 sentinel means "not set" (rollout never ran a step, shouldn't happen)
                for i in range(batch_n):
                    if batch_frozen_toks[i] >= 0:
                        response_n_tokens_last_step[batch_start + i] = int(batch_frozen_toks[i])

                # Memory cleanup between batches (with timing)
                gc_start = time.perf_counter()
                gc.collect()
                gc_end = time.perf_counter()
                total_gc_time += (gc_end - gc_start)

                if n_batches > 1:
                    self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: Completed batch {batch_idx + 1}/{n_batches}")

            # =========== END BATCH LOOP ===========

        finally:
            # Pooled VecEnvs are NOT closed here - they're reused across evaluations
            # and closed by MultiEnvEvaluator.close() at trainer shutdown.
            # But late-created (non-pooled) VecEnvs MUST be closed to avoid leaks.
            vecenv_close_start = time.perf_counter()
            if not used_pooled_vecenv and vec_envs is not None:
                try:
                    vec_envs.close()
                    logger.debug(f"[MultiEnvEvaluator] {eval_name}: Closed late-created VecEnv")
                except Exception as e:
                    logger.warning(f"[MultiEnvEvaluator] {eval_name}: Failed to close late-created VecEnv: {e}")
            vecenv_close_end = time.perf_counter()
            total_vecenv_close_time = vecenv_close_end - vecenv_close_start

        # =========== METRIC COMPUTATION (from accumulated data) ===========
        metric_dict: Dict[str, float] = {}

        if track_standard_metrics and all_rew_of_traj:
            rew_of_traj_arr = np.array(all_rew_of_traj, dtype=np.float64)
            len_of_traj_arr = np.array(all_len_of_traj, dtype=np.float64)
            pos_rew_of_traj_arr = np.array(all_pos_rew_of_traj, dtype=np.float64)
            response_n_tokens_float = None
            if any(x is not None for x in response_n_tokens_last_step):
                response_n_tokens_float = np.array(
                    [x for x in response_n_tokens_last_step if x is not None],
                    dtype=np.float64,
                )

            metric_dict.update({
                "rewards_mean": float(rew_of_traj_arr.mean()),
                "rewards_std": float(rew_of_traj_arr.std()),
                "pos_reward_any_prop_mean": float(pos_rew_of_traj_arr.mean()),
                "pos_reward_any_prop_std": float(pos_rew_of_traj_arr.std()),
                "traj_length_mean": float(len_of_traj_arr.mean()),
                "traj_length_std": float(len_of_traj_arr.std()),
            })

            if all_score_of_traj:
                score_arr = np.array(all_score_of_traj, dtype=np.float64)
                metric_dict.update({
                    "score_mean": float(score_arr.mean()),
                    "score_std": float(score_arr.std()),
                })

            if response_n_tokens_float is not None:
                metric_dict.update({
                    "toks_out_mean": float(response_n_tokens_float.mean()),
                    "toks_out_std": float(response_n_tokens_float.std()),
                })

            metric_dict.update({
                "tokens_per_rollout": total_tokens_generated / n_rollouts,
                # Normalize by actual executed steps (attempted actions), not max possible steps.
                "tokens_per_step": total_tokens_generated / max(1, total_attempted_actions),
                # Backwards-compatible metric: normalize by configured max steps.
                "tokens_per_step_cap": total_tokens_generated / max(1, (env_config['episode_length'] * n_rollouts)),
            })

        metric_dict.update({
            "inference_time_seconds": total_inference_time,
            "inference_time_per_rollout": total_inference_time / max(1, n_rollouts),
            # Normalize by actual executed steps (attempted actions), not max possible steps.
            "inference_time_per_step": total_inference_time / max(1, total_attempted_actions),
            # Backwards-compatible metric: normalize by configured max steps.
            "inference_time_per_step_cap": total_inference_time / max(1, (env_config['episode_length'] * n_rollouts)),
            "env_step_time_seconds": total_env_step_time,
            "env_step_time_per_rollout": total_env_step_time / max(1, n_rollouts),
            "env_step_time_per_step": total_env_step_time / max(1, total_attempted_actions),
        })

        if entropy_enabled:
            num_entropy_measurements = len(entropy_measurements)
            entropy_mean = float(np.mean(entropy_measurements)) if entropy_measurements else 0.0
            entropy_std = float(np.std(entropy_measurements)) if entropy_measurements else 0.0
            metric_dict.update({
                "action_entropy_mean": entropy_mean,
                "action_entropy_std": entropy_std,
                "action_entropy_num_measurements": float(num_entropy_measurements),
                "action_entropy_probe_time_seconds": entropy_probe_time,
            })

            if entropy_action_counter:
                total_actions = sum(entropy_action_counter.values())
                if total_actions > 0:
                    normalized_dist = {
                        action: count / total_actions
                        for action, count in entropy_action_counter.items()
                    }
                    metric_dict["val/entropy_dist"] = json.dumps(normalized_dist)


            if entropy_unique_executed_actions_per_unique_text:
                metric_dict.update({
                    "unique_executed_actions_per_unique_text_mean": float(np.mean(entropy_unique_executed_actions_per_unique_text)),
                    "unique_executed_actions_per_unique_text_std": float(np.std(entropy_unique_executed_actions_per_unique_text)),
                })
            if entropy_unique_valid_actions_per_unique_valid_text:
                metric_dict.update({
                    "unique_valid_actions_per_unique_valid_text_mean": float(np.mean(entropy_unique_valid_actions_per_unique_valid_text)),
                    "unique_valid_actions_per_unique_valid_text_std": float(np.std(entropy_unique_valid_actions_per_unique_valid_text)),
                })

            metric_dict.update({
                "unique_texts_step_mean": float(np.mean(entropy_unique_texts)),
                "unique_texts_step_std": float(np.std(entropy_unique_texts)),
                "unique_executed_actions_step_mean": float(np.mean(entropy_unique_executed_actions)),
                "unique_executed_actions_step_std": float(np.std(entropy_unique_executed_actions)),
                "unique_valid_actions_step_mean": float(np.mean(entropy_unique_valid_actions)),
                "unique_valid_actions_step_std": float(np.std(entropy_unique_valid_actions)),
            })

        # Timing for n_groups > 1 metric computation
        metric_computation_time = 0.0
        if n_groups > 1:
            metric_comp_start = time.perf_counter()
            distinct_state_actions_valid_by_group = [len(set(group_state_actions)) for group_state_actions in group_state_action_texts_valid]
            distinct_state_actions_by_group = [len(set(group_state_actions)) for group_state_actions in group_state_action_texts_all]
            self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: distinct_state_actions_per_group={distinct_state_actions_valid_by_group}")
            # Debug sample of raw state-action strings before set() for the first group
            if group_state_action_texts_valid and group_state_action_texts_valid[0]:
                sample_sa = group_state_action_texts_valid[0][:5]
                self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: sample state-action strings (group 0): {sample_sa}")
            metric_dict.update({
                "n_distinct_state_actions_valid_mean": np.mean(distinct_state_actions_valid_by_group),
                "n_distinct_state_actions_valid_std": np.std(distinct_state_actions_valid_by_group),
                "n_distinct_state_actions_mean": np.mean(distinct_state_actions_by_group),
                "n_distinct_state_actions_std": np.std(distinct_state_actions_by_group),
            })

            # Per-frame / coverage metrics require trajectory lengths (standard rollouts).
            if all_len_of_traj:
                # Frame counting (per group and total)
                total_frames_per_group = [0.0] * n_groups
                len_array = np.atleast_1d(np.array(all_len_of_traj, dtype=float))
                for i in range(n_rollouts):
                    group_idx = i // seed_group_size
                    traj_len = len_array[i] if i < len_array.shape[0] else len_array[-1]
                    total_frames_per_group[group_idx] += traj_len

                # Per-frame distinct count including invalid actions
                distinct_state_actions_per_frame_by_group = [
                    float(n_distinct_state_actions) / n_frames
                    for n_distinct_state_actions, n_frames in zip(distinct_state_actions_by_group, total_frames_per_group)
                    if n_frames > 0
                ]
                distinct_state_actions_valid_per_frame_by_group = [
                    float(n_distinct_state_actions) / n_frames
                    for n_distinct_state_actions, n_frames in zip(distinct_state_actions_valid_by_group, total_frames_per_group)
                    if n_frames > 0
                ]

                self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: total_frames_per_group={total_frames_per_group}")
                self._dbg_print(
                    f"[MultiEnvEvaluator] {eval_name}: distinct_state_actions_per_frame_by_group (incl invalid)={distinct_state_actions_per_frame_by_group}"
                )
                self._dbg_print(
                    f"[MultiEnvEvaluator] {eval_name}: distinct_state_actions_valid_per_frame_by_group={distinct_state_actions_valid_per_frame_by_group}"
                )

                metric_dict.update({
                    "distinct_state_actions_per_frame_mean": np.mean(distinct_state_actions_per_frame_by_group),
                    "distinct_state_actions_per_frame_std": np.std(distinct_state_actions_per_frame_by_group),
                    "distinct_state_actions_valid_per_frame_mean": np.mean(distinct_state_actions_valid_per_frame_by_group),
                    "distinct_state_actions_valid_per_frame_std": np.std(distinct_state_actions_valid_per_frame_by_group),
                })

                # Coverage vs opportunity (valid-only distinct counts)
                opportunity = float(seed_group_size * env_config['episode_length'])
                if opportunity > 0:
                    distinct_state_action_valid_coverage_by_group = [
                        float(n_distinct_state_actions) / opportunity
                        for n_distinct_state_actions in distinct_state_actions_valid_by_group
                    ]
                    distinct_state_action_coverage_by_group = [
                        float(n_distinct_state_actions) / opportunity
                        for n_distinct_state_actions in distinct_state_actions_by_group
                    ]
                    self._dbg_print(
                        f"[MultiEnvEvaluator] {eval_name}: distinct_state_action_valid_coverage_by_group={distinct_state_action_valid_coverage_by_group}, opportunity={opportunity}"
                    )
                    self._dbg_print(
                        f"[MultiEnvEvaluator] {eval_name}: distinct_state_action_coverage_by_group={distinct_state_action_coverage_by_group}, opportunity={opportunity}"
                    )
                    metric_dict.update({
                        "distinct_state_actions_valid_coverage_mean": np.mean(distinct_state_action_valid_coverage_by_group),
                        "distinct_state_actions_valid_coverage_std": np.std(distinct_state_action_valid_coverage_by_group),
                        "distinct_state_actions_coverage_mean": np.mean(distinct_state_action_coverage_by_group),
                        "distinct_state_actions_coverage_std": np.std(distinct_state_action_coverage_by_group),
                    })
                else:
                    self._dbg_print(f"[MultiEnvEvaluator] {eval_name} WARNING: opportunity is 0, skipping coverage metrics")
            else:
                self._dbg_print(
                    f"[MultiEnvEvaluator] {eval_name}: Skipping per-frame/coverage metrics because trajectory lengths are unavailable"
                )
            metric_comp_end = time.perf_counter()
            metric_computation_time = metric_comp_end - metric_comp_start

        # Print timing breakdown for batched evaluation debugging
        if n_batches > 1:
            print(f"[MultiEnvEvaluator] {eval_name} timing breakdown:")
            print(f"  VecEnv creation:    {total_vecenv_create_time:.2f}s")
            print(f"  VecEnv reset:       {total_vecenv_reset_time:.2f}s")
            print(f"  VecEnv close:       {total_vecenv_close_time:.2f}s")
            print(f"  Tokenizer:          {total_tokenizer_time:.2f}s")
            print(f"  GC:                 {total_gc_time:.2f}s")
            print(f"  State-action accum: {total_state_action_accum_time:.2f}s")
            print(f"  Metric computation: {metric_computation_time:.2f}s")
            # Also add to metrics for logging
            metric_dict.update({
                "debug_vecenv_create_time": total_vecenv_create_time,
                "debug_vecenv_reset_time": total_vecenv_reset_time,
                "debug_vecenv_close_time": total_vecenv_close_time,
                "debug_tokenizer_time": total_tokenizer_time,
                "debug_gc_time": total_gc_time,
                "debug_state_action_accum_time": total_state_action_accum_time,
                "debug_metric_computation_time": metric_computation_time,
            })

        # Valid action tracking across the episode
        total_len = float(np.sum(all_len_of_traj)) if all_len_of_traj else 0.0
        metric_dict.update({
            "total_len_of_trajs": total_len,
            "valid_actions_total": float(total_valid_actions),
            "attempted_actions_total": float(total_attempted_actions),
            # Convenience alias: number of executed environment steps across all rollouts.
            # (Matches attempted_actions_total; provided as an explicit denominator for rate metrics.)
            "executed_steps_total": float(total_attempted_actions),
            "executed_steps_per_rollout": float(total_attempted_actions) / max(1, float(n_rollouts)),
            "valid_action_ratio": float(total_valid_actions) / max(1.0, float(total_attempted_actions)),
        })

        # Always include VecEnv creation time (useful for diagnosing slow evals)
        metric_dict["vecenv_create_time_seconds"] = total_vecenv_create_time

        episode_data = None
        if track_standard_metrics and episode_tracked:
            # Use reward as fallback for score when env doesn't provide score
            score_value = episode_total_score if episode_total_score is not None else episode_total_reward
            episode_data = {
                'inputs': episode_inputs,
                'outputs': episode_outputs,
                'total_score': float(score_value) if score_value is not None else None,
                'total_reward': float(episode_total_reward) if episode_total_reward is not None else None,
                'max_length_steps': episode_max_length_steps,
            }
            
        return metric_dict, episode_data

    def _validate_action_entropy_config(self, entropy_cfg: Dict) -> Dict:
        """Validate and normalize the action entropy configuration."""
        cfg = dict(entropy_cfg)
        n_samples = int(cfg.get('n_samples', 1))
        if n_samples <= 0:
            raise ValueError("action_entropy.n_samples must be > 0")

        temperature = float(cfg.get('temperature', 1.0))
        if temperature <= 0:
            raise ValueError("action_entropy.temperature must be > 0")

        measure_mode = str(cfg.get('measure_at_steps', 'start')).lower()
        allowed_modes = {"start", "random", "every_n"}
        if measure_mode not in allowed_modes:
            raise ValueError(f"action_entropy.measure_at_steps must be one of {allowed_modes}")

        step_interval = int(cfg.get('step_interval', 1))
        if step_interval <= 0:
            step_interval = 1

        max_batch_size = int(cfg.get('max_batch_size', 64))
        if max_batch_size <= 0:
            raise ValueError("action_entropy.max_batch_size must be > 0")

        return {
            "n_samples": n_samples,
            "temperature": temperature,
            "measure_at_steps": measure_mode,
            "step_interval": step_interval,
            "exclusive_metric": bool(cfg.get('exclusive_metric', False)),
            "max_batch_size": max_batch_size,
        }

    def _compute_entropy_measurement_steps(
        self,
        entropy_cfg: Dict,
        episode_length: int,
        seed: Optional[int],
    ) -> List[int]:
        """Determine the rollout steps at which entropy should be measured."""
        episode_length = max(1, int(episode_length))
        mode = entropy_cfg['measure_at_steps']
        
        if mode == "start":
            steps = [0]
        elif mode == "random":
            rng = np.random.default_rng(seed)
            max_step = max(episode_length - 1, 0)
            steps = [int(rng.integers(0, max_step + 1))]
        else:  # every_n
            interval = entropy_cfg['step_interval']
            steps = list(range(0, episode_length, interval))
        
        normalized_steps = sorted({step for step in steps if 0 <= step < episode_length})
        return normalized_steps or [0]

    def _probe_action_entropy(
        self,
        prompt_texts: Sequence[str],
        entropy_cfg: Dict,
        active_rollouts: Optional[np.ndarray],
        max_seq_len: int,
        action_extraction_fn: Callable,
    ):
        """Run entropy probes (n_samples completions) for the provided prompts.

        Returns:
            entropies: list of entropy values of _executed_ actions (including default replacements for invalid actions) (per active rollout)
            action_counter: Counter of normalized actions across all samples
            total_probe_time: total inference time
            unique_texts_count: number of distinct raw response strings across all samples
            unique_valid_texts_count: number of distinct raw response strings that yielded valid actions
            unique_executed_actions_count: number of distinct executed action strings across all samples
            unique_valid_actions_count: number of distinct valid generated action strings across all samples
        """
        n_samples = entropy_cfg['n_samples']
        prompts: List[str] = []
        prompt_owner: List[int] = []

        for idx, prompt in enumerate(prompt_texts):
            is_active = True if active_rollouts is None else bool(active_rollouts[idx])
            if not is_active:
                continue
            prompts.extend([prompt] * n_samples)
            prompt_owner.extend([idx] * n_samples)

        if not prompts:
            return [], Counter(), 0.0, 0, 0, 0, 0

        chunk_size = entropy_cfg['max_batch_size']
        probe_meta = dict(self._current_gen_config)
        probe_meta['temperature'] = entropy_cfg['temperature']
        probe_meta['do_sample'] = True
        probe_meta.pop('n', None)

        action_counter: Counter = Counter()
        valid_action_counter: Counter = Counter()
        all_responses: List[str] = []  # All raw response texts
        all_valid_response_texts: List[str] = []  # Raw response texts that yielded valid actions
        rollout_samples: Dict[int, List[str]] = {}
        total_probe_time = 0.0

        for start in range(0, len(prompts), chunk_size):
            end = start + chunk_size
            chunk_prompts = prompts[start:end]
            chunk_owners = prompt_owner[start:end]
            chunk_responses, chunk_time = self._run_entropy_generation_chunk(
                chunk_prompts,
                probe_meta,
                max_seq_len,
            )
            total_probe_time += chunk_time
            all_responses.extend(chunk_responses)

            for owner_idx, response in zip(chunk_owners, chunk_responses):
                # Decode the model response into structured action info; surface errors if decoding fails.
                full_action, extracted_action, executed_action, is_valid, _ = action_extraction_fn(response)

                action_counter[executed_action] += 1  # Count instances of actually executed actions (with default replacements for invalid actions)
                if is_valid:
                    valid_action_counter[executed_action] += 1  # Count instances of correctly generated actions
                    all_valid_response_texts.append(response)  # Store full response text, not action
                rollout_samples.setdefault(owner_idx, []).append(executed_action)

        entropies = [
            self._compute_shannon_entropy(samples)
            for samples in rollout_samples.values()
        ]
        unique_texts_count = len(set(all_responses))
        unique_valid_texts_count = len(set(all_valid_response_texts))
        unique_executed_actions_count = len(set(action_counter.keys()))
        unique_valid_actions_count = len(set(valid_action_counter.keys()))
        return entropies, action_counter, total_probe_time, unique_texts_count, unique_valid_texts_count, unique_executed_actions_count, unique_valid_actions_count

    def _run_entropy_generation_chunk(
        self,
        prompts: Sequence[str],
        probe_meta: Dict,
        max_seq_len: int,
    ):
        """Generate a batch of probe completions."""
        tokenized = self.tokenizer(
            list(prompts),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_seq_len
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        val_obs_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        val_gen_batch = DataProto.from_dict(tensors=val_obs_data)
        val_gen_batch.meta_info["step"] = None

        for key, value in probe_meta.items():
            val_gen_batch.meta_info[key] = value

        inference_start = time.perf_counter()
        val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
        inference_end = time.perf_counter()

        responses = self.tokenizer.batch_decode(
            val_gen_batch_output.batch['responses'],
            skip_special_tokens=True
        )
        return responses, (inference_end - inference_start)

    @staticmethod
    def _compute_shannon_entropy(samples: Sequence[str]) -> float:
        """Compute Shannon entropy (in nats) for the provided samples."""
        if not samples:
            return 0.0
        counts = Counter(samples)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = np.array(list(counts.values()), dtype=np.float64) / total
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy)

    @staticmethod
    def _compute_seed_sequence(
        initial_seed: Optional[int],
        n_rollouts: int,
        seed_group_size: Optional[int] = None,
    ) -> List[Optional[int]]:
        """Compute the full seed sequence for all rollouts upfront.

        Uses the same logic as VecEnv.reset() to ensure determinism:
        - If seed_group_size is set: groups of rollouts get same seed (initial_seed + group_index)
        - Otherwise: each rollout gets unique seed (initial_seed + i)

        Args:
            initial_seed: Base seed (None means no seeding)
            n_rollouts: Total number of rollouts
            seed_group_size: If provided, group rollouts to share seeds

        Returns:
            List of seeds, one per rollout
        """
        if initial_seed is None:
            return [None] * n_rollouts

        seeds = []
        # Normalize seed_group_size
        effective_group_size = seed_group_size
        if effective_group_size == n_rollouts:
            effective_group_size = None

        for i in range(n_rollouts):
            if effective_group_size is not None:
                group_index = i // effective_group_size
                seeds.append(initial_seed + group_index)
            else:
                # Incremental seeds (each rollout gets unique seed)
                seeds.append(initial_seed + i)

        return seeds
 
