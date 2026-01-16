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
import numpy as np
import time
import gc
import psutil
import os
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Callable
from verl import DataProto
from verl.utils.tracking import ValidationGenerationsLogger

from verl.envs.environments import get_action_extraction_fn


class VecEnvContextManager:
    """Context manager to ensure proper cleanup of vectorized environments."""
    
    def __init__(self, env_name, task, config, render_mode=None):
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
                print(f"[VecEnvContextManager] Successfully closed environment {self.env_name}")
            except Exception as e:
                print(f"[ERROR] VecEnvContextManager: Failed to close environment {self.env_name}: {e}")
                import traceback
                print(f"[ERROR] VecEnvContextManager: Close traceback: {traceback.format_exc()}")
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

            start_time = time.time()

            # Run evaluation (VecEnv creation now happens inside _evaluate_single_env_body)
            try:
                env_metrics, episode_data = self._evaluate_single_env(env_config, eval_name)

                end_time = time.time()
                eval_time = end_time - start_time

                self._maybe_log_episode_generation(episode_data, eval_name, global_step)

                prefixed_metrics = {}
                for key, value in env_metrics.items():
                    prefixed_key = f"eval_{eval_name}/{key}"
                    prefixed_metrics[prefixed_key] = value

                all_metrics.update(prefixed_metrics)
                self._dbg_print(f"[MultiEnvEvaluator] Added {len(prefixed_metrics)} metrics for {eval_name}")
                self._dbg_print(f"[MultiEnvEvaluator] Sample metrics for {eval_name}: {list(prefixed_metrics.keys())[:5]}")

                inference_time = env_metrics.get("inference_time_seconds", 0.0)
                print(f"Completed evaluation for {eval_name} in {eval_time:.2f}s (inference: {inference_time:.2f}s)")

            except Exception as e:
                print(f"[ERROR] MultiEnvEvaluator: Failed to evaluate environment {eval_name}: {e}")
                import traceback
                print(f"[ERROR] MultiEnvEvaluator: Traceback: {traceback.format_exc()}")
                raise

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
        score_log_value = "N/A" if total_score is None else total_score
        if total_score is None:
            self._dbg_print(f"[MultiEnvEvaluator] Episode score unavailable for {env_name}; logging 'N/A'.")
        
        # Create sample tuple (input, output, score) as expected by ValidationGenerationsLogger
        sample = (formatted_input, formatted_output, score_log_value)
        
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
        total_attempted_actions = 0
        total_valid_actions = 0

        # Episode tracking (first rollout only, from first batch)
        episode_inputs = []
        episode_outputs = []
        episode_total_score = None
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

        # =========== BATCH LOOP ===========
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_rollouts)
            batch_n = batch_end - batch_start
            batch_seeds = all_seeds[batch_start:batch_end]

            if n_batches > 1:
                self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: Starting batch {batch_idx + 1}/{n_batches} (rollouts {batch_start}-{batch_end - 1})")

            # Create temp config with batch_n rollouts
            temp_config = self._create_env_config(env_config, n_rollouts_override=batch_n)

            # Create VecEnv for this batch
            with VecEnvContextManager(
                env_config['env_name'],
                temp_config.envs.task,
                temp_config,
                render_mode=None
            ) as vec_envs:
                # Reset with explicit seeds for this batch
                obs_vec, info_vec = vec_envs.reset(seeds=batch_seeds)

                # Per-batch state
                pending_entropy_steps = set(entropy_measure_steps) if entropy_enabled else set()
                active_rollouts = np.ones(batch_n, dtype=bool) if entropy_enabled else None
                end_of_traj = None
                rew_of_traj = 0.
                score_of_traj = None
                len_of_traj = 0.
                pos_rew_of_traj = None

                # Track the last-step token counts for this batch.
                batch_response_n_tokens_last = None

                # Episode loop for this batch
                for step_idx in range(env_config['episode_length']):
                    val_input_obs_text = self.tokenizer.apply_chat_template(
                        obs_vec, tokenize=False, add_generation_prompt=True
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

                    inference_start = time.time()
                    val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
                    inference_end = time.time()
                    total_inference_time += (inference_end - inference_start)

                    response_ids = val_gen_batch_output.batch['responses']
                    full_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                    # Track first rollout for logging (only from first batch)
                    if track_standard_metrics and not episode_tracked and batch_idx == 0:
                        episode_inputs.append(val_input_obs_text[0])
                        episode_outputs.append(full_responses[0])

                    response_n_tokens = (response_ids != self.tokenizer.pad_token_id).sum(dim=-1)
                    total_tokens_generated += response_n_tokens.sum().item()

                    # Save last-step token counts for later aggregation.
                    batch_response_n_tokens_last = response_n_tokens

                    try:
                        obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = vec_envs.step(full_responses)
                    except Exception as e:
                        print(f"[ERROR] MultiEnvEvaluator: Exception in val_env.step: {e}")
                        import traceback
                        print(f"[ERROR] MultiEnvEvaluator: Traceback: {traceback.format_exc()}")
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

                    # Accumulate state-action texts with GLOBAL indexing
                    for local_idx, (observation_text, executed_action, was_valid_action, rollout_already_ended) in enumerate(
                        zip(val_input_obs_text, executed_actions, was_valid_list, use_end_of_traj)
                    ):
                        global_idx = batch_start + local_idx
                        group_idx = global_idx // seed_group_size
                        if was_valid_action and not rollout_already_ended:
                            group_state_action_texts_valid[group_idx].append(f"{observation_text} {executed_action}")
                        if not rollout_already_ended:
                            group_state_action_texts_all[group_idx].append(f"{observation_text} {executed_action}")

                    if entropy_enabled and active_rollouts is not None:
                        done_mask = np.logical_or(terminated_vec, truncated_vec)
                        active_rollouts = np.logical_not(done_mask)

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

                # Capture last-step token counts for this batch (if any generation occurred).
                if batch_response_n_tokens_last is not None:
                    response_n_tokens_last_step[batch_start:batch_end] = [
                        int(x) for x in batch_response_n_tokens_last.tolist()
                    ]

            # End of VecEnv context manager - environment closed
            # Memory cleanup between batches
            gc.collect()
            if n_batches > 1:
                self._dbg_print(f"[MultiEnvEvaluator] {eval_name}: Completed batch {batch_idx + 1}/{n_batches}")

        # =========== END BATCH LOOP ===========

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

        if n_groups > 1:
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

        episode_data = None
        if track_standard_metrics and episode_tracked:
            episode_data = {
                'inputs': episode_inputs,
                'outputs': episode_outputs,
                'total_score': float(episode_total_score) if episode_total_score is not None else None
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

        inference_start = time.time()
        val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
        inference_end = time.time()

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
 
