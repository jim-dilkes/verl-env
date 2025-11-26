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
from typing import Dict, List, Optional, Sequence
from verl import DataProto
from verl.utils.tracking import ValidationGenerationsLogger


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
    
    def __init__(self, config, tokenizer, actor_rollout_wg, val_reward_fn, eval_config=None):
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
        
        # Initialize validation generations logger
        self.validation_generations_logger = ValidationGenerationsLogger()
        
        # Extract evaluation environments from config
        if eval_config is not None:
            self.eval_environments = eval_config.environments
        else:
            self.eval_environments = getattr(config.evaluation, 'environments', [])
        
        if not self.eval_environments:
            raise ValueError("No evaluation environments found in evaluation config")
        
        print(f"MultiEnvEvaluator initialized with {len(self.eval_environments)} evaluation environments")
        for i, env_config in enumerate(self.eval_environments):
            print(f"  Environment {i}: {env_config.get('name', f'env_{i}')}")
    
    def evaluate(self, global_step):
        """
        Run evaluation across all configured environments.
        
        Args:
            global_step: Current global step for logging purposes
            
        Returns:
            dict: Combined metrics from all evaluation environments
        """
        print(f"[MultiEnvEvaluator] Starting evaluation at global_step={global_step}")
        print(f"[MultiEnvEvaluator] Number of environments to evaluate: {len(self.eval_environments)}")
        
        # Log initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"[MultiEnvEvaluator] Initial memory usage: {initial_memory:.1f} MB")
        
        all_metrics = {}
        
        for env_idx, env_config in enumerate(self.eval_environments):
            env_name = env_config.get('name', f'env_{env_idx}')
            print(f"Evaluating environment: {env_name}")
            
            # try:
            # Record start time for this environment
            start_time = time.time()
            
            # Create a temporary config for this environment
            temp_config = self._create_env_config(env_config)
            
            # Create vectorized environment for this evaluation using context manager
            try:
                with VecEnvContextManager(
                    env_config['env_name'],
                    temp_config.envs.task,
                    temp_config,
                    render_mode=None
                ) as val_env:
                    # Run evaluation for this environment
                    env_metrics, episode_data = self._evaluate_single_env(val_env, env_config, env_name)
                    
                    # Record end time and calculate duration
                    end_time = time.time()
                    eval_time = end_time - start_time
                    
                    # Log episode generation if configured
                    self._maybe_log_episode_generation(episode_data, env_name, global_step)
                    
                    # Add environment-specific prefix to metrics
                    prefixed_metrics = {}
                    for key, value in env_metrics.items():
                        prefixed_key = f"eval_{env_name}/{key}"
                        prefixed_metrics[prefixed_key] = value
                    
                    all_metrics.update(prefixed_metrics)
                    print(f"[MultiEnvEvaluator] Added {len(prefixed_metrics)} metrics for {env_name}")
                    print(f"[MultiEnvEvaluator] Sample metrics for {env_name}: {list(prefixed_metrics.keys())[:5]}")
                    
                    inference_time = env_metrics.get("inference_time_seconds", 0.0)
                    print(f"Completed evaluation for {env_name} in {eval_time:.2f}s (inference: {inference_time:.2f}s)")
                    
            except Exception as e:
                print(f"[ERROR] MultiEnvEvaluator: Failed to evaluate environment {env_name}: {e}")
                import traceback
                print(f"[ERROR] MultiEnvEvaluator: Traceback: {traceback.format_exc()}")
                raise
            
            # Force garbage collection after each environment to free memory
            gc.collect()
            
            # Log memory usage after each environment
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"[MultiEnvEvaluator] Memory usage after {env_name}: {current_memory:.1f} MB (delta: {current_memory - initial_memory:+.1f} MB)")
        
        # Final memory cleanup and logging
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"[MultiEnvEvaluator] Final memory usage: {final_memory:.1f} MB (total delta: {final_memory - initial_memory:+.1f} MB)")
        
        print(f"[MultiEnvEvaluator] Evaluation completed. Total metrics collected: {len(all_metrics)}")
        print(f"[MultiEnvEvaluator] All metric keys: {list(all_metrics.keys())}")
        print(f"[MultiEnvEvaluator] Sample metric values: {dict(list(all_metrics.items())[:3])}")
        
        return all_metrics
    
    def _create_env_config(self, env_config):
        """
        Create a temporary config object for a specific environment.
        
        Args:
            env_config: Environment-specific configuration
            
        Returns:
            OmegaConf object: Temporary config for the environment
        """
        from omegaconf import OmegaConf, open_dict
        
        print(f"[MultiEnvEvaluator] Creating config for environment: {env_config.get('name', 'unknown')}")
        print(f"[MultiEnvEvaluator] Original env_config: {env_config}")
        print(f"[MultiEnvEvaluator] Training config n_rollouts: {self.config.envs.n_rollouts}")
        print(f"[MultiEnvEvaluator] Evaluation env_config n_rollouts: {env_config['n_rollouts']}")
        
        # Create a proper copy using OmegaConf methods to avoid struct mode issues
        # Convert to container, modify, then recreate OmegaConf object
        temp_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        
        # Use open_dict to allow modifications even if struct mode was enabled
        with open_dict(temp_config):
            # Override environment-specific settings
            temp_config.envs.n_rollouts = env_config['n_rollouts']
            temp_config.envs.episode_length = env_config['episode_length']
            temp_config.envs.env_name = env_config['env_name']
            temp_config.envs.task = env_config.get('task', None)  # Set the task from env_config
            temp_config.envs.freeze_completed_episodes = env_config.get('freeze_completed_episodes', False)
            temp_config.envs.duplication_mode = env_config.get('duplication_mode', 'none')
            temp_config.envs.format_penalty = env_config.get('format_penalty', 0.0)
            temp_config.envs.binary_reward = env_config.get('binary_reward', False)
            
            # Handle instruction_prompt from evaluation config
            if 'instruction_prompt' in env_config:
                temp_config.envs.instruction_prompt = env_config['instruction_prompt']
                print(f"[MultiEnvEvaluator] Set instruction_prompt from eval config (length: {len(env_config['instruction_prompt']) if env_config['instruction_prompt'] else 0} chars)")
            
            print(f"[MultiEnvEvaluator] After basic overrides - n_rollouts: {temp_config.envs.n_rollouts}, task: {temp_config.envs.task}, env_name: {temp_config.envs.env_name}")
            
            # Handle captioner configuration
            if 'captioner' in env_config:
                print(f"[MultiEnvEvaluator] Original captioner config: {self.config.envs.captioner}")
                print(f"[MultiEnvEvaluator] Environment captioner config: {env_config['captioner']}")
                
                # Merge captioner config with defaults to ensure all required fields are present
                captioner_config = OmegaConf.to_container(self.config.envs.captioner)
                captioner_config.update(env_config['captioner'])
                temp_config.envs.captioner = captioner_config
                
                print(f"[MultiEnvEvaluator] Final captioner config: {temp_config.envs.captioner}")
            
            # Handle environment-specific kwargs
            env_name = env_config['env_name']
            if f'{env_name}_kwargs' in env_config:
                print(f"[MultiEnvEvaluator] Setting {env_name}_kwargs: {env_config[f'{env_name}_kwargs']}")
                temp_config.envs[f'{env_name}_kwargs'] = env_config[f'{env_name}_kwargs']
            
            # Set initial seed if specified
            if 'initial_seed' in env_config:
                temp_config.envs.group_initial_seed = env_config['initial_seed']
                print(f"[MultiEnvEvaluator] Set initial_seed: {env_config['initial_seed']}")
        
        print(f"[MultiEnvEvaluator] Final temp_config.envs.n_rollouts: {temp_config.envs.n_rollouts}")
        print(f"[MultiEnvEvaluator] Final temp_config.envs keys: {list(temp_config.envs.keys())}")
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
            print(f"[MultiEnvEvaluator] Using generation config from env_config: {gen_config_from_env}")
            
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
                    print(f"[MultiEnvEvaluator] Using generation config from val_kwargs: {val_kwargs}")
                    
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
        
        print(f"[MultiEnvEvaluator] Final generation config: {gen_config}")
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
            print(f"[MultiEnvEvaluator] No episode data to log for {env_name}")
            return
        
        # Format the episode data for logging
        formatted_input = self._format_episode(episode_data['inputs'])
        formatted_output = self._format_episode(episode_data['outputs'])
        total_score = episode_data['total_score']
        score_log_value = "N/A" if total_score is None else total_score
        if total_score is None:
            print(f"[MultiEnvEvaluator] Episode score unavailable for {env_name}; logging 'N/A'.")
        
        # Create sample tuple (input, output, score) as expected by ValidationGenerationsLogger
        sample = (formatted_input, formatted_output, score_log_value)
        
        # Log to each configured logger
        logger_backends = getattr(self.config.trainer, 'logger', ['console'])
        self.validation_generations_logger.log(logger_backends, [sample], global_step, table_name=f"eval_{env_name}_gen/generations")
        
        print(f"[MultiEnvEvaluator] Logged episode generation for {env_name} at step {global_step}")
    
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

    def _extract_info_array(self, info_obj, key: str, expected_len: int) -> Optional[np.ndarray]:
        """
        Attempt to extract a per-rollout array for `key` from a Gym-like info object.

        Returns:
            np.ndarray of shape (expected_len,) with float64 values, or None if the key
            is unavailable or the shape does not match.
        """
        if info_obj is None:
            return None

        if isinstance(info_obj, Mapping):
            if key not in info_obj:
                return None
            values = info_obj[key]
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 0:
                if expected_len != 1:
                    return None
                arr = arr.reshape(1)
            if arr.shape[0] != expected_len:
                return None
            return arr

        if isinstance(info_obj, (list, tuple)):
            if len(info_obj) != expected_len:
                return None
            extracted = []
            for item in info_obj:
                if not isinstance(item, Mapping) or key not in item:
                    return None
                extracted.append(item[key])
            return np.asarray(extracted, dtype=np.float64)

        return None
    
    def _evaluate_single_env(self, val_env, env_config, env_name):
        """Run evaluation while preserving tokenizer state."""
        original_padding_side = getattr(self.tokenizer, "padding_side", None)
        self.tokenizer.padding_side = "left"
        try:
            return self._evaluate_single_env_body(val_env, env_config, env_name)
        finally:
            if original_padding_side is not None:
                self.tokenizer.padding_side = original_padding_side

    def _evaluate_single_env_body(self, val_env, env_config, env_name):
        """
        Run evaluation for a single environment, optionally collecting action entropy metrics.
        
        Args:
            val_env: The vectorized environment to evaluate
            env_config: Environment-specific configuration
            env_name: Name of the environment for logging
            
        Returns:
            tuple: (dict of metrics, dict episode_data)
        """
        max_seq_len = self.config.data.max_prompt_length
        initial_seed = env_config.get('initial_seed', None)
        if initial_seed is not None:
            initial_seed = int(initial_seed)
        val_obs, val_info = val_env.reset(seed=initial_seed, use_incremental_seeds=True)

        # Action entropy configuration (if provided)
        action_entropy_cfg = env_config.get('action_entropy') or {}
        entropy_enabled = bool(action_entropy_cfg.get('enabled', False))
        entropy_cfg = None
        entropy_measure_steps: Sequence[int] = []
        pending_entropy_steps = set()
        entropy_measurements: List[float] = []
        entropy_action_counter: Counter = Counter()
        entropy_probe_time = 0.0
        active_rollouts = None

        if entropy_enabled:
            entropy_cfg = self._validate_action_entropy_config(action_entropy_cfg)
            entropy_measure_steps = self._compute_entropy_measurement_steps(
                entropy_cfg,
                env_config['episode_length'],
                initial_seed,
            )
            pending_entropy_steps = set(entropy_measure_steps)
            active_rollouts = np.ones(env_config['n_rollouts'], dtype=bool)
            print(
                f"[MultiEnvEvaluator] Action entropy enabled for {env_name} "
                f"with measurement steps: {sorted(entropy_measure_steps)}"
            )

        exclusive_entropy_metrics = bool(entropy_cfg['exclusive_metric']) if entropy_enabled else False
        track_standard_metrics = not (entropy_enabled and exclusive_entropy_metrics)

        # Lists to collect samples for logging (only populated when tracking metrics)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        end_of_traj = None
        rew_of_traj = 0.
        score_of_traj = None
        len_of_traj = 0.
        total_tokens_generated = 0
        total_inference_time = 0.0
        score_tracking_active = False
        score_warning_logged = False
        
        # Track one full episode for logging (first rollout)
        episode_inputs = []
        episode_outputs = []
        episode_total_score = None
        episode_tracked = False

        # Get generation config for this environment (once per environment)
        self._current_gen_config = self._get_generation_config(env_config)
        response_lengths = None

        for step_idx in range(env_config['episode_length']):
            val_input_obs_text = self.tokenizer.apply_chat_template(
                val_obs, tokenize=False, add_generation_prompt=True
            )

            if entropy_enabled and step_idx in pending_entropy_steps:
                entropies, step_action_counts, probe_time = self._probe_action_entropy(
                    val_input_obs_text,
                    entropy_cfg,
                    active_rollouts,
                    max_seq_len,
                )
                entropy_measurements.extend(entropies)
                entropy_action_counter.update(step_action_counts)
                entropy_probe_time += probe_time
                pending_entropy_steps.remove(step_idx)

                if not track_standard_metrics and not pending_entropy_steps:
                    # Entropy-only evaluation is complete; no need to continue rollouts.
                    break

            if track_standard_metrics:
                sample_inputs.extend(val_input_obs_text)
                if not episode_tracked:
                    episode_inputs.append(val_input_obs_text[0])

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
            actions = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            if track_standard_metrics:
                sample_outputs.extend(actions)
                if not episode_tracked:
                    episode_outputs.append(actions[0])
            
            response_lengths = (response_ids != self.tokenizer.pad_token_id).sum(dim=-1)
            total_tokens_generated += response_lengths.sum().item()
            
            try:
                val_obs, val_reward, val_terminated, val_truncated, val_info = val_env.step(actions)
            except Exception as e:
                print(f"[ERROR] MultiEnvEvaluator: Exception in val_env.step: {e}")
                import traceback
                print(f"[ERROR] MultiEnvEvaluator: Traceback: {traceback.format_exc()}")
                raise

            if entropy_enabled and active_rollouts is not None:
                done_mask = np.logical_or(val_terminated, val_truncated)
                active_rollouts = np.logical_not(done_mask)
            
            if track_standard_metrics:
                score_values = self._extract_info_array(val_info, "score", len(val_reward))
                if score_values is not None:
                    score_tracking_active = True
                elif score_tracking_active and not score_warning_logged:
                    print(
                        f"[MultiEnvEvaluator] Warning: score info missing at step {step_idx} "
                        f"for {env_name}, keeping last known values."
                    )
                    score_warning_logged = True

                if end_of_traj is None:
                    end_of_traj = np.logical_or(val_terminated, val_truncated)
                    rew_of_traj = val_reward
                    len_of_traj = np.ones_like(val_reward)
                    pos_rew_of_traj = (np.array(val_reward) > 0.) * 1.
                    if score_values is not None:
                        score_of_traj = score_values
                else:
                    done = np.logical_or(val_terminated, val_truncated)
                    active_mask = (~end_of_traj).astype(np.float32)
                    rew_of_traj += val_reward * active_mask
                    len_of_traj += active_mask
                    end_of_traj = np.logical_or(end_of_traj, done)
                    pos_rew_of_traj = np.logical_or(pos_rew_of_traj, (np.array(val_reward) > 0.) * 1.)
                    if score_values is not None:
                        if score_of_traj is None:
                            score_of_traj = score_values
                        else:
                            score_of_traj = np.where(~end_of_traj, score_values, score_of_traj)
                
                if score_tracking_active and score_of_traj is not None:
                    sample_scores.extend(score_of_traj.tolist())
                
                if not episode_tracked:
                    if score_tracking_active and score_of_traj is not None:
                        episode_total_score = float(score_of_traj[0])
                    if val_terminated[0] or val_truncated[0] or step_idx == env_config['episode_length'] - 1:
                        episode_tracked = True

                if end_of_traj.all():
                    break
        
        metric_dict: Dict[str, float] = {}

        if track_standard_metrics and end_of_traj is not None:
            rew_of_traj = np.array(rew_of_traj, dtype=np.float64)
            len_of_traj = np.array(len_of_traj, dtype=np.float64)
            pos_rew_of_traj = np.array(pos_rew_of_traj, dtype=np.float64)
            response_lengths = response_lengths.float() if response_lengths is not None else None
            
            metric_dict.update({
                "rewards_mean": float(rew_of_traj.mean()),
                "rewards_std": float(rew_of_traj.std()),
                "pos_reward_any_prop_mean": float(pos_rew_of_traj.mean()),
                "pos_reward_any_prop_std": float(pos_rew_of_traj.std()),
                "traj_length_mean": float(len_of_traj.mean()),
                "traj_length_std": float(len_of_traj.std()),
            })

            if score_tracking_active and score_of_traj is not None:
                score_arr = np.array(score_of_traj, dtype=np.float64)
                metric_dict.update({
                    "score_mean": float(score_arr.mean()),
                    "score_std": float(score_arr.std()),
                })
            elif not score_tracking_active:
                print(f"[MultiEnvEvaluator] Score info unavailable for {env_name}; skipping score metrics.")

            if response_lengths is not None:
                metric_dict.update({
                    "toks_out_mean": float(response_lengths.mean()),
                    "toks_out_std": float(response_lengths.std()),
                })

            metric_dict.update({
                "tokens_per_rollout": total_tokens_generated / env_config['n_rollouts'],
                "tokens_per_step": total_tokens_generated / max(1, (env_config['episode_length'] * env_config['n_rollouts'])),
            })

        metric_dict.update({
            "inference_time_seconds": total_inference_time,
            "inference_time_per_rollout": total_inference_time / max(1, env_config['n_rollouts']),
            "inference_time_per_step": total_inference_time / max(1, (env_config['episode_length'] * env_config['n_rollouts'])),
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
    ):
        """Run entropy probes (n_samples completions) for the provided prompts."""
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
            return [], Counter(), 0.0

        chunk_size = entropy_cfg['max_batch_size']
        probe_meta = dict(self._current_gen_config)
        probe_meta['temperature'] = entropy_cfg['temperature']
        probe_meta['do_sample'] = True
        probe_meta.pop('n', None)

        action_counter: Counter = Counter()
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

            for owner_idx, response in zip(chunk_owners, chunk_responses):
                action_text = self._extract_action_string(response)
                normalized_action = self._normalize_action_text(action_text)
                action_counter[normalized_action] += 1
                rollout_samples.setdefault(owner_idx, []).append(normalized_action)

        entropies = [
            self._compute_shannon_entropy(samples)
            for samples in rollout_samples.values()
        ]
        return entropies, action_counter, total_probe_time

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
    def _extract_action_string(text: str | None) -> str:
        """Extract the content inside <action> tags, falling back to raw text."""
        if not text:
            return ""
        
        lower_text = text.lower()
        try:
            start_idx = lower_text.index("<action>")
            end_idx = lower_text.index("</action>", start_idx)
            tag_length = len("<action>")
            return text[start_idx + tag_length:end_idx]
        except ValueError:
            return "__parse_error__"

    @staticmethod
    def _normalize_action_text(action: str | None) -> str:
        """Normalize action strings for counting."""
        if action is None:
            return "__invalid__"
        normalized = action.strip().lower()
        return normalized if normalized else "__invalid__"

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
