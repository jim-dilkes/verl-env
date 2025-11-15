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

import numpy as np
import time
import gc
import psutil
import os
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
        
        # Create sample tuple (input, output, score) as expected by ValidationGenerationsLogger
        sample = (formatted_input, formatted_output, total_score)
        
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
    
    def _evaluate_single_env(self, val_env, env_config, env_name):
        """
        Run evaluation for a single environment.
        
        This method implements the core evaluation logic, similar to the
        original _validate method but adapted for multi-environment use.
        
        Args:
            val_env: The vectorized environment to evaluate
            env_config: Environment-specific configuration
            env_name: Name of the environment for logging
            
        Returns:
            tuple: (dict of metrics, dict episode_data)
        """
        max_seq_len = self.config.data.max_prompt_length
        
        # Use environment-specific seed if configured
        initial_seed = env_config.get('initial_seed', None)
        val_obs, val_info = val_env.reset(seed=initial_seed, use_incremental_seeds=True)
        
        # Lists to collect samples for logging
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        end_of_traj = None
        rew_of_traj = 0.
        score_of_traj = 0.
        len_of_traj = 0.
        total_tokens_generated = 0
        total_inference_time = 0.0
        
        # Track one full episode for logging (first rollout)
        episode_inputs = []
        episode_outputs = []
        episode_total_score = 0.0
        episode_tracked = False

        # Get generation config for this environment (once per environment)
        self._current_gen_config = self._get_generation_config(env_config)

        for j in range(env_config['episode_length']):
        
            self.tokenizer.padding_side = "left"
            val_input_obs_text = self.tokenizer.apply_chat_template(
                val_obs, tokenize=False, add_generation_prompt=True
            )
            sample_inputs.extend(val_input_obs_text)
            
            # Track episode data for the first rollout (index 0)
            if not episode_tracked:
                episode_inputs.append(val_input_obs_text[0])  # First rollout's input
            
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
            
            # Apply generation config to meta_info (temperature, top_p, etc.)
            # This was set once at the start of the evaluation for this environment
            for key, value in self._current_gen_config.items():
                val_gen_batch.meta_info[key] = value
            
            # Generate actions using the shared policy (measure inference time)
            inference_start = time.time()
            val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
            inference_end = time.time()
            total_inference_time += (inference_end - inference_start)
            
            response_ids = val_gen_batch_output.batch['responses']
            actions = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            sample_outputs.extend(actions)
            
            # Track episode data for the first rollout (index 0)
            if not episode_tracked:
                episode_outputs.append(actions[0])  # First rollout's action
            
            # Count tokens generated (sum of response lengths)
            response_lengths = (response_ids != self.tokenizer.pad_token_id).sum(dim=-1)
            total_tokens_generated += response_lengths.sum().item()
            
            # Step the environment
            try:
                val_obs, val_reward, val_terminated, val_truncated, val_info = val_env.step(actions)
            except Exception as e:
                print(f"[ERROR] MultiEnvEvaluator: Exception in val_env.step: {e}")
                import traceback
                print(f"[ERROR] MultiEnvEvaluator: Traceback: {traceback.format_exc()}")
                raise
            
            # Track trajectory metrics
            if end_of_traj is None:
                end_of_traj = np.logical_or(val_terminated, val_truncated)
                rew_of_traj = val_reward
                score_of_traj = np.where(val_reward > 0., val_reward, 0.)
                len_of_traj = np.ones_like(val_reward)
                pos_rew_of_traj = (np.array(val_reward) > 0.) * 1.
            else:
                done = np.logical_or(val_terminated, val_truncated)
                rew_of_traj += val_reward * (~end_of_traj).astype(np.float32)
                score_of_traj += np.where(val_reward > 0., val_reward * (~end_of_traj).astype(np.float32), 0.) 
                len_of_traj += (~end_of_traj).astype(np.float32)
                end_of_traj = np.logical_or(end_of_traj, done)
                pos_rew_of_traj = np.logical_or(pos_rew_of_traj, (np.array(val_reward) > 0.) * 1.)
            
            sample_scores.extend(rew_of_traj)
            
            # Track episode data for the first rollout (index 0)
            if not episode_tracked:
                episode_total_score += val_reward[0]  # First rollout's reward
                # Check if the first rollout's episode is done
                if val_terminated[0] or val_truncated[0] or j == env_config['episode_length'] - 1:
                    episode_tracked = True  # Stop tracking after episode completion
            
            # Check if all episodes are done
            if end_of_traj.all():
                break
        
        # Compute final metrics - ensure all arrays are float type for mean calculations
        rew_of_traj = np.array(rew_of_traj, dtype=np.float64)
        score_of_traj = np.array(score_of_traj, dtype=np.float64)
        len_of_traj = np.array(len_of_traj, dtype=np.float64)
        pos_rew_of_traj = np.array(pos_rew_of_traj, dtype=np.float64)
        succ_of_traj = (rew_of_traj > 0.) * 1.
        response_lengths = response_lengths.float()  # Convert tensor to float
        
        metric_dict = {
            "rewards_mean": float(rew_of_traj.mean()),
            "rewards_std": float(rew_of_traj.std()),
            "score_mean": float(score_of_traj.mean()),
            "score_std": float(score_of_traj.std()),
            "pos_reward_any_prop_mean": float(pos_rew_of_traj.mean()),
            "pos_reward_any_prop_std": float(pos_rew_of_traj.std()),
            "traj_length_mean": float(len_of_traj.mean()),
            "traj_length_std": float(len_of_traj.std()),
            "toks_out_mean": float(response_lengths.mean()),
            "toks_out_std": float(response_lengths.std()),
            "inference_time_seconds": total_inference_time,
            "inference_time_per_rollout": total_inference_time / env_config['n_rollouts'],
            "inference_time_per_step": total_inference_time / (env_config['episode_length'] * env_config['n_rollouts']),
            "tokens_per_rollout": total_tokens_generated / env_config['n_rollouts'],
            "tokens_per_step": total_tokens_generated / (env_config['episode_length'] * env_config['n_rollouts']),
        }
        
        # Prepare episode data for logging
        episode_data = {
            'inputs': episode_inputs,
            'outputs': episode_outputs,
            'total_score': float(episode_total_score)
        } if episode_tracked else None
        
        return metric_dict, episode_data
