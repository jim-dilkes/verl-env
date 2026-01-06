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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from collections import defaultdict
from functools import partial

import numpy as np
from pprint import pprint
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from typing import Dict, Optional
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.envs.captioners import make_captioner
from verl.envs.environments import make_env
from verl.envs.vec_env import VecEnv
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
    bootstrap_metric,
    calc_maj_val,
    process_validation_metrics,
)
from verl.trainer.ppo.rollout_corr_helper import (
    apply_rollout_correction,
    compute_rollout_correction_and_add_to_batch,
)
from verl.trainer.ppo.multi_env_evaluator import MultiEnvEvaluator
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn as rl_collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {
        'actor/reward_kl_penalty': current_kl, 
        'actor/reward_kl_penalty_coeff': beta,
    }

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, step_gamma=1.0, step_lam=1.0, token_gamma=1.0, token_lam=1.0, n_rollouts=1, group_all=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_multistep_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            token_gamma=token_gamma,
            step_gamma=step_gamma,
            dones=data.batch["done"],
            token_lam=token_lam,
            step_lam=step_lam,
            n_rollouts=n_rollouts,
            frozen_mask=data.batch.get("frozen_mask"),
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO: 
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            group_all=group_all)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

@contextmanager
def _timer_accumulate(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.0) + timer.last


class RayMultistepTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(role_worker_mapping)
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        self.use_rm = need_reward_model(role_worker_mapping)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger()

        
        self.freeze_completed_episodes = getattr(self.config.envs, 'freeze_completed_episodes', False)

        # Seed management for environment resets
        group_initial_seed_config = getattr(config.envs, 'group_initial_seed', 'none')
        if group_initial_seed_config == "random":
            # Generate a random initial seed
            import random
            self.env_seed = random.randint(0, 2**31 - 1)
            print(f"Using random initial seed: {self.env_seed}")
        elif isinstance(group_initial_seed_config, int):
            # Use the provided integer seed
            self.env_seed = group_initial_seed_config
            print(f"Using preset initial seed: {self.env_seed}")
        else:
            # No seeding (None or other values)
            self.env_seed = None
            print("No environment seeding enabled")
        
        self.env_seed_counter = 0

        # If GRPO is used, freeze_completed_episodes must be true and explicit env_seed must be set
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
            self.seed_group_size = getattr(self.config.envs, 'group_rollout_size', None)
            if self.seed_group_size is None:
                self.seed_group_size = self.config.envs.n_rollouts
                print(f"Warning: seed_group_size not set, using n_rollouts ({self.seed_group_size}), all rollouts will be in the same group")
            if not self.freeze_completed_episodes:
                self.freeze_completed_episodes = True
                print(f"Warning: freeze_completed_episodes is false but is required for GRPO, setting to True")
            if self.env_seed is None:
                self.env_seed = 0
                print(f"Warning: env_seed not set but is required for GRPO, using 0")
        else:
            self.seed_group_size = None
            

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self.use_critic = need_critic(self.config)
        if self.use_critic:
            print(f"[RayMultistepTrainer] Using {self.config.algorithm.adv_estimator} with critic enabled")
        else:
            print(f"[RayMultistepTrainer] Using {self.config.algorithm.adv_estimator} without critic")

        self.critic_warmup_micro_batch_size_per_gpu = getattr(
            self.config.trainer, "critic_warmup_micro_batch_size_per_gpu", None
        )

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        self.env = self._make_vec_env(
            config.envs.env_name, 
            config.envs.task, 
            config, 
            render_mode=None
        )
        
        # Reuse training environment for validation to save memory
        # No need to create a separate val_env
        self.val_env = self.env
        
        # Initialize multi-environment evaluator if evaluation config is present
        print(f"[RayPPOTrainer] Checking evaluation config...")
        print(f"[RayPPOTrainer] hasattr(config, 'evaluation'): {hasattr(config, 'evaluation')}")
        print(f"[RayPPOTrainer] hasattr(config, 'eval'): {hasattr(config, 'eval')}")
        
        # Check both possible locations for evaluation config
        eval_config = None
        if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'environments'):
            eval_config = config.evaluation
            print(f"[RayPPOTrainer] Found evaluation config at config.evaluation")
        elif hasattr(config, 'eval') and hasattr(config.eval, 'evaluation') and hasattr(config.eval.evaluation, 'environments'):
            eval_config = config.eval.evaluation
            print(f"[RayPPOTrainer] Found evaluation config at config.eval.evaluation")
        
        if eval_config is not None:
            print(f"[RayPPOTrainer] Initializing MultiEnvEvaluator with {len(eval_config.environments)} environments")
            self.multi_env_evaluator = MultiEnvEvaluator(
                config=config,
                tokenizer=tokenizer,
                actor_rollout_wg=None,  # Will be set after init_workers
                val_reward_fn=val_reward_fn,
                eval_config=eval_config  # Pass the evaluation config directly
            )
        else:
            print(f"[RayPPOTrainer] No evaluation config found, setting multi_env_evaluator to None")
            self.multi_env_evaluator = None

    def get_next_env_seed(self):
        """Get the next seed for environment reset, incrementing the counter."""
        if self.env_seed is not None:
            # Use modulo to ensure seed stays within valid range [0, 2^32 - 1]
            seed = (self.env_seed + self.env_seed_counter) % (2**32)
            self.env_seed_counter += 10_000
            return seed
        return None

    def _make_vec_env(self, env_name, task, config, render_mode=None):
        """Create a vectorized environment."""
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

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        warmup_micro_batch_size = getattr(config.trainer, "critic_warmup_micro_batch_size_per_gpu", None)
        if self.use_critic and warmup_micro_batch_size is not None:
            assert warmup_micro_batch_size > 0, "trainer.critic_warmup_micro_batch_size_per_gpu must be positive"
            assert not config.critic.use_dynamic_bsz, (
                "trainer.critic_warmup_micro_batch_size_per_gpu is not supported when critic.use_dynamic_bsz is True"
            )
            assert (
                config.critic.ppo_mini_batch_size % warmup_micro_batch_size == 0
            ), (
                "trainer.critic_warmup_micro_batch_size_per_gpu must divide critic.ppo_mini_batch_size "
                f"(got {config.critic.ppo_mini_batch_size} vs {warmup_micro_batch_size})"
            )

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(
        self,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
    ):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type

        if train_dataset is None:
            if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
                dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
                if not issubclass(dataset_cls, Dataset):
                    raise TypeError(
                        f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                        f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset"
                    )
            else:
                dataset_cls = RLHFDataset

            train_dataset = dataset_cls(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )

        if val_dataset is None:
            if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
                dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
                if not issubclass(dataset_cls, Dataset):
                    raise TypeError(
                        f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                        f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset"
                    )
            else:
                dataset_cls = RLHFDataset

            val_dataset = dataset_cls(
                data_files=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            if self.config.data.shuffle:
                train_dataloader_generator = torch.Generator()
                train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
                train_sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
            else:
                train_sampler = SequentialSampler(data_source=self.train_dataset)

        if collate_fn is None:
            collate_fn = rl_collate_fn

        num_workers = self.config.data.get("dataloader_num_workers", 8)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert (
            len(self.val_dataloader) == 1
        ), "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps, table_name="val_gen/generations")

    def _validate(self):

        max_seq_len = self.config.data.max_prompt_length
        # For validation, use the base seed if configured, or None for random
        # Ensure seed is in valid range [0, 2^32 - 1] for NumPy RandomState
        if self.env_seed is not None:
            # Use a large offset to differentiate validation from training seeds
            # Use modulo to ensure it stays within valid range
            val_seed = (self.env_seed + 2**31) % (2**32)
        else:
            val_seed = 0
        val_obs, _ = self.val_env.reset(seed=val_seed, use_incremental_seeds=True)
        
        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        end_of_traj = None
        rew_of_traj = 0.
        len_of_traj = 0.
        
        while True:
            
            self.tokenizer.padding_side = "left"
            val_input_obs_text = self.tokenizer.apply_chat_template(val_obs, tokenize=False, add_generation_prompt=True) #, enable_thinking=True)
            sample_inputs.extend(val_input_obs_text)
            val_input_obs = self.tokenizer(val_input_obs_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_seq_len)
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
            
            # val_gen_batch.meta_info = {
            #     'eos_token_id': self.tokenizer.eos_token_id,
            #     'pad_token_id': self.tokenizer.pad_token_id,
            #     'recompute_log_prob': False,
            #     'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            #     'validate': True,
            # }
            
            val_gen_batch.meta_info["step"] = None
            val_gen_batch_output = self.actor_rollout_wg.generate_sequences(val_gen_batch)
                        
            response_ids = val_gen_batch_output.batch['responses']
            actions = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            sample_outputs.extend(actions)
            
            val_obs, val_reward, val_terminated, val_truncated, _ = self.val_env.step(actions)
            
            if end_of_traj is None:
                end_of_traj = np.logical_or(val_terminated, val_truncated)
                rew_of_traj = val_reward
                len_of_traj = np.ones_like(val_reward)
                pos_rew_of_traj = (np.array(val_reward) > 0.) * 1.
            else:
                done = np.logical_or(val_terminated, val_truncated)
                rew_of_traj += val_reward * (~end_of_traj).astype(np.float32)
                len_of_traj += (~end_of_traj).astype(np.float32)
                end_of_traj = np.logical_or(end_of_traj, done)
                pos_rew_of_traj = np.logical_or(pos_rew_of_traj, (np.array(val_reward) > 0.) * 1.)
            sample_scores.extend(rew_of_traj)
            
            if end_of_traj.all():
                break
        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        
        succ_of_traj = (np.array(rew_of_traj) > 0.) * 1.
        
        metric_dict = {
            "val/rewards_mean": rew_of_traj.mean(),
            "val/pos_reward_total_prop_mean": succ_of_traj.mean(),
            "val/traj_length_mean": len_of_traj.mean(),
            "val/pos_reward_any_prop_mean": pos_rew_of_traj.mean(),
            "val/rewards_std": rew_of_traj.std(),
            "val/pos_reward_total_prop_std": succ_of_traj.std(),
            "val/traj_length_std": len_of_traj.std(),
            "val/pos_reward_any_prop_std": pos_rew_of_traj.std(),
        }

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        
        # Set the actor_rollout_wg reference in the multi-environment evaluator
        if self.multi_env_evaluator is not None:
            print(f"[RayPPOTrainer] Setting actor_rollout_wg in multi_env_evaluator")
            self.multi_env_evaluator.actor_rollout_wg = self.actor_rollout_wg

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          group=self.config.trainer.get('group', None))

        rollout_corr_config = getattr(self.config.algorithm, "rollout_correction", None)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            print(f"[RayPPOTrainer] Starting initial validation at global_step={self.global_steps}")
            if self.multi_env_evaluator is not None:
                print(f"[RayPPOTrainer] Using multi_env_evaluator initial evaluation")
                # Close training env temporarily to free memory for multi-env evaluation
                print(f"[RayPPOTrainer] Closing training env to free memory for multi-env evaluation")
                self.env.close()
                
                evaluation_metrics = self.multi_env_evaluator.evaluate(self.global_steps)
                print(f"[RayPPOTrainer] Initial evaluation completed. Metrics keys: {list(evaluation_metrics.keys())}")
                pprint(f'Initial evaluation metrics: {evaluation_metrics}')
                print(f"[RayPPOTrainer] Logging {len(evaluation_metrics)} metrics to logger")
                logger.log(data=evaluation_metrics, step=self.global_steps)
                
                # Recreate training env after multi-env evaluation
                print(f"[RayPPOTrainer] Recreating training env after multi-env evaluation")
                self.env = self._make_vec_env(
                    self.config.envs.env_name, 
                    self.config.envs.task, 
                    self.config, 
                    render_mode=None
                )
                self.val_env = self.env  # Keep val_env as reference to training env

            print(f"[RayPPOTrainer] Using standard _validate for initial validation")
            validation_metrics = self._validate()
            print(f"[RayPPOTrainer] Initial validation completed. Metrics keys: {list(validation_metrics.keys())}")
            pprint(f'Initial validation metrics: {validation_metrics}')
            print(f"[RayPPOTrainer] Logging {len(validation_metrics)} metrics to logger")
            logger.log(data=validation_metrics, step=self.global_steps)

            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = {}
        
        obs_vec, info_vec = self.env.reset()
        
        for epoch in range(self.config.trainer.total_epochs):
            
            # Reset environments with appropriate seeding strategy
            if self.env_seed is not None:
                seed = self.get_next_env_seed()
                # Use seed_group_size for GRPO to ensure proper grouping
                obs_vec, info_vec = self.env.reset(seed=seed, seed_group_size=self.seed_group_size)
        
            self.critic_warmup_step = self.config.trainer.critic_warmup # TODO: move to the config file
            if self.global_steps <= self.critic_warmup_step:
                bsize = self.config.data.train_batch_size * self.config.trainer.critic_warmup
            else:
                bsize = self.config.data.train_batch_size
                
            if self.global_steps == 1 or self.global_steps > self.critic_warmup_step:
                esize = self.config.envs.n_rollouts
                plen = self.config.data.max_prompt_length
                rlen = self.config.data.max_response_length
                meta_size = bsize + esize
                batch_dict = {
                    "input_ids": torch.zeros([bsize + esize, plen + rlen], dtype=torch.int64),
                    "attention_mask": torch.zeros([bsize + esize, plen + rlen], dtype=torch.int64),
                    "position_ids": torch.zeros([bsize + esize, plen + rlen], dtype=torch.int64),
                    "responses": torch.zeros([bsize + esize, rlen], dtype=torch.int64),
                    "reward": torch.zeros([bsize + esize], dtype=torch.float64),
                    "done": torch.zeros([bsize + esize], dtype=torch.float64),
                    "data_source": np.zeros([meta_size]),
                    "ability": np.zeros([meta_size]),
                    "reward_model": np.zeros([meta_size]),
                    "extra_info": np.zeros([meta_size]),
                    "raw_prompt_ids": np.zeros([meta_size]),
                    "index": np.zeros([meta_size]),
                    "frozen_mask": torch.zeros([bsize + esize], dtype=torch.int64),
                }
    
            metrics = {}
            timing_raw = {}
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            
            is_last_step = self.global_steps >= self.total_training_steps
            
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            max_seq_len = self.config.data.max_prompt_length # TODO: query from config
            
            with _timer('step', timing_raw):

                assert self.config.data.train_batch_size % self.config.envs.n_rollouts == 0, \
                    f"train_batch_size ({self.config.data.train_batch_size}) must be divisible by n_rollouts ({self.config.envs.n_rollouts})."
                episode_len = bsize // self.config.envs.n_rollouts
                
                if self.global_steps == 1 or self.global_steps > self.critic_warmup_step:
                    
                    # Initialize episode tracking for freezing logic
                    # This prevents cross-batch episode issues by freezing environments when episodes complete
                    # Frozen environments receive "__SKIP__" actions and return cached data
                    if self.freeze_completed_episodes:
                        # Track which environments have completed episodes
                        env_frozen = np.zeros(self.config.envs.n_rollouts, dtype=bool)
                    
                    for time_step in range(episode_len+1):
                        
                        # TODO: move this to a function 
                        
                        with _timer_accumulate('text_gen_proc', timing_raw):
                            self.tokenizer.padding_side = "left"
                            input_obs = self.tokenizer.apply_chat_template(obs_vec, tokenize=False, add_generation_prompt=True) #, enable_thinking=True)
                            
                            input_obs = self.tokenizer(input_obs, return_tensors='pt', padding='max_length', truncation=True, max_length=max_seq_len)
                                    
                            input_ids = input_obs['input_ids']
                            attention_mask = input_obs['attention_mask']
                            position_ids = attention_mask.long().cumsum(-1) - 1
                            position_ids.masked_fill_(attention_mask == 0, 1)
                            
                            obs_data = {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask,
                                'position_ids': position_ids,
                            }
                            gen_batch = DataProto.from_dict(tensors=obs_data)
                            
                            if time_step == episode_len:
                                batch.insert(
                                    gen_batch,
                                    start_idx = time_step * self.config.envs.n_rollouts,
                                    end_idx = (time_step + 1) * self.config.envs.n_rollouts,
                                    diff_size=True,
                                )
                                break
                                
                            gen_batch.meta_info["step"] = time_step if time_step < episode_len - 1 else -1
                            
                            # Handle episode freezing logic
                            if self.freeze_completed_episodes:
                                # Only generate actions for non-frozen environments
                                active_envs = ~env_frozen
                                if np.any(active_envs):
                                    # Filter observations for active environments only
                                    active_obs_data = {}
                                    for key in gen_batch.batch.keys():
                                        active_obs_data[key] = gen_batch.batch[key][active_envs]
                                    
                                    active_gen_batch = DataProto.from_dict(tensors=active_obs_data)
                                    active_gen_batch.meta_info = gen_batch.meta_info.copy()
                                    
                                    # Pad the batch to be divisible by the number of GPUs
                                    dp_size = self.actor_rollout_wg.world_size
                                    active_gen_batch_padded, pad_size = pad_dataproto_to_divisor(active_gen_batch, dp_size)
                                    
                                    with _timer_accumulate('text_gen', timing_raw):
                                        active_gen_batch_output = self.actor_rollout_wg.generate_sequences(active_gen_batch_padded)
                                    
                                    # Remove padding from the output
                                    if pad_size > 0:
                                        active_gen_batch_output = unpad_dataproto(active_gen_batch_output, pad_size)
                                    
                                    # Decode actions for active environments
                                    active_response_ids = active_gen_batch_output.batch['responses']
                                    active_actions = self.tokenizer.batch_decode(active_response_ids, skip_special_tokens=True)
                                
                                    # Create full action array with skip actions for frozen environments
                                    actions = ['__SKIP__'] * self.config.envs.n_rollouts
                                    active_idx = 0
                                    for i in range(self.config.envs.n_rollouts):
                                        if active_envs[i]:
                                            actions[i] = active_actions[active_idx]
                                            active_idx += 1
                                else:
                                    actions = ['__SKIP__'] * self.config.envs.n_rollouts
                            else:
                                # Original behavior when freezing is disabled
                                with _timer_accumulate('text_gen', timing_raw):
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                response_ids = gen_batch_output.batch['responses']
                                actions = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                                active_envs = np.ones(self.config.envs.n_rollouts, dtype=bool)
                        
                        with _timer_accumulate('env_step', timing_raw):
                            obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = self.env.step(actions)

                        # Collect metrics from each environment's info for this step
                        # Later these are used to calculate mean value of the metrics per executed step across all environments
                        for active_flag, info in zip(active_envs, info_vec):
                            if active_flag:
                                for key, value in info['metrics'].items():
                                    if key in metrics:
                                        metrics[key].append(value)
                                    else:
                                        metrics[key] = [value]
                        
                        done_vec = np.logical_or(terminated_vec, truncated_vec)
                        
                        # Handle batch insertion based on freezing logic
                        if self.freeze_completed_episodes:
                            # Create batch output for all environments (including frozen ones)
                            if 'active_gen_batch_output' in locals() and np.any(active_envs):
                                # Create full batch output with correct dimensions from the start
                                # We need to create a batch with n_rollouts elements, preserving the exact ordering
                                
                                # First, create the full batch structure with correct dimensions
                                full_batch_dict = {}
                                
                                # Initialize all tensors with the correct batch size (n_rollouts)
                                for key in active_gen_batch_output.batch.keys():
                                    # Create full tensor with correct dimensions for all keys
                                    full_data = torch.zeros((self.config.envs.n_rollouts, *active_gen_batch_output.batch[key].shape[1:]), dtype=active_gen_batch_output.batch[key].dtype)
                                    
                                    # Fill in the data for active environments at their correct indices
                                    active_idx = 0
                                    for i in range(self.config.envs.n_rollouts):
                                        if active_envs[i]:
                                            full_data[i] = active_gen_batch_output.batch[key][active_idx]
                                            active_idx += 1
                                        # For frozen environments, leave as zeros (must be 0 for attention mask, otherwise just dummy data)
                                    
                                    full_batch_dict[key] = full_data
                                
                                # Create the full batch output with correct dimensions and preserved ordering
                                full_batch_output = DataProto.from_dict(tensors=full_batch_dict)
                                full_batch_output.meta_info = active_gen_batch_output.meta_info.copy()
                                
                                # Set done and reward for all environments
                                # For frozen environments, ensure done=True for proper GAE behavior
                                full_batch_output.batch["done"] = torch.tensor(done_vec, dtype=torch.float64)
                                full_batch_output.batch["reward"] = torch.tensor(reward_vec, dtype=torch.float64)
                                full_batch_output.batch["frozen_mask"] = torch.tensor(env_frozen, dtype=torch.int64)
                                
                                batch.insert(
                                    full_batch_output,
                                    start_idx = time_step * self.config.envs.n_rollouts,
                                    end_idx = (time_step + 1) * self.config.envs.n_rollouts,
                                )
                            else:
                                # All environments frozen, create dummy batch with correct dimensions
                                # Create dummy batch with proper dimensions for all environments
                                dummy_batch_dict = {}
                                
                                # Create dummy tensors with correct dimensions
                                # The batch expects specific dimensions: plen + rlen for input tensors, rlen for responses
                                plen = self.config.data.max_prompt_length
                                rlen = self.config.data.max_response_length
                                
                                for key in gen_batch.batch.keys():
                                    if key == 'responses':
                                        # Responses should have max_response_length dimension
                                        dummy_batch_dict[key] = torch.zeros((self.config.envs.n_rollouts, rlen), dtype=torch.long)
                                    else:
                                        # For other keys (input_ids, attention_mask, position_ids), use plen + rlen dimensions
                                        dummy_batch_dict[key] = torch.zeros((self.config.envs.n_rollouts, plen + rlen), dtype=gen_batch.batch[key].dtype)
                                
                                # Add the required fields
                                dummy_batch_dict["done"] = torch.zeros(self.config.envs.n_rollouts, dtype=torch.float64)
                                dummy_batch_dict["reward"] = torch.zeros(self.config.envs.n_rollouts, dtype=torch.float64)
                                dummy_batch_dict["frozen_mask"] = torch.ones(self.config.envs.n_rollouts, dtype=torch.int64)
                                
                                # Create the dummy batch
                                dummy_batch = DataProto.from_dict(tensors=dummy_batch_dict)
                                dummy_batch.meta_info = gen_batch.meta_info.copy()
                                
                                batch.insert(
                                    dummy_batch,
                                    start_idx = time_step * self.config.envs.n_rollouts,
                                    end_idx = (time_step + 1) * self.config.envs.n_rollouts,
                                )
                        else:
                            # Original behavior when freezing is disabled
                            gen_batch_output.batch["done"] = torch.tensor(done_vec, dtype=torch.float64)
                            gen_batch_output.batch["reward"] = torch.tensor(reward_vec, dtype=torch.float64)
                            gen_batch_output.batch["frozen_mask"] = torch.zeros(self.config.envs.n_rollouts, dtype=torch.int64)

                            if self.config.envs.group_rollout_size is not None and type(self.config.envs.group_rollout_size) == int:
                                gen_batch_output.batch["uid"] = np.array([i // self.config.envs.group_rollout_size for i in range(self.config.envs.n_rollouts)])
                            
                            batch.insert(
                                gen_batch_output,
                                start_idx = time_step * self.config.envs.n_rollouts,
                                end_idx = (time_step + 1) * self.config.envs.n_rollouts,
                            )
                        
                        # Update any newly completed episodes to be frozen
                        if self.freeze_completed_episodes:
                            # Freeze environments that have completed episodes
                            env_frozen = np.logical_or(env_frozen, done_vec)
                    
                    # merge batch metrics
                    for key in metrics.keys():
                        metrics[key] = np.mean(metrics[key]) # Mean per exectued step
                        
                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with _timer('gen_max', timing_raw):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info['do_sample'] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch['reward_baselines'] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output
                
                if self.config.envs.group_rollout_size is None:
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            dtype=object)
                else:
                    step_uids = np.array([i // self.config.envs.group_rollout_size for i in range(self.config.envs.n_rollouts)])
                    full_uids = np.repeat(step_uids, episode_len+1)
                    batch.non_tensor_batch['uid'] = full_uids

                # # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)
                assert self.config.actor_rollout_ref.rollout.n == 1, "For multi-turn rollout, we only support n=1"
    
                batch.batch['response_mask'] = compute_response_mask(batch)
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                # Operating Mode Selection:
                # - Bypass mode: Uses rollout_log_probs as anchor (no recompute)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor
                if bypass_recomputing_logprobs:
                    apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                else:
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        metrics.update({"actor/entropy": entropy_agg.detach().item()})
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer('ref', timing_raw):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer('adv', timing_raw):

                    # compute rewards. apply_kl_penalty if available
                    batch.batch['token_level_rewards'] = torch.zeros_like(batch.batch['response_mask'], dtype=torch.float64)
                    seq_len = batch.batch['response_mask'].sum(-1) - 1
                    indices = torch.arange(batch.batch['response_mask'].shape[0], device=seq_len.device)
                    batch.batch['token_level_rewards'][indices, seq_len] = batch.batch['reward']
                    batch.batch['token_level_scores'] = batch.batch['token_level_rewards'].clone() 
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch,
                                                                kl_ctrl=self.kl_ctrl_in_reward,
                                                                kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)

                    if rollout_corr_config is not None and 'rollout_log_probs' in batch.batch and not bypass_recomputing_logprobs:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        metrics.update(is_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                                adv_estimator=self.config.algorithm.adv_estimator,
                                                step_gamma=self.config.algorithm.step_gamma,
                                                step_lam=self.config.algorithm.step_lam,
                                                token_gamma=self.config.algorithm.token_gamma,
                                                token_lam=self.config.algorithm.token_lam,
                                                n_rollouts=self.config.envs.n_rollouts,
                                                group_all=self.config.envs.group_rollout_size is None)

                if self.global_steps > self.critic_warmup_step:
                    batch4train = deepcopy(batch)
                    batch4train.batch = batch4train.batch[:bsize].contiguous()
                    for key in batch4train.non_tensor_batch.keys():
                        batch4train.non_tensor_batch[key] = batch4train.non_tensor_batch[key][:bsize]
                    for key in batch4train.meta_info.keys():
                        if isinstance(batch4train.meta_info[key], list):
                            batch4train.meta_info[key] = batch4train.meta_info[key][:bsize]
                else:
                    batch4train = deepcopy(batch)
                    random_len = self.config.data.train_batch_size * 10
                    random_indices = torch.randperm(bsize)[:random_len]
                    batch4train.batch = batch4train.batch[random_indices].contiguous()
                    for key in batch4train.non_tensor_batch.keys():
                        batch4train.non_tensor_batch[key] = batch4train.non_tensor_batch[key][random_indices]
                    for key in batch4train.meta_info.keys():
                        if isinstance(batch4train.meta_info[key], list):
                            batch4train.meta_info[key] = [batch4train.meta_info[key][i.item()] for i in random_indices]


                # update critic
                if self.use_critic:
                    # Add flag to indicate if we're in critic warmup phase
                    is_critic_warmup = self.global_steps <= self.critic_warmup_step
                    batch4train.meta_info['is_critic_warmup'] = is_critic_warmup
                    warmup_micro_batch_size = None
                    if is_critic_warmup and self.critic_warmup_micro_batch_size_per_gpu is not None:
                        warmup_micro_batch_size = self.critic_warmup_micro_batch_size_per_gpu
                        batch4train.meta_info['critic_micro_batch_size_per_gpu'] = warmup_micro_batch_size
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(batch4train)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                    metrics.update(critic_output_metrics)
                    if warmup_micro_batch_size is not None:
                        batch4train.meta_info.pop('critic_micro_batch_size_per_gpu', None)
                    # Clear the warmup flag after use to avoid memory issues
                    if 'is_critic_warmup' in batch4train.meta_info:
                        del batch4train.meta_info['is_critic_warmup']
                # implement critic warmup
                if self.critic_warmup_step <= self.global_steps:
                    # update actor
                    # Pass step info for entropy bounds decay
                    batch4train.meta_info['global_step'] = self.global_steps
                    batch4train.meta_info['total_training_steps'] = self.total_training_steps
                    with _timer('update_actor', timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch4train)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                # validate
                # Use evaluation test_freq if available, otherwise fall back to trainer test_freq
                eval_test_freq = getattr(self.config.evaluation, 'test_freq', None) if hasattr(self.config, 'evaluation') else None
                test_freq = eval_test_freq if eval_test_freq is not None else self.config.trainer.test_freq
                
                if self.val_reward_fn is not None and test_freq > 0 and \
                    (is_last_step or self.global_steps % test_freq == 0) and (self.global_steps > self.critic_warmup_step):
                    print(f"[RayPPOTrainer] Starting periodic validation at global_step={self.global_steps}")
                    with _timer('testing', timing_raw):
                        if self.multi_env_evaluator is not None:
                            # Close training env temporarily to free memory for multi-env evaluation
                            print(f"[RayPPOTrainer] Closing training env to free memory for multi-env evaluation")
                            self.env.close()
                            
                            evaluation_metrics: dict = self.multi_env_evaluator.evaluate(self.global_steps)
                            print(f"[RayPPOTrainer] Periodic evaluation completed. Metrics keys: {list(evaluation_metrics.keys())}")
                            logger.log(data=evaluation_metrics, step=self.global_steps)
                            if is_last_step:
                                last_val_metrics.update(evaluation_metrics)
                            
                            # Recreate training env after multi-env evaluation
                            print(f"[RayPPOTrainer] Recreating training env after multi-env evaluation")
                            self.env = self._make_vec_env(
                                self.config.envs.env_name, 
                                self.config.envs.task, 
                                self.config, 
                                render_mode=None
                            )
                            self.val_env = self.env  # Keep val_env as reference to training env
                        
                        validation_metrics: dict = self._validate()
                        print(f"[RayPPOTrainer] Periodic validation completed. Metrics keys: {list(validation_metrics.keys())}")
                        if is_last_step:
                            last_val_metrics.update(validation_metrics)
                    metrics.update(validation_metrics)

                # Save checkpoint periodically or always save final checkpoint
                if (self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0) or is_last_step:
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f'Final validation metrics: {last_val_metrics}')
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
            
        self.env.close()
