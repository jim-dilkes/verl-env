#!/usr/bin/env python3
"""
Standalone environment evaluation script - ROLLOUT ONLY.

This bypasses the full PPO trainer and only loads the vLLM rollout engine,
avoiding the GPU memory overhead of actor/critic FSDP models.

Usage:
  python scripts/standalone_env_eval.py \
    model.path=Qwen/Qwen3-4B-Instruct \
    rollout.gpu_memory_utilization=0.8 \
    envs.n_rollouts=10
"""

import os
import sys

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import ray
from omegaconf import OmegaConf, DictConfig
from pprint import pprint

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils import hf_tokenizer


@ray.remote(num_cpus=1)
class EvalRunner:
    """Runs environment evaluation with rollout-only worker."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
    def run(self):
        from verl.trainer.ppo.multi_env_evaluator import MultiEnvEvaluator
        from verl.envs.make_env import make_env
        
        pprint(OmegaConf.to_container(self.config, resolve=True))
        
        # Load tokenizer
        tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        
        # Create rollout-only worker group
        # KEY: role='rollout' - only loads vLLM, not FSDP actor
        print("\n=== Creating ROLLOUT-ONLY worker (no actor/critic) ===")
        
        rollout_config = OmegaConf.create({
            'model': self.config.model,
            'rollout': self.config.rollout,
        })
        
        ray_cls = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=rollout_config,
            role='rollout'  # <-- THIS IS THE KEY: rollout only, no actor
        )
        
        resource_pool = RayResourcePool(
            process_on_nodes=[self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes
        )
        
        rollout_wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls,
        )
        
        print("Initializing rollout model...")
        rollout_wg.init_model()
        print("Rollout model initialized!")
        
        # Create evaluator
        eval_config = OmegaConf.to_container(
            self.config.get('evaluation', {}), 
            resolve=True
        )
        
        evaluator = MultiEnvEvaluator(
            config=self.config,
            tokenizer=tokenizer,
            actor_rollout_wg=rollout_wg,
            val_reward_fn=None,  # Use env rewards
            eval_config=eval_config,
        )
        
        # Run evaluation
        print("\n=== Running Evaluation ===")
        metrics = evaluator.evaluate(global_step=0)
        
        print("\n=== Evaluation Results ===")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")
        
        # Cleanup
        evaluator.close()
        
        return metrics


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    # Initialize Ray
    if not ray.is_initialized():
        ray_init_kwargs = OmegaConf.to_container(
            config.ray_kwargs.get("ray_init", {}),
            resolve=True
        )
        ray_init_kwargs.setdefault("num_cpus", 8)
        ray_init_kwargs.setdefault("num_gpus", 1)
        print(f"Ray init: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)
    
    # Run evaluation
    runner = EvalRunner.remote(config)
    metrics = ray.get(runner.run.remote())
    
    print("\n=== DONE ===")
    return metrics


if __name__ == "__main__":
    main()
