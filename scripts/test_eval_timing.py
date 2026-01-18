#!/usr/bin/env python3
"""
Test script to measure evaluation timing with mock LLM generation.
Identifies bottlenecks in batched evaluation without needing actual GPU inference.
"""

import sys
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf

# Add verl to path
sys.path.insert(0, "/Users/jim/Projects/verl-env-2/.brisk/worktrees/fix/slow-eval-batching")

from verl import DataProto


class MockTokenizer:
    """Mock tokenizer that doesn't need any network access."""

    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.padding_side = "left"
        self.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def __call__(self, texts, return_tensors='pt', padding='max_length',
                 truncation=True, max_length=2048):
        """Tokenize texts into mock token IDs."""
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)
        # Create mock token IDs (just use character codes mod vocab_size)
        input_ids = []
        attention_masks = []

        for text in texts:
            # Simple mock tokenization: one token per character
            tokens = [ord(c) % self.vocab_size for c in text[:max_length]]
            # Pad or truncate
            if len(tokens) < max_length:
                pad_len = max_length - len(tokens)
                if self.padding_side == "left":
                    tokens = [self.pad_token_id] * pad_len + tokens
                    mask = [0] * pad_len + [1] * (max_length - pad_len)
                else:
                    tokens = tokens + [self.pad_token_id] * pad_len
                    mask = [1] * (max_length - pad_len) + [0] * pad_len
            else:
                tokens = tokens[:max_length]
                mask = [1] * max_length

            input_ids.append(tokens)
            attention_masks.append(mask)

        result = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks),
        }
        return result

    def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=True):
        """Apply chat template to conversations."""
        results = []
        for conv in conversations:
            # Simple concatenation of message contents
            text = ""
            for msg in conv:
                text += msg.get('content', '')
            if add_generation_prompt:
                text += "\nAssistant:"
            results.append(text)
        return results

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to strings."""
        results = []
        for ids in token_ids:
            # Simple mock decode
            chars = []
            for tid in ids.tolist():
                if tid == self.pad_token_id:
                    continue
                chars.append(chr(tid % 128))  # ASCII range
            results.append(''.join(chars))
        return results


class MockActorRolloutWG:
    """Mock actor that returns dummy responses without LLM inference."""

    def __init__(self, tokenizer, response_template: str = "<action>stay</action>"):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.call_count = 0

    def generate_sequences(self, batch: DataProto) -> DataProto:
        """Return mock responses."""
        self.call_count += 1
        batch_size = batch.batch['input_ids'].shape[0]

        # Simulate some inference time (very small, just to not be zero)
        time.sleep(0.001 * batch_size)

        # Generate dummy responses - create token IDs directly
        # Use simple encoding: one token per character
        response_tokens = [ord(c) % self.tokenizer.vocab_size for c in self.response_template]
        # Pad to 64 tokens
        max_len = 64
        if len(response_tokens) < max_len:
            response_tokens = response_tokens + [self.tokenizer.pad_token_id] * (max_len - len(response_tokens))
        else:
            response_tokens = response_tokens[:max_len]

        response_ids = torch.tensor([response_tokens] * batch_size)

        # Create output batch
        output = DataProto.from_dict(tensors={
            'responses': response_ids,
        })
        return output


def create_test_config(n_rollouts: int = 300, batch_size: int = 50, episode_length: int = 8):
    """Create a minimal config for testing."""
    config = OmegaConf.create({
        'envs': {
            'n_rollouts': n_rollouts,
            'episode_length': episode_length,
            'env_name': 'overcooked',
            'task': 'none',
            'freeze_completed_episodes': True,
            'duplication_mode': 'none',
            'format_penalty': 0.0,
            'binary_reward': False,
            'vec_env_multiprocessing': 'spawn',
            'group_initial_seed': 40000,
            'captioner': {
                'type': 'naive',
                'max_text_history': 0,
                'max_image_history': 0,
                'max_cot_history': 0,
            },
            'overcooked_kwargs': {
                'layout_name': 'cramped_room',
                'horizon': 15,
                'partner_policy': 'none',
                'shaped_reward': False,
                'pot_cook_time': 5,
                'print_visualization': False,
                'print_coordinates': True,
            },
        },
        'prompt': {
            'prompt': {
                'epsilon': 0.0,
                'multi_action_reasoning': False,
                'environment_instruction': 'Test instruction prompt for evaluation.',
            },
        },
        'data': {
            'max_prompt_length': 2048,
        },
        'trainer': {
            'log_val_generations': 0,
            'logger': ['console'],
        },
        'actor_rollout_ref': {
            'rollout': {
                'val_kwargs': {
                    'temperature': 1.25,
                    'top_p': 1.0,
                    'top_k': -1,
                    'min_p': 0.0,
                    'do_sample': True,
                },
            },
        },
    })
    return config


def create_eval_config(n_rollouts: int = 300, batch_size: int = 50, episode_length: int = 8):
    """Create eval config matching StateVisitation."""
    return {
        'name': 'Test-StateVisitation',
        'n_rollouts': n_rollouts,
        'batch_size': batch_size,
        'episode_length': episode_length,
        'initial_seed': 40000,
        'seed_group_size': 20,
        'freeze_completed_episodes': True,
        'duplication_mode': 'none',
        'env_name': 'overcooked',
        'task': 'none',
        'format_penalty': 0.0,
        'generation': {
            'temperature': 1.25,
            'top_p': 1.0,
            'top_k': -1,
            'min_p': 0.0,
            'do_sample': True,
        },
        'captioner': {
            'type': 'naive',
            'max_text_history': 0,
        },
        'overcooked_kwargs': {
            'layout_name': 'cramped_room',
            'horizon': 15,
            'partner_policy': 'none',
            'shaped_reward': False,
            'pot_cook_time': 5,
            'print_visualization': False,
            'print_coordinates': True,
        },
    }


def main():
    print("=" * 60)
    print("Evaluation Timing Test (Mock LLM)")
    print("=" * 60)

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rollouts', type=int, default=60, help='Number of rollouts (default: 60 for quick test)')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size (default: 20)')
    parser.add_argument('--episode-length', type=int, default=4, help='Episode length (default: 4)')
    parser.add_argument('--seed-group-size', type=int, default=10, help='Seed group size (default: 10)')
    args = parser.parse_args()

    print(f"\nConfig: n_rollouts={args.n_rollouts}, batch_size={args.batch_size}, "
          f"episode_length={args.episode_length}, seed_group_size={args.seed_group_size}")
    print(f"Expected batches: {(args.n_rollouts + args.batch_size - 1) // args.batch_size}")
    print(f"Expected n_groups: {args.n_rollouts // args.seed_group_size}")

    # Use mock tokenizer (no network needed)
    print("\nCreating mock tokenizer...")
    tokenizer = MockTokenizer()

    # Create mock actor
    mock_actor = MockActorRolloutWG(tokenizer)

    # Create configs
    config = create_test_config(args.n_rollouts, args.batch_size, args.episode_length)
    eval_env_config = create_eval_config(args.n_rollouts, args.batch_size, args.episode_length)
    eval_env_config['seed_group_size'] = args.seed_group_size

    # Create evaluator
    print("\nCreating MultiEnvEvaluator...")
    from verl.trainer.ppo.multi_env_evaluator import MultiEnvEvaluator

    eval_config = OmegaConf.create({
        'environments': [eval_env_config],
    })

    evaluator = MultiEnvEvaluator(
        config=config,
        tokenizer=tokenizer,
        actor_rollout_wg=mock_actor,
        val_reward_fn=None,
        eval_config=eval_config,
    )

    # Run evaluation with timing
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    start_time = time.perf_counter()
    metrics = evaluator.evaluate(global_step=0)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nTotal evaluation time: {total_time:.2f}s")
    print(f"Mock generate_sequences calls: {mock_actor.call_count}")

    # Print key timing metrics
    print("\nTiming metrics from evaluator:")
    timing_keys = [k for k in metrics.keys() if 'time' in k.lower() or 'debug' in k.lower()]
    for key in sorted(timing_keys):
        print(f"  {key}: {metrics[key]:.2f}s" if isinstance(metrics[key], float) else f"  {key}: {metrics[key]}")

    # Calculate "other" time
    inference_time = metrics.get('eval_Test-StateVisitation/inference_time_seconds', 0)
    env_step_time = metrics.get('eval_Test-StateVisitation/env_step_time_seconds', 0)
    other_time = total_time - inference_time - env_step_time
    print(f"\nCalculated 'other' time: {other_time:.2f}s")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
