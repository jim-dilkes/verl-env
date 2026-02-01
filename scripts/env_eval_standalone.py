#!/usr/bin/env python3
"""
Truly standalone environment evaluation - BYPASSES PPO trainer entirely.

This script:
1. Loads vLLM directly (no FSDP overhead)
2. Creates environment
3. Runs rollouts
4. Reports metrics

No actor, no critic, no training - pure inference evaluation.

Usage:
  python scripts/env_eval_standalone.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --env overcooked \
    --layout cramped_room \
    --n_rollouts 50 \
    --gpu_memory 0.85

Or via HuggingFace cache:
  python scripts/env_eval_standalone.py \
    --model /path/to/cached/model \
    --env overcooked
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Any, Optional
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Environment Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model ID")
    parser.add_argument("--env", type=str, default="overcooked", help="Environment name")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--n_rollouts", type=int, default=10, help="Number of rollouts")
    parser.add_argument("--horizon", type=int, default=40, help="Max steps per episode")
    parser.add_argument("--gpu_memory", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max response tokens")
    parser.add_argument("--diverse_prompts", type=int, default=0, help="Number of diverse prompts (0=disabled)")
    return parser.parse_args()


def create_overcooked_env(layout_name: str, horizon: int, n_envs: int):
    """Create Overcooked environment."""
    try:
        from verl.envs.overcooked.overcooked_env import OvercookedEnv
        
        env = OvercookedEnv(
            layout_name=layout_name,
            horizon=horizon,
            partner_policy="none",
            shaped_reward=True,
            pot_cook_time=5,
            print_coordinates=True,
            print_visualization=False,
        )
        return env
    except ImportError:
        print("Error: Could not import OvercookedEnv. Make sure verl is installed.")
        sys.exit(1)


def get_instruction_prompt():
    """Get the instruction prompt for Overcooked."""
    return """[Instructions]
You are a chef cooking soup in a kitchen.
Your goal is to cook and deliver the soups as fast as possible to earn rewards.

[How to Cook]
1. Pick up ingredients (e.g., onions) from ingredient piles using 'interact' while facing them
2. Place 3 ingredients in a pot using 'interact'
3. Wait for the soup to cook
4. Pick up a dish from the dish pile using 'interact'
5. Pick up the cooked soup from the pot using 'interact' (with dish in hand)
6. Deliver the soup to the serving counter using 'interact'

[Available Actions]
"right": move right,
"down": move down,
"left": move left,
"up": move up,
"stay": stay in place (wait),
"interact": interact with object in front of you

[Response Format]
Respond using ONLY valid XML with <plan>...</plan> and <action>...</action> tags.

<plan>{Think about what to do}</plan>
<action>{Your selected action}</action>

[Rules]
- You can only hold one object at a time
- Each soup requires exactly 3 ingredients
"""


def parse_action(response: str) -> tuple:
    """Parse action from response."""
    import re
    
    valid_actions = {"right", "down", "left", "up", "stay", "interact"}
    
    match = re.search(r'<action>\s*(\w+)\s*</action>', response, re.IGNORECASE)
    if match:
        action = match.group(1).lower()
        if action in valid_actions:
            return action, True
    
    return "stay", False  # Default action if parsing fails


def run_episode(
    llm: LLM,
    tokenizer,
    env,
    sampling_params: SamplingParams,
    instruction: str,
    max_steps: int,
) -> Dict[str, Any]:
    """Run a single episode."""
    obs = env.reset()
    total_reward = 0
    valid_actions = 0
    steps = 0
    
    for step in range(max_steps):
        # Build prompt
        prompt = f"{instruction}\n\n[Current State]\n{obs}\n"
        
        # Generate response
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = llm.generate([formatted_prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        # Parse action
        action, is_valid = parse_action(response)
        if is_valid:
            valid_actions += 1
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    return {
        "total_reward": total_reward,
        "steps": steps,
        "valid_action_ratio": valid_actions / max(steps, 1),
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("STANDALONE ENVIRONMENT EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Layout: {args.layout}")
    print(f"N Rollouts: {args.n_rollouts}")
    print(f"GPU Memory: {args.gpu_memory}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load vLLM - SINGLE model load, no FSDP overhead!
    print("\nLoading vLLM engine (direct HF load, no FSDP)...")
    start_time = time.time()
    
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory,
        trust_remote_code=True,
        max_model_len=1024,
        enforce_eager=True,  # For stability
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0,
    )
    
    # Create environment
    print(f"\nCreating {args.env} environment...")
    env = create_overcooked_env(args.layout, args.horizon, 1)
    
    # Get instruction
    instruction = get_instruction_prompt()
    
    # Run rollouts
    print(f"\nRunning {args.n_rollouts} rollouts...")
    results = []
    
    for i in range(args.n_rollouts):
        result = run_episode(llm, tokenizer, env, sampling_params, instruction, args.horizon)
        results.append(result)
        
        if (i + 1) % 10 == 0 or i == args.n_rollouts - 1:
            avg_reward = np.mean([r["total_reward"] for r in results])
            avg_valid = np.mean([r["valid_action_ratio"] for r in results])
            print(f"  [{i+1}/{args.n_rollouts}] Avg reward: {avg_reward:.2f}, Valid action ratio: {avg_valid:.2%}")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    rewards = [r["total_reward"] for r in results]
    valid_ratios = [r["valid_action_ratio"] for r in results]
    steps = [r["steps"] for r in results]
    
    print(f"Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Valid Action Ratio: {np.mean(valid_ratios):.2%} ± {np.std(valid_ratios):.2%}")
    print(f"Avg Steps: {np.mean(steps):.1f}")
    print(f"Episodes with positive reward: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
