#!/usr/bin/env python3
"""
Create a minimal test dataset for verl GRPO testing.
This creates a small parquet file with simple math-like prompts.
"""

import os
import pandas as pd
import random

def create_simple_math_prompt():
    """Create a simple math problem prompt in GSM8K format."""
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        answer = a + b
        prompt = f"Solve: {a} + {b} = ?"
    elif operation == '-':
        answer = max(a, b) - min(a, b)
        prompt = f"Solve: {max(a, b)} - {min(a, b)} = ?"
    else:  # *
        answer = a * b
        prompt = f"Solve: {a} × {b} = ?"
    
    # Format response in GSM8K style (with #### answer format)
    response = f"Let me solve this step by step.\n{a} {operation} {b} = {answer}\n#### {answer}"
    
    return [
        {
            "role": "user",
            "content": prompt,
        }
    ], str(answer), response

def main():
    data_path = os.path.expanduser("~/data/test_minimal")
    os.makedirs(data_path, exist_ok=True)
    
    # Create a small dataset (100 train, 20 test)
    train_data = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": []
    }
    
    test_data = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": []
    }
    
    # Generate train data
    for _ in range(100):
        prompt, ground_truth, response = create_simple_math_prompt()
        train_data["prompt"].append(prompt)
        train_data["data_source"].append("openai/gsm8k")  # Use GSM8K data_source for reward function
        train_data["ability"].append("math")
        train_data["reward_model"].append({"style": "rule", "ground_truth": ground_truth})
        train_data["extra_info"].append({"response": response})
    
    # Generate test data
    for _ in range(20):
        prompt, ground_truth, response = create_simple_math_prompt()
        test_data["prompt"].append(prompt)
        test_data["data_source"].append("openai/gsm8k")  # Use GSM8K data_source for reward function
        test_data["ability"].append("math")
        test_data["reward_model"].append({"style": "rule", "ground_truth": ground_truth})
        test_data["extra_info"].append({"response": response})
    
    # Convert to DataFrame and save
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_file = os.path.join(data_path, "train.parquet")
    test_file = os.path.join(data_path, "test.parquet")
    
    train_df.to_parquet(train_file)
    test_df.to_parquet(test_file)
    
    print(f"Created minimal test dataset:")
    print(f"  Train: {train_file} ({len(train_df)} samples)")
    print(f"  Test: {test_file} ({len(test_df)} samples)")
    print(f"\nTo use this dataset, update the script with:")
    print(f"  data.train_files={train_file}")
    print(f"  data.val_files={test_file}")

if __name__ == "__main__":
    main()

