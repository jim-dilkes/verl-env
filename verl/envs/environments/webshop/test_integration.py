"""
Test script to verify WebShop environment integration.

This script tests that:
1. The environment can be created
2. It can be reset
3. It can process actions
4. The observation format is correct
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf


def test_webshop_integration():
    """Test basic WebShop integration."""
    print("=" * 60)
    print("Testing WebShop Environment Integration")
    print("=" * 60)
    
    # Create a minimal config
    config_dict = {
        'envs': {
            'env_name': 'webshop',
            'n_rollouts': 4,
            'webshop_kwargs': {
                'observation_mode': 'text',
                'num_products': 100,
                'human_goals': 0,
            },
            'format_penalty': 0.1,
            'binary_reward': False,
        }
    }
    config = OmegaConf.create(config_dict)
    
    # Import the make_env function
    from verl.envs.environments import make_env
    
    print("\n[1] Creating WebShop environment...")
    try:
        env = make_env('webshop', 'default', config)
        print("✓ Environment created successfully!")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[2] Testing environment reset...")
    try:
        obs, info = env.reset()
        print("✓ Environment reset successfully!")
        print(f"   Observation keys: {obs.keys()}")
        print(f"   Text keys: {obs.get('text', {}).keys()}")
        print(f"   Info: {info}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[3] Checking observation format...")
    try:
        assert 'text' in obs, "Observation missing 'text' key"
        assert 'long_term_context' in obs['text'], "Observation missing 'long_term_context'"
        assert 'short_term_context' in obs['text'], "Observation missing 'short_term_context'"
        print("✓ Observation format is correct!")
        print(f"   Long-term context length: {len(obs['text']['long_term_context'])} chars")
        print(f"   Observation:")
        print("   " + "-" * 56)
        print("   " + obs['text']['long_term_context'].replace('\n', '\n   '))
        print("   " + "-" * 56)
    except AssertionError as e:
        print(f"✗ Observation format incorrect: {e}")
        return False
    
    print("\n[4] Testing action extraction...")
    try:
        # Test valid search action
        test_action = "<think>I need to search for something</think><action>search[red shoes]</action>"
        full_action, executed_action, is_valid, metrics = env.extract_action(test_action)
        print(f"   Test action: {test_action[:80]}...")
        print(f"   Extracted: {executed_action}")
        print(f"   Is valid: {is_valid}")
        print(f"   Metrics: {metrics}")
        print("✓ Action extraction works!")
    except Exception as e:
        print(f"✗ Failed to extract action: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[5] Testing environment step...")
    try:
        # Take a search action
        obs, reward, terminated, truncated, info = env.step('search[laptop]', is_valid=True)
        print("✓ Environment step executed successfully!")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}, Truncated: {truncated}")
        print(f"   Observation:")
        print("   " + obs['text']['long_term_context'].replace('\n', '\n   '))
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[6] Testing instruction prompt...")
    try:
        inst_prompt = env.get_instruction_prompt()
        print("✓ Instruction prompt retrieved!")
        print(f"   Prompt length: {len(inst_prompt)} chars")
        print(f"   Prompt:")
        print("   " + "-" * 56)
        print("   " + inst_prompt.replace('\n', '\n   '))
        print("   " + "-" * 56)
    except Exception as e:
        print(f"✗ Failed to get instruction prompt: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[7] Closing environment...")
    try:
        env.close()
        print("✓ Environment closed successfully!")
    except Exception as e:
        print(f"✗ Failed to close environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_webshop_integration()
    sys.exit(0 if success else 1)

