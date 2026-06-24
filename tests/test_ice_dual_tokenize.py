"""Test Group 2+6 [P0/P1]: Dual tokenization consistency + decode verification.

Requires a tokenizer with chat template support.
Run with: python tests/test_ice_dual_tokenize.py
Set TEST_MODEL_ID env var to override model (default: Qwen/Qwen3-0.6B-Base).
"""

import os
import torch

from verl.envs.environments.focus_instructions import inject_focus_into_obs

TEMPLATE = 'Pay particular attention to this aspect of the task: "{STEP_TEXT}". Consider how it could apply in the current situation before choosing your action.'


def get_tokenizer():
    from transformers import AutoTokenizer

    model_id = os.environ.get("TEST_MODEL_ID", "Qwen/Qwen3-0.6B-Base")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def test_dual_tokenize_same_shape():
    """Both tokenizations produce same tensor shape."""
    tokenizer = get_tokenizer()
    obs_vec = [
        [{"role": "user", "content": "Observation: kitchen state A"}],
        [{"role": "user", "content": "Observation: kitchen state A"}],
        [{"role": "user", "content": "Observation: kitchen state B"}],
        [{"role": "user", "content": "Observation: kitchen state B"}],
    ]
    focus_per_rollout = ["Pick up ingredients", None, "Wait for soup", None]
    max_len = 512

    rollout_obs = inject_focus_into_obs(obs_vec, focus_per_rollout, TEMPLATE)
    rollout_text = tokenizer.apply_chat_template(rollout_obs, tokenize=False, add_generation_prompt=True)
    base_text = tokenizer.apply_chat_template(obs_vec, tokenize=False, add_generation_prompt=True)

    rollout_tokens = tokenizer(rollout_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    base_tokens = tokenizer(base_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)

    assert rollout_tokens["input_ids"].shape == base_tokens["input_ids"].shape


def test_no_focus_identical_tokens():
    """No-focus rollouts have identical tokens in both passes."""
    tokenizer = get_tokenizer()
    obs_vec = [
        [{"role": "user", "content": "Observation: kitchen state A"}],
        [{"role": "user", "content": "Observation: kitchen state A"}],
    ]
    focus_per_rollout = ["Pick up ingredients", None]
    max_len = 512

    rollout_obs = inject_focus_into_obs(obs_vec, focus_per_rollout, TEMPLATE)
    rollout_text = tokenizer.apply_chat_template(rollout_obs, tokenize=False, add_generation_prompt=True)
    base_text = tokenizer.apply_chat_template(obs_vec, tokenize=False, add_generation_prompt=True)

    rollout_tokens = tokenizer(rollout_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    base_tokens = tokenizer(base_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)

    # Index 1 has None focus — should be identical
    assert torch.equal(rollout_tokens["input_ids"][1], base_tokens["input_ids"][1])


def test_focus_rollouts_longer():
    """Focus rollouts have more real tokens (less left-padding)."""
    tokenizer = get_tokenizer()
    obs_vec = [
        [{"role": "user", "content": "Observation: kitchen state A"}],
        [{"role": "user", "content": "Observation: kitchen state A"}],
    ]
    focus_per_rollout = ["Pick up ingredients", None]
    max_len = 512

    rollout_obs = inject_focus_into_obs(obs_vec, focus_per_rollout, TEMPLATE)
    rollout_text = tokenizer.apply_chat_template(rollout_obs, tokenize=False, add_generation_prompt=True)
    base_text = tokenizer.apply_chat_template(obs_vec, tokenize=False, add_generation_prompt=True)

    rollout_tokens = tokenizer(rollout_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    base_tokens = tokenizer(base_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)

    # Index 0 has focus — rollout should be longer
    rollout_real = rollout_tokens["attention_mask"][0].sum()
    base_real = base_tokens["attention_mask"][0].sum()
    assert rollout_real > base_real, f"Focus prompt should be longer: rollout={rollout_real}, base={base_real}"


def test_decode_focus_present_then_absent():
    """Decode verification: focus text present in rollout tokens, absent in base tokens."""
    tokenizer = get_tokenizer()
    obs_vec = [[{"role": "user", "content": "Observation: kitchen state A"}]]
    focus_per_rollout = ["Pick up ingredients"]
    max_len = 512

    rollout_obs = inject_focus_into_obs(obs_vec, focus_per_rollout, TEMPLATE)
    rollout_text = tokenizer.apply_chat_template(rollout_obs, tokenize=False, add_generation_prompt=True)
    base_text = tokenizer.apply_chat_template(obs_vec, tokenize=False, add_generation_prompt=True)

    rollout_tokens = tokenizer(rollout_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    base_tokens = tokenizer(base_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)

    rollout_decoded = tokenizer.decode(rollout_tokens["input_ids"][0], skip_special_tokens=False)
    base_decoded = tokenizer.decode(base_tokens["input_ids"][0], skip_special_tokens=False)

    assert "carefully consider" in rollout_decoded, "Focus template text missing from rollout"
    assert "carefully consider" not in base_decoded, "Focus template text leaked into base"


if __name__ == "__main__":
    test_dual_tokenize_same_shape()
    print("  [PASS] dual tokenize same shape")
    test_no_focus_identical_tokens()
    print("  [PASS] no-focus identical tokens")
    test_focus_rollouts_longer()
    print("  [PASS] focus rollouts longer")
    test_decode_focus_present_then_absent()
    print("  [PASS] decode focus present/absent")
    print("All Group 2+6 tests passed!")
