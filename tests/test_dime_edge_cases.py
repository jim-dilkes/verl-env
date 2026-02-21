"""Test Group 5 [P1]: DIME edge cases."""

import torch
from verl import DataProto
from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base


def make_batch(plen, rlen, n_rollouts, n_steps, pad_token_id=0):
    """Minimal synthetic batch for edge case testing."""
    total = n_rollouts * (n_steps + 1)
    base_prompt_tokens_by_step = []

    all_input_ids = torch.full((total, plen + rlen), pad_token_id, dtype=torch.int64)
    all_attention_mask = torch.zeros(total, plen + rlen, dtype=torch.int64)
    all_responses = torch.full((total, rlen), pad_token_id, dtype=torch.int64)

    for step_idx in range(n_steps + 1):
        base_ids = torch.full((n_rollouts, plen), pad_token_id, dtype=torch.int64)
        base_mask = torch.zeros(n_rollouts, plen, dtype=torch.int64)

        for env_idx in range(n_rollouts):
            sample_idx = step_idx * n_rollouts + env_idx
            # Minimal prompt: 5 tokens
            base_ids[env_idx, plen - 5 :] = torch.arange(100, 105)
            base_mask[env_idx, plen - 5 :] = 1
            # Rollout prompt same as base (simulates no-focus)
            all_input_ids[sample_idx, plen - 5 : plen] = torch.arange(100, 105)
            all_attention_mask[sample_idx, plen - 5 : plen] = 1

        base_prompt_tokens_by_step.append({"input_ids": base_ids, "attention_mask": base_mask})

    all_position_ids = all_attention_mask.long().cumsum(-1) - 1
    all_position_ids.masked_fill_(all_attention_mask == 0, 1)

    batch = DataProto.from_dict(tensors={
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "responses": all_responses,
    })
    return batch, base_prompt_tokens_by_step


def test_all_non_instructed_is_identity():
    """All rollouts with no instruction — swap should be identity."""
    plen, rlen, n_rollouts, n_steps = 32, 16, 4, 2
    batch, base_tokens = make_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [False] * n_rollouts

    ids_before = batch.batch["input_ids"].clone()
    mask_before = batch.batch["attention_mask"].clone()
    pos_before = batch.batch["position_ids"].clone()

    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    assert torch.equal(batch.batch["input_ids"], ids_before)
    assert torch.equal(batch.batch["attention_mask"], mask_before)
    assert torch.equal(batch.batch["position_ids"], pos_before)


def test_single_rollout():
    """n_rollouts=1 should work without indexing errors."""
    plen, rlen, n_rollouts, n_steps = 32, 16, 1, 3
    batch, base_tokens = make_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True]
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)
    assert batch.batch["input_ids"].shape == (n_steps + 1, plen + rlen)


def test_zero_episode_length():
    """episode_len=0 (just initial step) should work."""
    plen, rlen, n_rollouts, n_steps = 32, 16, 4, 0
    batch, base_tokens = make_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True] * n_rollouts
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)
    assert batch.batch["input_ids"].shape == (n_rollouts, plen + rlen)


def test_empty_response():
    """All-pad response should be preserved correctly."""
    plen, rlen, n_rollouts, n_steps = 32, 16, 2, 1
    batch, base_tokens = make_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True] * n_rollouts

    # Ensure responses are all pad (already the case from make_batch)
    assert (batch.batch["responses"] == 0).all()

    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    # Responses should still be all pad
    assert (batch.batch["responses"] == 0).all()
    # Response portion of input_ids should also be all pad
    total = n_rollouts * (n_steps + 1)
    for i in range(total):
        assert (batch.batch["input_ids"][i, -rlen:] == 0).all()


if __name__ == "__main__":
    test_all_non_instructed_is_identity()
    print("  [PASS] all non-instructed is identity")
    test_single_rollout()
    print("  [PASS] single rollout")
    test_zero_episode_length()
    print("  [PASS] zero episode length")
    test_empty_response()
    print("  [PASS] empty response")
    print("All Group 5 tests passed!")
