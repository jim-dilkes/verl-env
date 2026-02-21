"""Test Group 3+4 [P0]: swap_all_instructed_to_base correctness + gold standard verification.

Tests use synthetic tensors mimicking real batch structure.
"""

import torch
import numpy as np


def make_synthetic_batch(plen, rlen, n_rollouts, n_steps, pad_token_id=0):
    """Build a synthetic batch mimicking DIME rollout structure.

    Returns:
        batch: DataProto-like object
        base_prompt_tokens_by_step: list of dicts with base prompt tokens
        rollout_prompts: the rollout prompts (with focus) that were used for generation
    """
    from verl import DataProto

    total = n_rollouts * (n_steps + 1)
    base_prompt_tokens_by_step = []
    all_input_ids = torch.full((total, plen + rlen), pad_token_id, dtype=torch.int64)
    all_attention_mask = torch.zeros(total, plen + rlen, dtype=torch.int64)
    all_responses = torch.full((total, rlen), pad_token_id, dtype=torch.int64)

    rng = torch.Generator()
    rng.manual_seed(42)

    for step_idx in range(n_steps + 1):
        base_ids = torch.full((n_rollouts, plen), pad_token_id, dtype=torch.int64)
        base_mask = torch.zeros(n_rollouts, plen, dtype=torch.int64)

        for env_idx in range(n_rollouts):
            sample_idx = step_idx * n_rollouts + env_idx

            # Base prompt: shorter (random length 10-30)
            base_len = torch.randint(10, 30, (1,), generator=rng).item()
            base_tokens = torch.randint(100, 10000, (base_len,), generator=rng)
            # Left-pad base prompt
            base_ids[env_idx, plen - base_len :] = base_tokens
            base_mask[env_idx, plen - base_len :] = 1

            # Rollout prompt: longer (base + 5-15 focus tokens)
            focus_extra = torch.randint(5, 15, (1,), generator=rng).item()
            rollout_len = min(base_len + focus_extra, plen)
            rollout_tokens = torch.randint(100, 10000, (rollout_len,), generator=rng)
            # Left-pad rollout prompt
            all_input_ids[sample_idx, plen - rollout_len : plen] = rollout_tokens
            all_attention_mask[sample_idx, plen - rollout_len : plen] = 1

            # Response: random length 5-20
            resp_len = torch.randint(5, 20, (1,), generator=rng).item()
            resp_tokens = torch.randint(100, 10000, (resp_len,), generator=rng)
            all_responses[sample_idx, :resp_len] = resp_tokens
            all_input_ids[sample_idx, plen : plen + resp_len] = resp_tokens
            all_attention_mask[sample_idx, plen : plen + resp_len] = 1

        base_prompt_tokens_by_step.append({
            "input_ids": base_ids.clone(),
            "attention_mask": base_mask.clone(),
        })

    # Build position_ids from attention_mask
    all_position_ids = all_attention_mask.long().cumsum(-1) - 1
    all_position_ids.masked_fill_(all_attention_mask == 0, 1)

    batch_dict = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "responses": all_responses,
    }
    batch = DataProto.from_dict(tensors=batch_dict)

    return batch, base_prompt_tokens_by_step


def test_response_tokens_unchanged():
    """P0: Response tokens must be identical after swap."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True] * n_rollouts

    responses_before = batch.batch["responses"].clone()
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    assert torch.equal(batch.batch["responses"], responses_before), "Responses changed after swap!"


def test_input_ids_response_portion():
    """P0: input_ids[-rlen:] must equal responses after swap."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True] * n_rollouts
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    total = n_rollouts * (n_steps + 1)
    for i in range(total):
        assert torch.equal(
            batch.batch["input_ids"][i, -rlen:], batch.batch["responses"][i]
        ), f"Sample {i}: input_ids response portion != responses tensor"


def test_input_ids_prompt_portion():
    """P0: input_ids[:plen] must equal base prompt after swap."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True] * n_rollouts
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    for step in range(n_steps + 1):
        for env in range(n_rollouts):
            idx = step * n_rollouts + env
            assert torch.equal(
                batch.batch["input_ids"][idx, :plen],
                base_tokens[step]["input_ids"][env],
            ), f"step={step} env={env}: prompt portion mismatch"


def test_position_ids_contiguous():
    """P0: position_ids must be contiguous (no gaps) in the real-token region."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps)
    has_instruction = [True] * n_rollouts
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    total = n_rollouts * (n_steps + 1)
    for i in range(total):
        mask = batch.batch["attention_mask"][i]
        pos = batch.batch["position_ids"][i]
        real_positions = pos[mask == 1]
        expected = torch.arange(len(real_positions))
        assert torch.equal(real_positions, expected), f"Position gap at sample {i}"


def test_attention_mask_consistent():
    """P0: Where attention_mask == 1, input_ids should not be pad_token."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    pad_token_id = 0
    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps, pad_token_id)
    has_instruction = [True] * n_rollouts
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    total = n_rollouts * (n_steps + 1)
    for i in range(total):
        ids = batch.batch["input_ids"][i]
        mask = batch.batch["attention_mask"][i]
        real_region = mask == 1
        assert (ids[real_region] != pad_token_id).all(), f"Pad token in real region at sample {i}"


def test_gold_standard_independent_construction():
    """P0 CRITICAL: Independently constructing [base_prompt | response] must match swap output."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    pad_token_id = 0
    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps, pad_token_id)
    has_instruction = [True] * n_rollouts

    original_responses = batch.batch["responses"].clone()
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    for step in range(n_steps + 1):
        for env in range(n_rollouts):
            idx = step * n_rollouts + env

            swapped_ids = batch.batch["input_ids"][idx]
            swapped_mask = batch.batch["attention_mask"][idx]
            swapped_pos = batch.batch["position_ids"][idx]

            base_prompt_ids = base_tokens[step]["input_ids"][env]
            base_prompt_mask = base_tokens[step]["attention_mask"][env]
            response = original_responses[idx]

            independent_ids = torch.cat([base_prompt_ids, response])
            response_mask = (response != pad_token_id).long()
            independent_mask = torch.cat([base_prompt_mask, response_mask])
            independent_pos = independent_mask.long().cumsum(-1) - 1
            independent_pos.masked_fill_(independent_mask == 0, 1)

            assert torch.equal(swapped_ids, independent_ids), (
                f"step={step} env={env}: input_ids mismatch"
            )
            assert torch.equal(swapped_mask, independent_mask), (
                f"step={step} env={env}: attention_mask mismatch"
            )
            assert torch.equal(swapped_pos, independent_pos), (
                f"step={step} env={env}: position_ids mismatch"
            )


def test_non_instructed_unchanged():
    """has_instruction=False rollouts must be unchanged."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps)

    original_ids = batch.batch['input_ids'].clone()
    original_mask = batch.batch['attention_mask'].clone()
    original_pos = batch.batch['position_ids'].clone()

    # Only rollouts 0,2 instructed; rollouts 1,3 should be unchanged
    has_instruction = [True, False, True, False]
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    for step in range(n_steps + 1):
        for env in range(n_rollouts):
            idx = step * n_rollouts + env
            if has_instruction[env]:
                assert torch.equal(
                    batch.batch['input_ids'][idx, :plen],
                    base_tokens[step]['input_ids'][env],
                ), f"step={step} env={env}: instructed rollout should have base prompt"
            else:
                assert torch.equal(
                    batch.batch['input_ids'][idx],
                    original_ids[idx],
                ), f"step={step} env={env}: non-instructed rollout should be unchanged"


def test_all_non_instructed_noop():
    """has_instruction=all False must leave batch entirely unchanged."""
    from verl.trainer.ppo.ray_multistep_trainer import swap_all_instructed_to_base

    plen, rlen, n_rollouts, n_steps = 64, 32, 4, 3
    batch, base_tokens = make_synthetic_batch(plen, rlen, n_rollouts, n_steps)

    original_ids = batch.batch['input_ids'].clone()
    original_mask = batch.batch['attention_mask'].clone()
    original_pos = batch.batch['position_ids'].clone()

    has_instruction = [False] * n_rollouts
    batch = swap_all_instructed_to_base(batch, base_tokens, n_rollouts, n_steps, rlen, has_instruction)

    assert torch.equal(batch.batch['input_ids'], original_ids), "input_ids changed with no instructed"
    assert torch.equal(batch.batch['attention_mask'], original_mask), "attention_mask changed"
    assert torch.equal(batch.batch['position_ids'], original_pos), "position_ids changed"


if __name__ == "__main__":
    test_response_tokens_unchanged()
    print("  [PASS] response tokens unchanged")
    test_input_ids_response_portion()
    print("  [PASS] input_ids response portion")
    test_input_ids_prompt_portion()
    print("  [PASS] input_ids prompt portion")
    test_position_ids_contiguous()
    print("  [PASS] position_ids contiguous")
    test_attention_mask_consistent()
    print("  [PASS] attention_mask consistent")
    test_gold_standard_independent_construction()
    print("  [PASS] gold standard independent construction")
    test_non_instructed_unchanged()
    print("  [PASS] non-instructed rollouts unchanged")
    test_all_non_instructed_noop()
    print("  [PASS] all non-instructed is noop")
    print("All tests passed!")
