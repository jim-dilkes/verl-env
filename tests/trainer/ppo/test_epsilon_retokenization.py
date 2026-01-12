"""Tests for epsilon-greedy re-tokenization for on-policy training.

Tests the helper functions used to rewrite response text and re-tokenize
when epsilon exploration triggers a different action than the model selected.

Run with: pytest tests/trainer/ppo/test_epsilon_retokenization.py -v
"""

import pytest
import torch
from transformers import AutoTokenizer

from verl.trainer.ppo.ray_multistep_trainer import (
    rewrite_decision_tag,
    retokenize_epsilon_sample,
)


class TestRewriteDecisionTag:
    """Unit tests for rewrite_decision_tag() function."""

    def test_valid_tag_replacement(self):
        """Standard case: replace decision tag content."""
        response = "<think>I should go up</think><decision>up</decision>"
        new_response, success = rewrite_decision_tag(response, "left")

        assert success is True
        assert new_response == "<think>I should go up</think><decision>left</decision>"

    def test_no_decision_tag_single_action_mode(self):
        """Single-action mode with <action> tag - should fail (not supported)."""
        response = "<think>I should go up</think><action>up</action>"
        new_response, success = rewrite_decision_tag(response, "left")

        assert success is False
        assert new_response == response  # Unchanged

    def test_multiple_decision_tags_replaces_last(self):
        """Multiple decision tags - only last (final selection) should be replaced."""
        response = "<decision>up</decision> then <decision>down</decision>"
        new_response, success = rewrite_decision_tag(response, "left")

        assert success is True
        assert new_response == "<decision>up</decision> then <decision>left</decision>"

    def test_empty_tag_content(self):
        """Empty decision tag should still be replaced."""
        response = "<think>hmm</think><decision></decision>"
        new_response, success = rewrite_decision_tag(response, "right")

        assert success is True
        assert new_response == "<think>hmm</think><decision>right</decision>"

    def test_tag_with_whitespace(self):
        """Whitespace inside tag should be replaced."""
        response = "<decision> up </decision>"
        new_response, success = rewrite_decision_tag(response, "down")

        assert success is True
        assert new_response == "<decision>down</decision>"

    def test_malformed_unclosed_tag(self):
        """Unclosed tag should fail."""
        response = "<think>going</think><decision>up"
        new_response, success = rewrite_decision_tag(response, "left")

        assert success is False
        assert new_response == response  # Unchanged

    def test_no_tags_at_all(self):
        """No XML tags at all should fail."""
        response = "I will go up"
        new_response, success = rewrite_decision_tag(response, "left")

        assert success is False
        assert new_response == response

    def test_complex_multi_action_response(self):
        """Full multi-action reasoning format."""
        response = """<actions>
<action_up><reasoning>Going up is safe</reasoning></action_up>
<action_down><reasoning>Going down hits wall</reasoning></action_down>
</actions>
<decision>up</decision>"""
        new_response, success = rewrite_decision_tag(response, "left")

        assert success is True
        assert "<decision>left</decision>" in new_response
        # Reasoning should be preserved
        assert "<action_up>" in new_response
        assert "Going up is safe" in new_response


class TestRetokenizeEpsilonSample:
    """Unit tests for retokenize_epsilon_sample() function."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load Qwen3 tokenizer once for all tests."""
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)

    def _get_prompt_tensors(self, tokenizer, prompt_text):
        """Helper to get prompt ids, attention mask, and position ids."""
        tokenized = tokenizer(prompt_text, return_tensors='pt')
        prompt_ids = tokenized['input_ids'].squeeze(0)
        prompt_attention_mask = tokenized['attention_mask'].squeeze(0)
        prompt_position_ids = torch.arange(len(prompt_ids))
        return prompt_ids, prompt_attention_mask, prompt_position_ids

    def _get_padded_prompt_tensors(self, tokenizer, prompt_text: str, max_prompt_length: int):
        """Match trainer behavior: left-pad prompt to max length and compute position_ids from mask."""
        old_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            tokenized = tokenizer(
                prompt_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_prompt_length,
            )
        finally:
            tokenizer.padding_side = old_padding_side

        prompt_ids = tokenized['input_ids'].squeeze(0)
        prompt_attention_mask = tokenized['attention_mask'].squeeze(0)
        prompt_position_ids = prompt_attention_mask.long().cumsum(-1) - 1
        prompt_position_ids.masked_fill_(prompt_attention_mask == 0, 1)
        return prompt_ids, prompt_attention_mask, prompt_position_ids

    def test_basic_retokenization_shapes(self, tokenizer):
        """Verify output tensor shapes are correct."""
        prompt_text = "You are a snake. Move carefully."
        response_text = "<think>I see an apple</think><decision>up</decision>"

        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, prompt_text)
        max_response_length = 64

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
            debug=True,
        )

        prompt_len = len(prompt_ids)

        assert result['responses'].shape == (max_response_length,)
        assert result['input_ids'].shape == (prompt_len + max_response_length,)
        assert result['attention_mask'].shape == (prompt_len + max_response_length,)
        assert result['position_ids'].shape == (prompt_len + max_response_length,)

    def test_prompt_prefix_preserved(self, tokenizer):
        """Re-tokenization must not modify the prompt portion of any full-seq tensors."""
        prompt_text = "You are a snake. Move carefully."
        response_text = "<think>I see an apple</think><decision>up</decision>"
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, prompt_text)
        max_response_length = 64

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        prompt_len = len(prompt_ids)
        assert torch.equal(result['input_ids'][:prompt_len], prompt_ids)
        assert torch.equal(result['attention_mask'][:prompt_len], prompt_attention_mask)
        assert torch.equal(result['position_ids'][:prompt_len], prompt_position_ids)

    def test_padding_short_response(self, tokenizer):
        """Response shorter than max_response_length should be padded."""
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Hello")
        response_text = "Hi"  # Very short
        max_response_length = 32

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        # Response should be padded to max_response_length
        assert result['responses'].shape[0] == max_response_length

        # Check padding tokens exist (attention_mask should have 0s at end)
        response_attention = result['attention_mask'][len(prompt_ids):]
        assert (response_attention == 0).any(), "Expected some padding in response"

    def test_response_attention_mask_matches_tokenized_length(self, tokenizer):
        """Response attention mask should be exact: 1s for real tokens then 0s for padding."""
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Hello")
        response_text = "<decision>up</decision>"
        max_response_length = 64

        tokenized = tokenizer(response_text, add_special_tokens=False)
        expected_len = min(len(tokenized['input_ids']), max_response_length)

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        response_attention = result['attention_mask'][len(prompt_ids):]
        assert torch.all(response_attention[:expected_len] == 1)
        assert torch.all(response_attention[expected_len:] == 0)

    def test_response_tokens_match_tokenizer_ids(self, tokenizer):
        """Response token IDs should exactly match tokenizer(new_response_text) up to truncation.

        This ensures the helper isn't silently shifting tokens or changing special-token behavior.
        """
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Hello")
        response_text = "<think>hmm</think><decision>up</decision>"
        max_response_length = 32

        expected_tokenized = tokenizer(response_text, add_special_tokens=False)
        expected_ids = expected_tokenized['input_ids']
        expected_len = min(len(expected_ids), max_response_length)

        # Match retokenize_epsilon_sample() pad-id fallback behavior
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        got = result['responses'].tolist()

        # Prefix matches tokenization output (possibly truncated)
        assert got[:expected_len] == expected_ids[:expected_len]

        # Remainder padded (if any)
        if expected_len < max_response_length:
            assert all(t == pad_token_id for t in got[expected_len:])

    def test_truncation_long_response(self, tokenizer):
        """Response longer than max_response_length should be truncated."""
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Prompt")
        # Create a very long response
        response_text = "word " * 100  # Should tokenize to many tokens
        max_response_length = 16  # Force truncation

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        # Response should be exactly max_response_length
        assert result['responses'].shape[0] == max_response_length

    def test_attention_mask_correctness(self, tokenizer):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        prompt_text = "Short prompt"
        response_text = "<decision>up</decision>"

        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, prompt_text)
        max_response_length = 64

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        attention_mask = result['attention_mask']

        # All values should be 0 or 1
        assert ((attention_mask == 0) | (attention_mask == 1)).all()

        # Should have some 1s (real tokens)
        assert (attention_mask == 1).any()

    def test_position_ids_monotonic(self, tokenizer):
        """Position IDs should be monotonically increasing for attended positions."""
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Test prompt here")
        response_text = "<think>thinking</think><decision>down</decision>"
        max_response_length = 48

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        position_ids = result['position_ids']
        attention_mask = result['attention_mask']

        # For attended positions, position_ids should be non-decreasing
        attended_positions = position_ids[attention_mask == 1]
        for i in range(1, len(attended_positions)):
            assert attended_positions[i] >= attended_positions[i-1], \
                f"Position IDs not monotonic: {attended_positions[i-1]} -> {attended_positions[i]}"

    def test_tensor_device_placement(self, tokenizer):
        """Output tensors should be on specified device."""
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Test")
        device = torch.device('cpu')

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text="<decision>up</decision>",
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=32,
            device=device,
        )

        for key, tensor in result.items():
            assert tensor.device == device, f"{key} on wrong device: {tensor.device}"

    def test_position_ids_continue_from_prompt(self, tokenizer):
        """Response position IDs should continue from last prompt position."""
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, "Test prompt")
        max_response_length = 32

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text="<decision>up</decision>",
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        # First response position should be last_prompt_pos + 1
        last_prompt_pos = prompt_position_ids[-1].item()
        first_response_pos = result['position_ids'][len(prompt_ids)].item()
        assert first_response_pos == last_prompt_pos + 1

    def test_left_padded_prompt_matches_trainer_position_ids(self, tokenizer):
        """When prompt is left-padded like training, response positions should still continue correctly."""
        max_prompt_length = 32
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_padded_prompt_tensors(
            tokenizer, "Short prompt", max_prompt_length=max_prompt_length
        )
        max_response_length = 16

        # Sanity: last prompt token should be attended and its position should be (#attended - 1)
        assert prompt_attention_mask[-1].item() == 1
        assert prompt_position_ids[-1].item() == int(prompt_attention_mask.sum().item()) - 1

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text="<decision>up</decision>",
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        first_response_pos = result['position_ids'][len(prompt_ids)].item()
        assert first_response_pos == prompt_position_ids[-1].item() + 1


class TestEndToEndTextModification:
    """Integration tests for the full text modification flow."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)

    def _get_prompt_tensors(self, tokenizer, prompt_text):
        """Helper to get prompt ids, attention mask, and position ids."""
        tokenized = tokenizer(prompt_text, return_tensors='pt')
        prompt_ids = tokenized['input_ids'].squeeze(0)
        prompt_attention_mask = tokenized['attention_mask'].squeeze(0)
        prompt_position_ids = torch.arange(len(prompt_ids))
        return prompt_ids, prompt_attention_mask, prompt_position_ids

    def test_epsilon_modification_preserves_reasoning(self, tokenizer):
        """Full flow: epsilon triggers, text rewritten, reasoning preserved."""
        # Simulate multi-action response
        original_response = """<actions>
<action_up><reasoning>Up leads to apple</reasoning></action_up>
<action_down><reasoning>Down is blocked</reasoning></action_down>
<action_left><reasoning>Left is safe</reasoning></action_left>
<action_right><reasoning>Right hits wall</reasoning></action_right>
</actions>
<decision>up</decision>"""

        # Epsilon chose "left" instead
        executed_action = "left"

        # Step 1: Rewrite decision tag
        new_response, success = rewrite_decision_tag(original_response, executed_action)
        assert success is True
        assert "<decision>left</decision>" in new_response
        assert "Up leads to apple" in new_response  # Original reasoning preserved

        # Step 2: Re-tokenize
        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(
            tokenizer, "You are playing snake."
        )
        max_response_length = 128

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=new_response,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        # Step 3: Verify we can decode back to the modified text
        # (accounting for tokenizer quirks)
        decoded = tokenizer.decode(result['responses'], skip_special_tokens=True)
        assert "left" in decoded.lower()

    def test_consistent_tensor_dimensions(self, tokenizer):
        """Verify all output tensors have consistent dimensions."""
        prompt_text = "Navigate the snake game."
        response_text = "<think>Analyzing...</think><decision>right</decision>"

        prompt_ids, prompt_attention_mask, prompt_position_ids = self._get_prompt_tensors(tokenizer, prompt_text)
        max_response_length = 64

        result = retokenize_epsilon_sample(
            tokenizer=tokenizer,
            new_response_text=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            max_response_length=max_response_length,
            device=torch.device('cpu'),
        )

        expected_full_len = len(prompt_ids) + max_response_length

        # All full-sequence tensors should have same length
        assert result['input_ids'].shape[0] == expected_full_len
        assert result['attention_mask'].shape[0] == expected_full_len
        assert result['position_ids'].shape[0] == expected_full_len

        # Response tensor should match max_response_length
        assert result['responses'].shape[0] == max_response_length

        # input_ids should equal prompt + response concatenation
        reconstructed = torch.cat([prompt_ids, result['responses']])
        assert torch.equal(result['input_ids'], reconstructed)
