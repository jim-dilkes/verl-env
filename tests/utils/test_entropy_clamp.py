import torch
import pytest

from verl.utils.torch_functional import (
    entropy_from_logits,
    clamped_entropy_from_logits,
)


def _sample_logits(shape, seed=0, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=generator, device=device)


def test_clamp_zero_matches_full_entropy():
    logits = _sample_logits(shape=(3, 128))
    full = entropy_from_logits(logits)
    clamped = clamped_entropy_from_logits(logits, clamp_p=0.0)
    torch.testing.assert_close(clamped, full, atol=1e-7, rtol=1e-6)


def test_clamp_reduces_entropy_when_tokens_removed():
    logits = torch.tensor([[3.0, 1.0, -2.0, -4.0]])
    entropy_full = clamped_entropy_from_logits(logits, clamp_p=0.0)
    entropy_clamped = clamped_entropy_from_logits(logits, clamp_p=0.5)
    assert torch.all(entropy_clamped <= entropy_full)
    assert torch.any(entropy_clamped < entropy_full)


@pytest.mark.parametrize("shape", [(2, 128), (1, 512)])
@pytest.mark.parametrize("clamp_p", [0.0, 0.1, 0.4])
def test_clamped_entropy_is_finite(shape, clamp_p):
    logits = _sample_logits(shape=shape, seed=42)
    entropy = clamped_entropy_from_logits(logits, clamp_p=clamp_p)
    assert torch.isfinite(entropy).all()

