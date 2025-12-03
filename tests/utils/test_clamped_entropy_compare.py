"""Quick script to verify the fast clamped entropy matches the slow reference."""

import time

import torch

from verl.utils.torch_functional import (
    clamped_entropy_from_logits,
    clamped_entropy_from_logits_SLOW,
)


def compare_once(shape, clamp_p, device):
    logits = torch.randn(*shape, device=device, dtype=torch.float32)

    start = time.perf_counter()
    fast = clamped_entropy_from_logits(logits, clamp_p)
    fast_time = time.perf_counter() - start

    start = time.perf_counter()
    slow = clamped_entropy_from_logits_SLOW(logits, clamp_p)
    slow_time = time.perf_counter() - start

    torch.testing.assert_close(fast.cpu(), slow.cpu(), atol=1e-6, rtol=1e-5)

    print(
        f"shape={shape}, clamp_p={clamp_p:.2f}, device={device}, "
        f"fast={fast_time*1000:.2f}ms, slow={slow_time*1000:.2f}ms, "
        f"speedup={(slow_time/fast_time):.2f}x"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes = [(8, 1024), (4, 4096)]
    clamp_ps = [0.0, 0.1, 0.3, 0.6]

    for shape in shapes:
        for clamp_p in clamp_ps:
            compare_once(shape, clamp_p, device)


if __name__ == "__main__":
    main()

