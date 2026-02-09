"""DIME (Diverse Instruction-Masked Exploration) focus instruction registry.

Provides per-environment focus instructions, sampling, and injection utilities
for the DIME technique. During rollout, a random focus instruction is appended
to each rollout's prompt. During training, the focus is stripped so the model
learns focus-guided behaviors without depending on the instruction at inference.
"""

import copy
import random
from typing import Optional

OVERCOOKED_FOCUS_INSTRUCTIONS = [
    "Pick up ingredients (e.g., onions) from ingredient piles using 'interact'",
    "Place 3 ingredients in a pot using 'interact' while facing it",
    "Wait for the soup to cook",
    "Pick up a dish from the dish pile",
    "Pick up the cooked soup from the pot (with dish in hand)",
    "Deliver the soup to the serving counter using 'interact'",
]

FOCUS_REGISTRY: dict[str, list[str]] = {
    "overcooked": OVERCOOKED_FOCUS_INSTRUCTIONS,
}


def get_focus_instructions(env_name: str) -> list[str]:
    """Return focus instruction steps for a given environment."""
    key = env_name.lower()
    if key not in FOCUS_REGISTRY:
        raise ValueError(
            f"No DIME focus instructions registered for env '{env_name}'. "
            f"Available: {list(FOCUS_REGISTRY.keys())}"
        )
    return FOCUS_REGISTRY[key]


def sample_focus_for_episode(
    n_rollouts: int,
    instructions: list[str],
    no_supplement_prob: float = 0.143,
) -> list[Optional[str]]:
    """Sample one focus instruction per rollout for an entire episode.

    Each rollout independently gets:
    - None (no focus) with probability no_supplement_prob
    - A uniformly random instruction otherwise
    """
    result = []
    for _ in range(n_rollouts):
        if random.random() < no_supplement_prob:
            result.append(None)
        else:
            result.append(random.choice(instructions))
    return result


def inject_focus_into_obs(
    obs_vec: list[list[dict]],
    focus_per_rollout: list[Optional[str]],
    template: str,
) -> list[list[dict]]:
    """Deepcopy obs_vec and append focus text to the last user message per rollout.

    For rollouts where focus is None, the observation is unchanged (still deepcopied).
    """
    result = copy.deepcopy(obs_vec)
    for i, focus in enumerate(focus_per_rollout):
        if focus is None:
            continue
        focus_text = template.replace("{STEP_TEXT}", focus)
        # Find last user message and append focus text
        for msg in reversed(result[i]):
            if msg["role"] == "user":
                msg["content"] = msg["content"] + "\n\n" + focus_text
                break
    return result
