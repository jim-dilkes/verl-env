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
    "Identify the ingredients needed for the meal from the recipe",
    "Pick up ingredients from ingredient piles using 'interact' while facing them - you can hold only one ingredient at a time",
    "Place 3 ingredients in a pot using 'interact'",
    "Wait for the meal to cook",
    "Pick up a dish from the dish pile using 'interact'",
    "Pick up the cooked meal from the pot using 'interact' (with dish in hand)",
    "Deliver the meal to the serving counter using 'interact'",
]

FOCUS_REGISTRY: dict[str, list[str]] = {
    "overcooked": OVERCOOKED_FOCUS_INSTRUCTIONS,
}


def has_focus_instructions(env_name: str) -> bool:
    """Check whether focus instructions are registered for an environment."""
    return env_name.lower() in FOCUS_REGISTRY


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
    no_supplement_prob: float = 0.125,
) -> list[Optional[str]]:
    """Sample one focus instruction per rollout for an entire episode.

    Each rollout independently gets:
    - None (no focus) with probability no_supplement_prob
    - A uniformly random instruction otherwise

    Default no_supplement_prob=0.125 = 1/(7+1) for 7 Overcooked instructions.
    Caller should pass 1/(N+1) for other instruction counts.
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
