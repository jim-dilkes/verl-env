"""ICE (Instruction-Conditioned Exploration) focus instruction registry.

Provides per-environment focus instructions, sampling, and injection utilities
for the ICE technique. During rollout, a random focus instruction is appended
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

GENERIC_FOCUS_INSTRUCTIONS = [
    "Think several steps ahead before committing to an action.",
    "Focus on what will matter most in the next 2-3 moves, not the long term.",
    "Avoid risky actions — prefer safe, reliable choices even if slower.",
    "Be bold — take the action with the highest potential payoff even if uncertain.",
    "If your current plan isn't working, abandon it quickly and try something different.",
    "Pay attention to what's NOT happening — what's being neglected or blocked?",
    "What's the single most important thing to accomplish right now? Do only that.",
    "Question your assumptions — re-read the rules carefully. Is the action you're about to take actually valid, appropriate, and useful right now?",
    "Look back at your recent actions and reasoning. Did they actually achieve what you intended? Are your justifications consistent with the rules and your goals?",
    "What would a perfect player do in this exact situation?",
]


def has_focus_instructions(env_name: str) -> bool:
    """Check whether env-specific focus instructions are registered."""
    return env_name.lower() in FOCUS_REGISTRY


def get_focus_instructions(env_name: str) -> list[str]:
    """Return env-specific focus instruction steps."""
    key = env_name.lower()
    if key not in FOCUS_REGISTRY:
        raise ValueError(
            f"No ICE focus instructions registered for env '{env_name}'. "
            f"Available: {list(FOCUS_REGISTRY.keys())}"
        )
    return FOCUS_REGISTRY[key]


def has_ice_instructions(env_name: str, source: str) -> bool:
    """Check whether ICE instructions are available for the given source."""
    if source == "generic":
        return True
    if source == "specific":
        return has_focus_instructions(env_name)
    raise ValueError(f"Unknown ICE source: '{source}'. Must be 'specific' or 'generic'.")


def get_ice_instructions(env_name: str, source: str) -> list[str]:
    """Return ICE instructions for the given source."""
    if source == "generic":
        return GENERIC_FOCUS_INSTRUCTIONS
    if source == "specific":
        return get_focus_instructions(env_name)
    raise ValueError(f"Unknown ICE source: '{source}'. Must be 'specific' or 'generic'.")


def sample_focus_for_episode(
    n_rollouts: int,
    instructions: list[str],
    no_supplement_prob: float,
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


def validate_deterministic_assignment(
    n_rollouts: int,
    n_instructions: int,
    n_duplicates: int,
    n_no_instruction: int,
) -> None:
    """Assert the deterministic group fills exactly n_rollouts.

    group_size = n_instructions * n_duplicates + n_no_instruction must equal
    n_rollouts (mirrors OpenRLHF DICEConfig.group_size).
    """
    if n_duplicates < 1:
        raise ValueError(f"ice.n_duplicates={n_duplicates} must be >= 1.")
    if n_no_instruction < 0:
        raise ValueError(f"ice.n_no_instruction={n_no_instruction} must be >= 0.")
    group_size = n_instructions * n_duplicates + n_no_instruction
    if group_size != n_rollouts:
        raise ValueError(
            f"deterministic ICE assignment: n_instructions*n_duplicates + "
            f"n_no_instruction = {n_instructions}*{n_duplicates} + {n_no_instruction} "
            f"= {group_size} != n_rollouts ({n_rollouts}). Adjust n_duplicates / "
            f"n_no_instruction so the group fills exactly n_rollouts."
        )


def assign_focus_deterministic(
    n_rollouts: int,
    instructions: list[str],
    n_duplicates: int,
    n_no_instruction: int,
    seed: int,
) -> list[Optional[str]]:
    """Deterministic covering assignment of focuses across a task group.

    Builds exactly n_duplicates copies of each instruction plus n_no_instruction
    unconditioned (None) slots, then shuffles per group using `seed` so the
    instruction identity is decorrelated from rollout index (mirrors OpenRLHF
    DICEAugmentor base-assignment + per-prompt shuffle) while remaining
    reproducible. Requires the group-size constraint (validated).
    """
    validate_deterministic_assignment(
        n_rollouts, len(instructions), n_duplicates, n_no_instruction
    )
    assignment: list[Optional[str]] = []
    for instr in instructions:
        assignment.extend([instr] * n_duplicates)
    assignment.extend([None] * n_no_instruction)
    rng = random.Random(seed)
    rng.shuffle(assignment)
    return assignment


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
                msg["content"] = msg["content"].rstrip() + "\n\n" + focus_text
                break
    return result
