"""Test Group 1 [P1]: DIME focus instruction registry, sampling, injection."""

from verl.envs.environments.focus_instructions import (
    get_focus_instructions,
    has_focus_instructions,
    get_dime_instructions,
    has_dime_instructions,
    GENERIC_FOCUS_INSTRUCTIONS,
    sample_focus_for_episode,
    inject_focus_into_obs,
)


def test_has_focus_instructions():
    assert has_focus_instructions("overcooked") is True
    assert has_focus_instructions("Overcooked") is True
    assert has_focus_instructions("nonexistent_env") is False


def test_get_focus_instructions():
    instructions = get_focus_instructions("overcooked")
    assert len(instructions) == 7
    assert "ingredient" in instructions[0].lower()

    # Case insensitive lookup
    instructions2 = get_focus_instructions("Overcooked")
    assert instructions == instructions2

    # Unknown env
    try:
        get_focus_instructions("nonexistent_env")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_sample_focus_for_episode():
    instructions = get_focus_instructions("overcooked")

    samples = sample_focus_for_episode(8, instructions, no_supplement_prob=0.143)
    assert len(samples) == 8
    assert all(s is None or s in instructions for s in samples)

    # All None with prob=1.0
    samples_none = sample_focus_for_episode(10, instructions, no_supplement_prob=1.0)
    assert all(s is None for s in samples_none)

    # All focus with prob=0.0
    samples_all = sample_focus_for_episode(10, instructions, no_supplement_prob=0.0)
    assert all(s is not None for s in samples_all)


def test_no_supplement_prob_statistical():
    instructions = get_focus_instructions("overcooked")
    none_count = sum(
        1 for _ in range(10000) if sample_focus_for_episode(1, instructions, 0.5)[0] is None
    )
    assert 4500 < none_count < 5500, f"Expected ~5000 None, got {none_count}"


def test_inject_focus_preserves_original():
    orig = [[{"role": "user", "content": "Hello"}]]
    focus = ["step1"]
    template = 'Focus: "{STEP_TEXT}"'
    result = inject_focus_into_obs(orig, focus, template)

    # Original unchanged
    assert orig[0][0]["content"] == "Hello"
    # Copy has focus appended
    assert "step1" in result[0][-1]["content"]
    assert "Focus:" in result[0][-1]["content"]


def test_inject_focus_none_leaves_unchanged():
    orig = [[{"role": "user", "content": "Hello"}]]
    result = inject_focus_into_obs(orig, [None], 'Focus: "{STEP_TEXT}"')
    assert result[0][-1]["content"] == "Hello"


def test_inject_focus_multi_message():
    """Focus appends to the LAST user message."""
    orig = [
        [
            {"role": "system", "content": "You are a chef."},
            {"role": "user", "content": "First obs"},
            {"role": "assistant", "content": "I will cook."},
            {"role": "user", "content": "Second obs"},
        ]
    ]
    result = inject_focus_into_obs(orig, ["step1"], 'Focus: "{STEP_TEXT}"')
    # Last user message ("Second obs") should have focus
    assert "step1" in result[0][3]["content"]
    # First user message should NOT
    assert "step1" not in result[0][1]["content"]


def test_get_dime_instructions_specific():
    instructions = get_dime_instructions("overcooked", "specific")
    assert len(instructions) == 7
    assert instructions == get_focus_instructions("overcooked")


def test_get_dime_instructions_generic():
    instructions = get_dime_instructions("overcooked", "generic")
    assert instructions is GENERIC_FOCUS_INSTRUCTIONS
    assert len(instructions) == 10


def test_get_dime_instructions_generic_any_env():
    instructions = get_dime_instructions("nonexistent_env", "generic")
    assert instructions is GENERIC_FOCUS_INSTRUCTIONS


def test_get_dime_instructions_specific_unknown_env():
    try:
        get_dime_instructions("nonexistent_env", "specific")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_get_dime_instructions_invalid_source():
    try:
        get_dime_instructions("overcooked", "invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e)


def test_has_dime_instructions_generic():
    assert has_dime_instructions("overcooked", "generic") is True
    assert has_dime_instructions("nonexistent_env", "generic") is True


def test_has_dime_instructions_specific():
    assert has_dime_instructions("overcooked", "specific") is True
    assert has_dime_instructions("snake", "specific") is False


def test_inject_focus_passthrough_template():
    """Generic mode: template='{STEP_TEXT}' passes raw instruction text."""
    instruction = "Think about your next move carefully."
    orig = [[{"role": "user", "content": "Hello"}]]
    result = inject_focus_into_obs(orig, [instruction], "{STEP_TEXT}")
    assert result[0][-1]["content"] == "Hello\n\n" + instruction


def test_sample_focus_requires_no_supplement_prob():
    """no_supplement_prob is required (no default)."""
    instructions = get_focus_instructions("overcooked")
    try:
        sample_focus_for_episode(8, instructions)
        assert False, "Should have raised TypeError for missing no_supplement_prob"
    except TypeError:
        pass


if __name__ == "__main__":
    test_has_focus_instructions()
    test_get_focus_instructions()
    test_sample_focus_for_episode()
    test_no_supplement_prob_statistical()
    test_inject_focus_preserves_original()
    test_inject_focus_none_leaves_unchanged()
    test_inject_focus_multi_message()
    test_get_dime_instructions_specific()
    test_get_dime_instructions_generic()
    test_get_dime_instructions_generic_any_env()
    test_get_dime_instructions_specific_unknown_env()
    test_get_dime_instructions_invalid_source()
    test_has_dime_instructions_generic()
    test_has_dime_instructions_specific()
    test_inject_focus_passthrough_template()
    test_sample_focus_requires_no_supplement_prob()
    print("All Group 1 tests passed!")
