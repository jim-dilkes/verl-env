import pytest

from verl.envs.captioners.prompt_builder.history import HistoryPromptBuilder


def make_obs(long_term: str = "", short_term: str = ""):
    return {"text": {"long_term_context": long_term, "short_term_context": short_term}}


def test_current_observation_only():
    """Mode A: only the current env observation, no history."""
    builder = HistoryPromptBuilder(
        use_llm_response_as_long_term=False,
        max_observation_messages=1,
    )
    print("test_current_observation_only")

    builder.update_observation(make_obs(long_term="LT1", short_term="ST1"))
    messages = builder.get_prompt()

    print(messages)

    assert len(messages) == 1
    assert messages[0].role == "user"
    content = messages[0].content
    assert "Current Observation:" in content
    assert "ST1" in content
    # Long-term context from env should appear when not using LLM responses.
    assert "LT1" in content


def test_current_observation_plus_prev_llm_response():
    """Mode B: current env observation + previous step's LLM response."""
    builder = HistoryPromptBuilder(
        use_llm_response_as_long_term=True,
        max_observation_messages=1,
    )
    print("test_current_observation_plus_prev_llm_response")

    # First step: observation then LLM response stored via update_action
    builder.update_observation(make_obs(long_term="LT1", short_term="ST1"))
    builder.update_action(action="move_up", full_response="LLM RESP 1")

    # Second step: new observation should surface previous LLM response as long-term
    builder.update_observation(make_obs(long_term="", short_term="ST2"))
    messages = builder.get_prompt()

    print(messages)

    # Expect assistant action message + current observation message
    assert len(messages) == 2

    assert messages[0].role == "assistant"
    assert messages[0].content == "move_up"

    assert messages[1].role == "user"
    content = messages[1].content
    assert "Current Observation:" in content
    # Current short-term context must be present
    assert "ST2" in content
    # Previous LLM response should be used as long-term context
    assert "LLM RESP 1" in content
    # Previous env short-term should not surface when capped to 1 observation
    assert "ST1" not in content


def test_observation_cap_keeps_most_recent_only():
    """When max_observation_messages=1, only the latest observation is emitted."""
    builder = HistoryPromptBuilder(
        use_llm_response_as_long_term=False,
        max_observation_messages=1,
    )
    print()
    print("test_observation_cap_keeps_most_recent_only")

    builder.update_observation(make_obs(long_term="LT1", short_term="ST1"))
    builder.update_observation(make_obs(long_term="LT2", short_term="ST2"))
    builder.update_observation(make_obs(long_term="LT3", short_term="ST3"))

    messages = builder.get_prompt()

    print(messages)

    assert len(messages) == 1
    assert messages[0].role == "user"
    content = messages[0].content
    assert "ST3" in content
    assert "LT3" in content
    # Earlier observations should be omitted
    assert "ST1" not in content and "ST2" not in content
    assert "LT1" not in content and "LT2" not in content


def test_actions_are_kept_while_observations_are_capped():
    """Observation cap should not drop action messages; only observations are limited."""
    builder = HistoryPromptBuilder(
        use_llm_response_as_long_term=True,
        max_observation_messages=2,
    )
    print()
    print("test_actions_are_kept_while_observations_are_capped")

    # Step 1
    builder.update_observation(make_obs(long_term="L1", short_term="S1"))
    builder.update_action(action="a1", full_response="RESP1")
    # Step 2
    builder.update_observation(make_obs(long_term="L2", short_term="S2"))
    builder.update_action(action="a2", full_response="RESP2")
    # Step 3
    builder.update_observation(make_obs(long_term="L3", short_term="S3"))
    builder.update_action(action="a3", full_response="RESP3")

    # Step 4
    builder.update_observation(make_obs(long_term="L4", short_term="S4"))
    builder.update_action(action="a4", full_response="RESP4")


    messages = builder.get_prompt()

    print(messages)

    # Expect: action1, action2, obs2, obs3 (obs1 dropped due to cap=2)
    assert len(messages) == 4
    assert messages[0].role == "assistant" and messages[0].content == "a2"
    assert messages[1].role == "assistant" and messages[1].content == "a3"

    # The two kept observations should be the last two
    obs_messages = [m for m in messages if m.role == "user"]
    assert len(obs_messages) == 2
    # Latest observation first in traversal order
    assert "S2" in obs_messages[0].content or "S3" in obs_messages[0].content
    assert "S3" in obs_messages[1].content or "S2" in obs_messages[1].content
    # Ensure the dropped first obs is not present
    assert all("S1" not in m.content for m in obs_messages)
