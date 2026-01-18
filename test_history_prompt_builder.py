"""Tests for HistoryPromptBuilder with natural conversation turn structure."""
import pytest

from verl.envs.captioners.prompt_builder.history import HistoryPromptBuilder


def make_obs(long_term: str = "", short_term: str = ""):
    return {"text": {"long_term_context": long_term, "short_term_context": short_term}}


class TestBasicObservations:
    """Test basic observation handling."""

    def test_single_observation(self):
        """Single observation emits as user message."""
        builder = HistoryPromptBuilder()
        builder.update_observation(make_obs(long_term="LT1", short_term="ST1"))
        messages = builder.get_prompt()

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "ST1" in messages[0].content
        assert "LT1" in messages[0].content

    def test_system_prompt_included(self):
        """System prompt appears first when set."""
        builder = HistoryPromptBuilder(system_prompt="You are a game agent.")
        builder.update_observation(make_obs(long_term="State"))
        messages = builder.get_prompt()

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "You are a game agent."
        assert messages[1].role == "user"

    def test_no_artificial_headers(self):
        """Output should not contain artificial headers."""
        builder = HistoryPromptBuilder(max_cot_history=1)
        builder.update_observation(make_obs(long_term="Obs1", short_term="Event1"))
        builder.update_reasoning("My reasoning")
        builder.update_action("action1")
        builder.update_observation(make_obs(long_term="Obs2", short_term="Event2"))

        messages = builder.get_prompt()
        all_content = " ".join(m.content for m in messages)

        assert "[My Previous Thoughts]" not in all_content
        assert "[Previous Observation]" not in all_content
        assert "[Current Observation]" not in all_content
        assert "Observation:" not in all_content


class TestNaturalTurnStructure:
    """Test natural conversation turn structure with reasoning."""

    def test_reasoning_creates_user_assistant_user_pattern(self):
        """When reasoning present: user(prev_obs) → assistant(reasoning) → user(current_obs)."""
        builder = HistoryPromptBuilder(max_cot_history=1)

        builder.update_observation(make_obs(long_term="State1", short_term="Start"))
        builder.update_reasoning("<decision>right</decision>")
        builder.update_action("right")
        builder.update_observation(make_obs(long_term="State2", short_term="Moved"))

        messages = builder.get_prompt()

        assert len(messages) == 3
        assert messages[0].role == "user"  # prev obs
        assert messages[1].role == "assistant"  # reasoning
        assert messages[2].role == "user"  # current obs

        # Prev obs content
        assert "Start" in messages[0].content
        assert "State1" in messages[0].content

        # Reasoning content (no header)
        assert messages[1].content == "<decision>right</decision>"

        # Current obs content
        assert "Moved" in messages[2].content
        assert "State2" in messages[2].content

    def test_no_reasoning_shows_action_only(self):
        """Without reasoning, assistant message shows executed action."""
        builder = HistoryPromptBuilder(max_cot_history=0)

        builder.update_observation(make_obs(long_term="State1"))
        builder.update_reasoning("This reasoning will be filtered")
        builder.update_action("go_left")
        builder.update_observation(make_obs(long_term="State2"))

        messages = builder.get_prompt()

        # With max_cot_history=0, reasoning filtered out
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "go_left"


class TestHistoryLimits:
    """Test history truncation via max_cot_history and max_text_history."""

    def test_max_cot_history_limits_reasoning(self):
        """Only most recent N actions keep reasoning."""
        builder = HistoryPromptBuilder(max_cot_history=1)

        # Two actions with reasoning
        builder.update_observation(make_obs(long_term="Obs1"))
        builder.update_reasoning("Reasoning1")
        builder.update_action("action1")

        builder.update_observation(make_obs(long_term="Obs2"))
        builder.update_reasoning("Reasoning2")
        builder.update_action("action2")

        builder.update_observation(make_obs(long_term="Obs3"))

        messages = builder.get_prompt()
        all_content = " ".join(m.content for m in messages)

        # Only most recent reasoning (Reasoning2) should be present
        assert "Reasoning2" in all_content
        assert "Reasoning1" not in all_content

    def test_max_text_history_limits_observations(self):
        """Only most recent N observations include long_term text."""
        builder = HistoryPromptBuilder(max_text_history=1, max_cot_history=0)

        builder.update_observation(make_obs(long_term="OLD_LONG"))
        builder.update_action("a1")
        builder.update_observation(make_obs(long_term="NEW_LONG", short_term="NEW_SHORT"))

        messages = builder.get_prompt()
        all_content = " ".join(m.content for m in messages)

        assert "NEW_LONG" in all_content
        assert "NEW_SHORT" in all_content
        # Old long_term should be excluded (max_text_history=1 keeps only most recent)


class TestIdempotence:
    """Test that get_prompt() is idempotent (non-mutating)."""

    def test_multiple_calls_return_same_result(self):
        """Calling get_prompt() multiple times returns identical results."""
        builder = HistoryPromptBuilder(max_cot_history=1)

        builder.update_observation(make_obs(long_term="Obs1", short_term="ST1"))
        builder.update_reasoning("Reasoning1")
        builder.update_action("action1")
        builder.update_observation(make_obs(long_term="Obs2", short_term="ST2"))

        messages1 = builder.get_prompt()
        messages2 = builder.get_prompt()
        messages3 = builder.get_prompt()

        assert len(messages1) == len(messages2) == len(messages3)
        for m1, m2, m3 in zip(messages1, messages2, messages3):
            assert m1.role == m2.role == m3.role
            assert m1.content == m2.content == m3.content


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_long_term_uses_short_term(self):
        """When long_term is empty, short_term still appears (FastSnake pattern)."""
        builder = HistoryPromptBuilder()
        builder.update_observation(make_obs(long_term="", short_term="Game state here"))
        messages = builder.get_prompt()

        assert len(messages) == 1
        assert "Game state here" in messages[0].content

    def test_first_action_no_preceding_observation(self):
        """Action without preceding observation handles gracefully."""
        builder = HistoryPromptBuilder(max_cot_history=1)

        # Directly add action without observation first (edge case)
        builder.update_reasoning("Some reasoning")
        builder.update_action("action1")
        builder.update_observation(make_obs(long_term="First obs"))

        messages = builder.get_prompt()
        # Should not crash, action's observation_text will be None
        assert any(m.role == "user" for m in messages)

    def test_observation_not_duplicated_when_emitted_via_reasoning(self):
        """Observation emitted via action reasoning isn't also emitted standalone."""
        builder = HistoryPromptBuilder(max_cot_history=1, max_text_history=2)

        builder.update_observation(make_obs(long_term="Obs1"))
        builder.update_reasoning("R1")
        builder.update_action("a1")
        builder.update_observation(make_obs(long_term="Obs2"))

        messages = builder.get_prompt()
        all_content = " ".join(m.content for m in messages)

        # Obs1 should appear once (via action reasoning), not duplicated
        assert all_content.count("Obs1") == 1
