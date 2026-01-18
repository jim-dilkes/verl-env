from collections import deque
from typing import List, Optional


class Message:
    """Represents a conversation message with role, content, and optional attachment."""

    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 1,
        system_prompt: Optional[str] = None,
        max_cot_history: int = 1,
    ):
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.system_prompt = system_prompt
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history
        self.max_history = max(max_text_history, max_image_history, max_cot_history)
        self._events = deque(maxlen=self.max_history * 2 + 1)  # Store n actions and n+1 observations

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs["text"].get("long_term_context", "")
        short_term_context = obs["text"].get("short_term_context", "")
        self._last_short_term_obs = short_term_context

        self._events.append({
            "type": "observation",
            "text": long_term_context,
            "short_term": short_term_context,
            "image": obs.get("image", None),
        })

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        # Get observation text this action responds to
        obs_text = None
        for event in reversed(self._events):
            if event["type"] == "observation":
                parts = []
                if event.get("short_term"):
                    parts.append(event["short_term"])
                if event.get("text"):
                    parts.append(event["text"])
                obs_text = "\n".join(parts) if parts else None
                break

        self._events.append({
            "type": "action",
            "action": action,
            "reasoning": self.previous_reasoning,
            "observation_text": obs_text,
        })

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Note: This method is idempotent and does not mutate event history.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.system_prompt))

        # Build index-based inclusion sets (non-mutating)
        obs_include_text = set()
        obs_include_image = set()
        actions_with_reasoning = set()

        # Determine which text observations to include (most recent first)
        text_needed = self.max_text_history + 1
        for idx in range(len(self._events) - 1, -1, -1):
            event = self._events[idx]
            if event["type"] == "observation" and text_needed > 0 and event.get("text") is not None:
                obs_include_text.add(idx)
                text_needed -= 1

        # Determine which image observations to include
        images_needed = self.max_image_history
        for idx in range(len(self._events) - 1, -1, -1):
            event = self._events[idx]
            if event["type"] == "observation" and images_needed > 0 and event.get("image") is not None:
                obs_include_image.add(idx)
                images_needed -= 1

        # Determine which actions keep reasoning (most recent first)
        reasoning_needed = self.max_cot_history
        for idx in range(len(self._events) - 1, -1, -1):
            event = self._events[idx]
            if event["type"] == "action" and event.get("reasoning") is not None:
                if reasoning_needed > 0:
                    actions_with_reasoning.add(idx)
                    reasoning_needed -= 1

        # Identify observations emitted via action reasoning (to avoid duplicates)
        obs_via_actions = set()
        for idx in actions_with_reasoning:
            for prev_idx in range(idx - 1, -1, -1):
                if self._events[prev_idx]["type"] == "observation":
                    obs_via_actions.add(prev_idx)
                    break

        # Process events to create messages
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                include_text = idx in obs_include_text
                include_image = idx in obs_include_image
                is_current = idx == len(self._events) - 1

                # Skip observations already emitted via action reasoning
                if idx in obs_via_actions and not is_current:
                    continue

                # Skip past observations when nothing selected for inclusion
                has_short_term = bool(self._last_short_term_obs) if is_current else False
                if not include_text and not include_image and not has_short_term:
                    continue

                message_parts = []
                if is_current and has_short_term:
                    message_parts.append(self._last_short_term_obs)
                if include_text and event.get("text"):
                    message_parts.append(event["text"])

                image = None
                if include_image:
                    image = event["image"]
                    message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content, attachment=image)

            elif event["type"] == "action":
                has_reasoning = idx in actions_with_reasoning
                if has_reasoning:
                    # Emit previous observation as user turn first
                    if event.get("observation_text"):
                        obs_msg = Message(role="user", content=event["observation_text"])
                        messages.append(obs_msg)
                    content = event["reasoning"]
                else:
                    content = event["action"]
                message = Message(role="assistant", content=content)

            messages.append(message)

        return messages
