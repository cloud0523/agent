from __future__ import annotations

from collections import deque
from typing import Any

from tiktoken import get_encoding


class ConversationMemory:
    """Store recent user/assistant turns with simple token-budget trimming."""

    def __init__(self, max_turns: int = 10, max_context_tokens: int = 4000):
        self._messages: deque[dict[str, Any]] = deque(maxlen=max_turns * 2)
        self._max_turns = max_turns
        self._max_tokens = max_context_tokens
        self._enc = get_encoding("cl100k_base")

    def add_user_message(self, content: str) -> None:
        """Append a user message to the conversation history."""
        self._messages.append({"role": "user", "content": content})
        self._trim_to_limits()

    def add_assistant_message(self, content: str, sources=None) -> None:
        """Append an assistant message.

        The sources argument is kept for future debugging and is ignored by the
        conversation history itself.
        """
        self._messages.append({"role": "assistant", "content": content})
        self._trim_to_limits()

    def get_history(self) -> list[dict]:
        """Return the recent history, trimmed to the configured limits."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear all stored history."""
        self._messages.clear()

    def _estimate_tokens(self) -> int:
        return sum(len(self._enc.encode(msg["content"])) for msg in self._messages)

    def _trim_to_limits(self) -> None:
        """Trim old messages when the turn count or token budget is exceeded."""
        while len(self._messages) > self._max_turns * 2:
            self._messages.popleft()

        while self._estimate_tokens() > self._max_tokens and len(self._messages) > 1:
            self._messages.popleft()
