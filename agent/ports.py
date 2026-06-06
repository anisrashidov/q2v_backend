"""Abstract port interfaces.

agent/ code depends ONLY on this module (and agent/types.py).
Adapters in adapters/ implement these protocols; tests supply fakes.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Minimal interface the agent needs from any LLM provider."""

    async def complete(self, system: str, user: str, schema: dict) -> dict:
        """Send a structured-output chat completion.

        Args:
            system: System-level instructions for the model.
            user:   The user message / prompt.
            schema: JSON Schema object the return value MUST conform to.

        Returns:
            A dict conforming to *schema*.

        Raises:
            ValueError: If the model fails to return a conforming response.
        """
        ...
