"""LLMPort implementation backed by the OpenAI Chat Completions API.

Uses tool-use with a single ``structured_output`` tool so the model is
forced to return JSON that matches the caller-supplied JSON Schema.
"""
from __future__ import annotations

import json

import openai

from agent.ports import LLMPort  # noqa: F401 — imported for type-checking


class OpenAILLM:
    """Concrete LLMPort adapter for OpenAI.

    Parameters
    ----------
    api_key:
        OpenAI API key (typically from settings.openai_api_key).
    model:
        OpenAI model ID, e.g. ``"gpt-4o"``.
    max_tokens:
        Hard cap on generated tokens; 2 048 is sufficient for structured
        responses and keeps latency low.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str, schema: dict) -> dict:
        """Force structured output via tool-use.

        The ``structured_output`` function is defined with *schema* as its
        ``parameters``; ``tool_choice`` forces the model to call it so it
        cannot produce free text instead.

        Returns
        -------
        dict
            The parsed arguments dict from the tool call, conforming to *schema*.

        Raises
        ------
        ValueError
            If the response contains no tool call.
        openai.APIError
            Propagated from the OpenAI SDK on network/auth errors.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "structured_output",
                        "description": (
                            "Return structured data that strictly conforms to the "
                            "provided JSON Schema. Fill every required field."
                        ),
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "structured_output"}},
        )

        message = response.choices[0].message
        if message.tool_calls:
            return json.loads(message.tool_calls[0].function.arguments)

        raise ValueError(
            f"OpenAI response contained no structured_output tool call. "
            f"finish_reason={response.choices[0].finish_reason!r}"
        )
