"""OpenAI Responses API adapter."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse
from app.tools import detect_tool_call


class OpenAIProvider(LLMProvider):
    """Adapt the official OpenAI SDK to JARVIS's provider-neutral contract."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, prompt: str) -> LLMResponse:
        tool_call = detect_tool_call(prompt)
        if tool_call is not None:
            return LLMResponse(
                text=f"I will check the weather for {tool_call.arguments.get('location', 'that location')}.",
                tool_call=tool_call,
            )

        try:
            response = self._client.responses.create(model=self._model, input=prompt)
        except Exception as exc:
            raise LLMProviderError("OpenAI request failed") from exc

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderResponseError("OpenAI returned an invalid response")
        return LLMResponse(text=text.strip())
