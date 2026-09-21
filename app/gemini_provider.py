"""Google Gemini generate-content adapter."""

from __future__ import annotations

from typing import Any

from google import genai

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse
from app.tools import detect_tool_call


class GeminiProvider(LLMProvider):
    """Adapt the official Google Gen AI SDK to JARVIS's LLM contract."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.gemini_api_key:
            raise LLMProviderError("Gemini API key is not configured")
        self._client = client or genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def generate(self, prompt: str) -> LLMResponse:
        tool_call = detect_tool_call(prompt)
        if tool_call is not None:
            return LLMResponse(
                text=f"I will check the weather for {tool_call.arguments.get('location', 'that location')}.",
                tool_call=tool_call,
            )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        except Exception as exc:
            raise LLMProviderError("Gemini request failed") from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderResponseError("Gemini returned an invalid response")
        return LLMResponse(text=text.strip())
