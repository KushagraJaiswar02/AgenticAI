"""OpenRouter provider through its OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from app.chat_provider import ChatCompletionProvider
from app.config import Settings
from app.errors import LLMProviderError


class OpenRouterProvider(ChatCompletionProvider):
    provider_name = "OpenRouter"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.openrouter_api_key:
            raise LLMProviderError("OpenRouter API key is not configured")
        if not settings.openrouter_model:
            raise LLMProviderError("OpenRouter model is not configured")
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=settings.openrouter_timeout,
                )
            except Exception as exc:
                raise LLMProviderError("OpenRouter SDK client is unavailable") from exc
        super().__init__(settings, client, settings.openrouter_model, settings.openrouter_timeout)
