"""xAI Grok provider through its OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from app.chat_provider import ChatCompletionProvider
from app.config import Settings
from app.errors import LLMProviderError


class GrokProvider(ChatCompletionProvider):
    provider_name = "Grok"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.grok_api_key:
            raise LLMProviderError("Grok API key is not configured")
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.grok_api_key, base_url="https://api.x.ai/v1", timeout=settings.grok_timeout)
            except Exception as exc:
                raise LLMProviderError("Grok SDK client is unavailable") from exc
        super().__init__(settings, client, settings.grok_model, settings.grok_timeout)
