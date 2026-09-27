"""Groq chat-completions provider."""

from __future__ import annotations

from typing import Any

from app.chat_provider import ChatCompletionProvider
from app.config import Settings
from app.errors import LLMProviderError


class GroqProvider(ChatCompletionProvider):
    provider_name = "Groq"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.groq_api_key:
            raise LLMProviderError("Groq API key is not configured")
        if not settings.groq_model:
            raise LLMProviderError("Groq model is not configured")
        if client is None:
            try:
                from groq import Groq
                client = Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
            except Exception as exc:
                raise LLMProviderError("Groq SDK is unavailable") from exc
        super().__init__(settings, client, settings.groq_model, settings.groq_timeout)
