"""Cerebras chat-completions provider."""

from __future__ import annotations

from typing import Any

from app.chat_provider import ChatCompletionProvider
from app.config import Settings
from app.errors import LLMProviderError


class CerebrasProvider(ChatCompletionProvider):
    provider_name = "Cerebras"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.cerebras_api_key:
            raise LLMProviderError("Cerebras API key is not configured")
        if not settings.cerebras_model:
            raise LLMProviderError("Cerebras model is not configured")
        if client is None:
            try:
                from cerebras.cloud.sdk import Cerebras
                client = Cerebras(api_key=settings.cerebras_api_key, timeout=settings.cerebras_timeout)
            except Exception as exc:
                raise LLMProviderError("Cerebras SDK is unavailable") from exc
        super().__init__(settings, client, settings.cerebras_model, settings.cerebras_timeout)
