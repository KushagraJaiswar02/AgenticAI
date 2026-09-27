"""Mistral chat-completions provider."""

from __future__ import annotations

from typing import Any

from app.chat_provider import ChatCompletionProvider
from app.config import Settings
from app.errors import LLMProviderError


class _MistralCompat:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = self
        self.completions = self

    def create(self, **payload: Any) -> Any:
        return self._client.chat.complete(**payload)


class MistralProvider(ChatCompletionProvider):
    provider_name = "Mistral"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.mistral_api_key:
            raise LLMProviderError("Mistral API key is not configured")
        if not settings.mistral_model:
            raise LLMProviderError("Mistral model is not configured")
        if client is None:
            try:
                from mistralai import Mistral
                client = Mistral(api_key=settings.mistral_api_key)
            except Exception as exc:
                raise LLMProviderError("Mistral SDK is unavailable") from exc
        if not hasattr(client, "chat") or not hasattr(getattr(client, "chat"), "completions"):
            client = _MistralCompat(client)
        super().__init__(settings, client, settings.mistral_model, settings.mistral_timeout)
