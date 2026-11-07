"""OpenRouter provider through its OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from app.chat_provider import ChatCompletionProvider
from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMResponse, ProviderTurn


class OpenRouterProvider(ChatCompletionProvider):
    provider_name = "OpenRouter"
    _GEMMA_FREE = "google/gemma-4-26b-a4b-it:free"

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

    def _create_completion(self, *, turn: ProviderTurn, continuation: bool, **payload: Any) -> Any:
        if self._model == self._GEMMA_FREE and continuation:
            endpoint = turn.backend_endpoint
            if not endpoint or not turn.backend_provider:
                raise InvalidProviderResponseError("OpenRouter continuation has no verified hosting provider")
            payload["extra_body"] = {
                "provider": {
                    "order": [endpoint],
                    "only": [endpoint],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            }
        return super()._create_completion(turn=turn, continuation=continuation, **payload)

    def _verify_backend(
        self, response: Any, normalized: LLMResponse, turn: ProviderTurn, *, continuation: bool,
    ) -> None:
        if self._model != self._GEMMA_FREE or (not continuation and normalized.tool_call is None):
            return
        if getattr(response, "model", None) != self._model:
            raise InvalidProviderResponseError("OpenRouter did not report the requested Gemma model")
        provider = getattr(response, "provider", None)
        if not isinstance(provider, str) or not provider.strip():
            raise InvalidProviderResponseError("OpenRouter did not report the Gemma hosting provider")
        identity = provider.strip().casefold()
        if continuation:
            if identity != turn.backend_provider:
                raise InvalidProviderResponseError("OpenRouter changed hosting providers during tool continuation")
            return

        turn.backend_endpoint = self._resolve_endpoint(identity)
        turn.backend_provider = identity

    def _resolve_endpoint(self, identity: str) -> str:
        # Chat Completions reports a display name. Routing needs a catalog tag.
        try:
            catalog = self._client.get(f"/models/{self._model}/endpoints", cast_to=object)
            data = catalog["data"]
            matches = [
                endpoint for endpoint in data["endpoints"]
                if endpoint["provider_name"].strip().casefold() == identity
            ]
        except Exception as exc:
            raise InvalidProviderResponseError("OpenRouter could not verify the Gemma hosting endpoint") from exc
        if data.get("id") == self._model and len(matches) == 1:
            tag = matches[0].get("tag")
            if isinstance(tag, str) and tag.strip() and "tools" in matches[0].get("supported_parameters", []):
                return tag
        raise InvalidProviderResponseError("OpenRouter could not uniquely identify a tool-capable Gemma hosting endpoint")
