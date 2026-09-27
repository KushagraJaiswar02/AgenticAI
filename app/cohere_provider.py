"""Cohere Chat API provider."""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse, ToolCall, ToolDefinition


class CohereProvider(LLMProvider):
    """Adapt Cohere's current client interface to JARVIS."""

    provider_name = "Cohere"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.cohere_api_key:
            raise LLMProviderError("Cohere API key is not configured")
        if not settings.cohere_model:
            raise LLMProviderError("Cohere model is not configured")
        if client is None:
            try:
                from cohere import ClientV2
                client = ClientV2(api_key=settings.cohere_api_key)
            except Exception as exc:
                raise LLMProviderError("Cohere SDK is unavailable") from exc
        self._client = client
        self._model = settings.cohere_model

    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        del think
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Tools are optional capabilities; answer knowledge questions directly."},
            {"role": "user", "content": prompt},
        ]
        if tool_call is not None and tool_result is not None:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id or tool_call.name,
                "content": json.dumps(tool_result),
            })
        payload: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "description": item.description,
                        "parameters": item.parameters,
                    },
                }
                for item in tools
            ]
        try:
            response = self._client.chat(**payload)
        except Exception as exc:
            raise LLMProviderError("Cohere request failed") from exc
        return self._normalize(response)

    def _normalize(self, response: Any) -> LLMResponse:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str) and content.strip():
            return LLMResponse(text=content.strip())
        if isinstance(content, list):
            for block in content:
                block_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if block_type in {"text", "message"}:
                    text = getattr(block, "text", None) or (
                        block.get("text") if isinstance(block, dict) else None
                    )
                    if isinstance(text, str) and text.strip():
                        return LLMResponse(text=text.strip())
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if tool_calls:
            call = tool_calls[0]
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if isinstance(name, str) and isinstance(arguments, dict):
                return LLMResponse(tool_call=ToolCall(
                    name=name,
                    arguments=arguments,
                    call_id=getattr(call, "id", None),
                ))
        raise InvalidProviderResponseError("Cohere returned no user-visible answer")
