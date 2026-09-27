"""Shared normalization for OpenAI-compatible chat-completion SDKs."""

from __future__ import annotations

import json
from typing import Any

from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse, ToolCall, ToolDefinition


class ChatCompletionProvider(LLMProvider):
    """Base adapter for providers exposing chat.completions.create."""

    provider_name = "chat"

    def __init__(self, settings: Any, client: Any, model: str, timeout: float) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout

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
            {"role": "system", "content": (
                "Tools are optional capabilities. If no tool is relevant, answer "
                "the user's question directly using your own knowledge."
            )},
            {"role": "user", "content": prompt},
        ]
        if tool_call is not None and tool_result is not None:
            messages.extend([
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tool_call.call_id or tool_call.name,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id or tool_call.name,
                    "content": json.dumps(tool_result),
                },
            ])
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
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
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMProviderError(f"{self.provider_name} request failed") from exc
        return self._normalize(response)

    def _normalize(self, response: Any) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise InvalidProviderResponseError(f"{self.provider_name} returned an invalid response")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise InvalidProviderResponseError(f"{self.provider_name} returned an invalid response")
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            call = calls[0]
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if not isinstance(name, str) or not name.strip():
                raise InvalidProviderResponseError(f"{self.provider_name} returned a malformed tool call")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, ValueError) as exc:
                raise InvalidProviderResponseError(f"{self.provider_name} returned a malformed tool call") from exc
            if not isinstance(parsed, dict):
                raise InvalidProviderResponseError(f"{self.provider_name} returned a malformed tool call")
            return LLMResponse(tool_call=ToolCall(
                name=name,
                arguments=parsed,
                call_id=getattr(call, "id", None),
            ))
        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderResponseError(f"{self.provider_name} returned an invalid response")
        return LLMResponse(text=text.strip())
