"""Ollama local chat provider."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.complexity import TaskComplexityRouter
from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse, ToolCall, ToolDefinition


class OllamaProvider(LLMProvider):
    """Adapt Ollama's local chat API to the JARVIS provider contract."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        complexity_router: TaskComplexityRouter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._base_url = settings.ollama_base_url
        legacy_model = settings.ollama_model
        self._simple_model = settings.ollama_simple_model
        self._complex_model = legacy_model or settings.ollama_complex_model
        self._simple_timeout = settings.ollama_simple_timeout
        self._complex_timeout = settings.ollama_complex_timeout
        self._client = client or httpx.Client(timeout=self._complex_timeout)
        self._complexity_router = complexity_router or TaskComplexityRouter()
        self._logger = logger or logging.getLogger("jarvis")
        self._pending_model: str | None = None

    _SYSTEM_INSTRUCTION = (
        "Tools are optional capabilities. Use a tool only when the user's request "
        "requires it. If no tool is relevant, answer the user's question directly "
        "using your own knowledge. The absence of a tool does not mean you cannot "
        "answer. For a tool plus knowledge request, use the tool and then explain "
        "the result."
    )

    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        complexity = self._complexity_router.classify(prompt)
        selected_model = self._pending_model or (
            self._complex_model if complexity.value == "complex" else self._simple_model
        )
        timeout = self._complex_timeout if complexity.value == "complex" else self._simple_timeout
        self._logger.info(
            "provider=ollama model=%s complexity=%s",
            selected_model,
            complexity.value,
        )
        if tool_call is not None and tool_result is not None and self._pending_model is None:
            raise LLMProviderError("Ollama tool continuation context is unavailable")
        if tool_call is not None and tool_result is not None:
            messages = [
                {"role": "system", "content": self._SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    }],
                },
                {"role": "tool", "content": json.dumps(tool_result)},
            ]
        else:
            messages = [
                {"role": "system", "content": self._SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ]
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
        }
        if selected_model == self._complex_model:
            payload["think"] = self._complexity_router.should_think(prompt) if think is None else think
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": item.name,
                    "description": item.description,
                    "parameters": item.parameters,
                }}
                for item in tools
            ]
        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._logger.info(
                "provider=ollama model=%s complexity=%s success=true",
                selected_model,
                complexity.value,
            )
        except Exception as exc:
            self._pending_model = None
            self._logger.warning(
                "provider=ollama model=%s complexity=%s success=false",
                selected_model,
                complexity.value,
            )
            raise LLMProviderError("Ollama request failed") from exc

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            self._pending_model = None
            raise InvalidProviderResponseError("Ollama returned an invalid response")
        calls = message.get("tool_calls") or []
        if calls:
            call = calls[0]
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                self._pending_model = None
                raise InvalidProviderResponseError("Ollama returned a malformed tool call")
            arguments = function.get("arguments") or {}
            if not isinstance(arguments, dict):
                self._pending_model = None
                raise InvalidProviderResponseError("Ollama returned a malformed tool call")
            self._pending_model = selected_model
            return LLMResponse(
                tool_call=ToolCall(
                    name=function["name"],
                    arguments=arguments,
                    call_id=call.get("id") if isinstance(call, dict) else None,
                )
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            self._pending_model = None
            raise InvalidProviderResponseError("Ollama returned an empty response")
        self._pending_model = None
        return LLMResponse(text=content.strip())
