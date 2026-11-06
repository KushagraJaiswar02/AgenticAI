"""Ollama local chat provider."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any

import httpx

from app.complexity import TaskComplexityRouter
from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ConversationMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition
from app.model_router import ModelRouter


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
        self._simple_model = settings.ollama_general_model
        self._complex_model = settings.ollama_coding_model
        self._tool_model = settings.ollama_tool_model
        self._vision_model = settings.ollama_vision_model
        self._keep_alive = settings.ollama_keep_alive
        self._context_window = settings.ollama_context_window
        self._timeout = settings.ollama_timeout
        self._fallback_enabled = settings.fallback_enabled
        self._client = client or httpx.Client(timeout=self._timeout)
        self._complexity_router = complexity_router or TaskComplexityRouter()
        self._model_router = ModelRouter(settings)
        self._logger = logger or logging.getLogger("jarvis")
        self._pending_models: dict[str, str] = {}
        self._active_model: str | None = None
        self._request_lock = threading.RLock()

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
        conversation: list[ConversationMessage] | None = None,
        images: list[str] | None = None,
    ) -> LLMResponse:
        with self._request_lock:
            return self._generate_locked(prompt, tools=tools, tool_call=tool_call, tool_result=tool_result, think=think, conversation=conversation, images=images)

    def _generate_locked(self, prompt, *, tools=None, tool_call=None, tool_result=None, think=None, conversation=None, images=None):
        tool_intent = bool(tools) and any(word in prompt.lower() for word in (
            "weather", "time", "date", "calculate", "open ", "create file", "read file",
            "write file", "delete file", "list files", "search files", "copy file", "move file",
        ))
        decision = self._model_router.local_model(prompt, has_tools=tool_intent, has_image=bool(images))
        complexity = self._complexity_router.classify(prompt)
        continuation_key = (tool_call.call_id or tool_call.name) if tool_call is not None and tool_result is not None else None
        selected_model = self._pending_models.get(continuation_key) if continuation_key else None
        selected_model = selected_model or decision.model
        if images and selected_model != self._vision_model:
            raise LLMProviderError("Selected Ollama model does not support image input")
        timeout = self._timeout
        self._logger.info(
            "provider=ollama model=%s complexity=%s",
            selected_model,
            complexity.value,
        )
        if tool_call is not None and tool_result is not None and continuation_key not in self._pending_models:
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
            ]
            for item in conversation or []:
                if item.role in {"user", "assistant"}:
                    messages.append({"role": item.role, "content": item.content})
            messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"num_ctx": self._context_window},
        }
        if images:
            messages[-1]["images"] = images
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
            if self._active_model and self._active_model != selected_model:
                unload = self._client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": self._active_model, "keep_alive": 0},
                    timeout=timeout,
                )
                unload.raise_for_status()
                self._active_model = None
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            try:
                response.raise_for_status()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                fallback_model = self._simple_model
                if status != 404 or images or not self._fallback_enabled or selected_model == fallback_model:
                    raise
                self._logger.warning("requested Ollama model unavailable; trying configured text fallback")
                payload["model"] = fallback_model
                selected_model = fallback_model
                response = self._client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
            response.raise_for_status()
            data = response.json()
            self._active_model = selected_model
            self._logger.info(
                "provider=ollama model=%s complexity=%s success=true",
                selected_model,
                complexity.value,
            )
        except Exception as exc:
            self._logger.warning(
                "provider=ollama model=%s complexity=%s success=false",
                selected_model,
                complexity.value,
            )
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                raise LLMProviderError(f"Requested Ollama model is unavailable: {selected_model}") from exc
            raise LLMProviderError("Ollama request failed") from exc

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise InvalidProviderResponseError("Ollama returned an invalid response")
        calls = message.get("tool_calls") or []
        if calls:
            call = calls[0]
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise InvalidProviderResponseError("Ollama returned a malformed tool call")
            arguments = function.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise InvalidProviderResponseError("Ollama returned a malformed tool call")
            call_id = call.get("id") if isinstance(call, dict) else None
            call_id = call_id or str(uuid.uuid4())
            self._pending_models[call_id] = selected_model
            return LLMResponse(
                tool_call=ToolCall(
                    name=function["name"],
                    arguments=arguments,
                    call_id=call_id,
                )
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            if continuation_key:
                self._pending_models.pop(continuation_key, None)
            raise InvalidProviderResponseError("Ollama returned an empty response")
        if continuation_key:
            self._pending_models.pop(continuation_key, None)
        return LLMResponse(text=content.strip())
