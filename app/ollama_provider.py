"""Ollama local chat provider."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any

import httpx

from app.browser_observation import browser_summary_input
from app.complexity import TaskComplexityRouter
from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ConversationMessage, LLMProvider, LLMResponse, ProviderTurn, ToolCall, ToolDefinition
from app.model_router import ModelRouter
from app.research_budget import ResearchLimitReached, active_research, research_timeout


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
        turn: ProviderTurn | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        with self._request_lock:
            return self._generate_locked(
                prompt, tools=tools, tool_call=tool_call, tool_result=tool_result, think=think,
                conversation=conversation, images=images, turn=turn, system_instruction=system_instruction,
            )

    def _generate_locked(
        self, prompt, *, tools=None, tool_call=None, tool_result=None, think=None,
        conversation=None, images=None, turn=None, system_instruction=None,
    ):
        tool_intent = bool(tools) and any(word in prompt.lower() for word in (
            "weather", "time", "date", "calculate", "open ", "create file", "read file",
            "write file", "delete file", "list files", "search files", "copy file", "move file",
        ))
        decision = self._model_router.local_model(prompt, has_tools=tool_intent, has_image=bool(images))
        complexity = self._complexity_router.classify(prompt)
        continuation = tool_call is not None and tool_result is not None
        turn = turn or (tool_call._turn if continuation else None)
        if continuation and (turn is None or not turn.model or not turn.messages):
            raise LLMProviderError("Ollama tool continuation context is unavailable")
        turn = turn or ProviderTurn()
        selected_model = turn.model or decision.model
        tools = tools if tools is not None else turn.tools
        if images and selected_model != self._vision_model:
            raise LLMProviderError("Selected Ollama model does not support image input")
        observation_only = continuation and not tools and system_instruction is not None and (
            tool_call.name == "get_page_content" or tool_result.get("navigation_status") == "duplicate_not_executed"
        )
        if continuation and not observation_only:
            messages = [*turn.messages, {
                "role": "tool",
                "tool_name": tool_call.name,
                "content": json.dumps(tool_result),
            }]
            if system_instruction is not None and messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": system_instruction}
        else:
            messages = [{"role": "system", "content": system_instruction or self._SYSTEM_INSTRUCTION}]
            for item in conversation or []:
                if item.role in {"user", "assistant"}:
                    messages.append({"role": item.role, "content": item.content})
            user_message = {
                "role": "user",
                "content": browser_summary_input(prompt, tool_result) if observation_only else prompt,
            }
            if images:
                user_message["images"] = images
            messages.append(user_message)
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"num_ctx": self._context_window},
        }
        if selected_model == self._complex_model:
            payload["think"] = self._complexity_router.should_think(prompt) if think is None else think
        if tools and selected_model != self._vision_model:
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
                    timeout=research_timeout(self._timeout),
                )
                unload.raise_for_status()
                self._active_model = None
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=research_timeout(self._timeout),
            )
            try:
                response.raise_for_status()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                fallback_model = self._simple_model
                if status != 404 or continuation or images or active_research.get() is not None or not self._fallback_enabled or selected_model == fallback_model:
                    raise
                self._logger.warning("requested Ollama model unavailable; trying configured text fallback")
                payload["model"] = fallback_model
                payload.pop("think", None)
                selected_model = fallback_model
                response = self._client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=research_timeout(self._timeout),
                )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise InvalidProviderResponseError("Ollama returned invalid JSON") from exc
            self._active_model = selected_model
            self._logger.info(
                "provider=ollama model=%s complexity=%s success=true",
                selected_model,
                complexity.value,
            )
        except (InvalidProviderResponseError, ResearchLimitReached):
            raise
        except Exception as exc:
            self._logger.warning(
                "provider=ollama model=%s complexity=%s success=false",
                selected_model,
                complexity.value,
            )
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                raise LLMProviderError(f"Requested Ollama model is unavailable: {selected_model}", model=selected_model) from exc
            raise LLMProviderError("Ollama request failed", model=selected_model) from exc

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise InvalidProviderResponseError("Ollama returned an invalid response")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise InvalidProviderResponseError("Ollama returned malformed tool calls")
        if len(calls) > 1:
            raise InvalidProviderResponseError("Ollama returned multiple tool calls in one response")
        if calls:
            call = calls[0]
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
                raise InvalidProviderResponseError("Ollama returned a malformed tool call")
            arguments = function.get("arguments")
            arguments = {} if arguments is None else arguments
            if not isinstance(arguments, dict):
                raise InvalidProviderResponseError("Ollama returned a malformed tool call")
            call_id = call.get("id") if isinstance(call, dict) else None
            call_id = call_id or str(uuid.uuid4())
            turn.model = selected_model
            turn.messages = [*messages, {**message, "role": "assistant"}]
            turn.tools = list(tools)
            return LLMResponse(
                tool_call=ToolCall(
                    name=function["name"],
                    arguments=arguments,
                    call_id=call_id,
                    _turn=turn,
                )
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise InvalidProviderResponseError("Ollama returned an empty response")
        return LLMResponse(text=content.strip())
