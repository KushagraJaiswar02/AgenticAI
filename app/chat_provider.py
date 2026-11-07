"""Shared normalization for OpenAI-compatible chat-completion SDKs."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from app.browser_observation import browser_summary_input
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ConversationMessage, LLMProvider, LLMResponse, ProviderTurn, ToolCall, ToolDefinition
from app.research_budget import ResearchLimitReached, active_research, research_timeout


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
        conversation: list[ConversationMessage] | None = None,
        turn: ProviderTurn | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        del think
        continuation = tool_call is not None and tool_result is not None
        turn = turn or (tool_call._turn if continuation else None) or ProviderTurn()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instruction or (
                "Tools are optional capabilities. If no tool is relevant, answer "
                "the user's question directly using your own knowledge."
            )},
        ]
        for item in conversation or []:
            if item.role in {"user", "assistant"}:
                messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": prompt})
        tools = tools if tools is not None else turn.tools
        observation_only = continuation and not tools and system_instruction is not None and (
            tool_call.name == "get_page_content" or tool_result.get("navigation_status") == "duplicate_not_executed"
        )
        if observation_only:
            messages[-1]["content"] = browser_summary_input(prompt, tool_result)
        elif continuation:
            messages = [
                *(turn.messages or [*messages, self._assistant_call(tool_call)]),
                {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id or tool_call.name,
                    "content": json.dumps(tool_result),
                },
            ]
            if system_instruction is not None and messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": system_instruction}
        payload: dict[str, Any] = {
            "model": turn.model or self._model,
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
            response = self._create_completion(turn=turn, continuation=continuation, **payload)
        except (InvalidProviderResponseError, ResearchLimitReached):
            raise
        except Exception as exc:
            raise LLMProviderError(f"{self.provider_name} request failed") from exc
        model = getattr(response, "model", None)
        if continuation and turn.backend_model and model and model != turn.backend_model:
            raise InvalidProviderResponseError(f"{self.provider_name} changed models during tool continuation")
        normalized = self._normalize(response)
        self._verify_backend(response, normalized, turn, continuation=continuation)
        if normalized.tool_call is not None:
            native_message = response.choices[0].message
            assistant = self._assistant_call(normalized.tool_call)
            native_arguments = native_message.tool_calls[0].function.arguments
            if isinstance(native_arguments, str):
                assistant["tool_calls"][0]["function"]["arguments"] = native_arguments
            if isinstance(getattr(native_message, "content", None), str):
                assistant["content"] = native_message.content
            if isinstance(getattr(native_message, "reasoning_details", None), list):
                assistant["reasoning_details"] = native_message.reasoning_details
            turn.model = payload["model"]
            turn.backend_model = turn.backend_model or model
            turn.messages = [*messages, assistant]
            turn.tools = list(tools)
            normalized = replace(normalized, tool_call=replace(normalized.tool_call, _turn=turn))
        return normalized

    def _create_completion(self, *, turn: ProviderTurn, continuation: bool, **payload: Any) -> Any:
        client = self._client
        if active_research.get() is not None:
            payload["timeout"] = research_timeout(self._timeout)
            if callable(getattr(client, "with_options", None)):
                client = client.with_options(max_retries=0)
        return client.chat.completions.create(**payload)

    def _verify_backend(
        self, response: Any, normalized: LLMResponse, turn: ProviderTurn, *, continuation: bool,
    ) -> None:
        """Allow an adapter to verify routing before returning an executable call."""

    @staticmethod
    def _assistant_call(call: ToolCall) -> dict[str, Any]:
        return {
            "role": "assistant",
            "tool_calls": [{
                "id": call.call_id or call.name,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }],
        }

    def _normalize(self, response: Any) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise InvalidProviderResponseError(f"{self.provider_name} returned an invalid response")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise InvalidProviderResponseError(f"{self.provider_name} returned an invalid response")
        calls = getattr(message, "tool_calls", None) or []
        if len(calls) > 1:
            raise InvalidProviderResponseError(f"{self.provider_name} returned multiple tool calls in one response")
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
