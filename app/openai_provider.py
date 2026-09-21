"""OpenAI Responses API adapter."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse, ToolCall, ToolDefinition


class OpenAIProvider(LLMProvider):
    """Adapt the official OpenAI SDK to JARVIS's provider-neutral contract."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if tool_result is not None and tool_call is not None:
            return self._generate_after_tool_result(tool_call, tool_result)

        payload: dict[str, Any] = {"model": self._model, "input": prompt}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        try:
            response = self._client.responses.create(**payload)
        except Exception as exc:
            raise LLMProviderError("OpenAI request failed") from exc

        tool = self._extract_tool_call(response)
        if tool is not None:
            return LLMResponse(text="", tool_call=tool)

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderResponseError("OpenAI returned an invalid response")
        return LLMResponse(text=text.strip())

    def _generate_after_tool_result(self, tool_call: ToolCall, tool_result: dict[str, Any]) -> LLMResponse:
        call_id = tool_call.call_id or tool_call.name
        try:
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(tool_result),
                    },
                    {"type": "message", "role": "user", "content": "Return a concise answer based on the tool result."},
                ],
            )
        except Exception as exc:
            raise LLMProviderError("OpenAI request failed") from exc

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderResponseError("OpenAI returned an invalid response")
        return LLMResponse(text=text.strip())

    @staticmethod
    def _extract_tool_call(response: Any) -> ToolCall | None:
        try:
            output = getattr(response, "output", None) or []
            for item in output:
                if getattr(item, "type", None) == "function_call":
                    args = getattr(item, "arguments", None) or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not isinstance(args, dict):
                        raise InvalidProviderResponseError("OpenAI returned a malformed function call")
                    name = getattr(item, "name", None)
                    if not isinstance(name, str) or not name.strip():
                        raise InvalidProviderResponseError("OpenAI returned a malformed function call")
                    call_id = getattr(item, "call_id", None)
                    if call_id is not None and not isinstance(call_id, str):
                        raise InvalidProviderResponseError("OpenAI returned a malformed function call")
                    return ToolCall(name=name, arguments=args, call_id=call_id)
        except Exception as exc:
            raise InvalidProviderResponseError("OpenAI returned a malformed function call") from exc
        return None
