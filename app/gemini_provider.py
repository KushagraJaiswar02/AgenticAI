"""Google Gemini generate-content adapter."""

from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai import types

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ConversationMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition


class GeminiProvider(LLMProvider):
    """Adapt the official Google Gen AI SDK to JARVIS's LLM contract."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.gemini_api_key:
            raise LLMProviderError("Gemini API key is not configured")
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.gemini_timeout * 1000),
            ),
        )
        self._model = settings.gemini_model
        self._pending_tool_context: Any | None = None
        self._pending_tool_history: list[Any] = []

    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
        think: bool | None = None,
        conversation: list[ConversationMessage] | None = None,
    ) -> LLMResponse:
        del think
        if tool_result is not None and tool_call is not None:
            return self._generate_after_tool_result(prompt, tool_call, tool_result, tools)

        self._pending_tool_context = None
        self._pending_tool_history = []
        tool_declarations = self._to_tool_declarations(tools or [])
        config = None
        if tool_declarations:
            config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=tool_declarations)])

        try:
            contents: Any = prompt
            if conversation:
                contents = [
                    types.Content(role=item.role, parts=[types.Part.from_text(text=item.content)])
                    for item in conversation
                ]
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            self._pending_tool_context = None
            self._pending_tool_history = []
            raise LLMProviderError("Gemini request failed") from exc

        call = self._extract_tool_call(response, tools)
        if call is not None:
            model_content = self._get_model_content(response)
            if model_content is None:
                textual = getattr(response, "text", None)
                if not isinstance(textual, str):
                    raise InvalidProviderResponseError("Gemini returned a malformed function call")
                model_content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=textual.strip())],
                )
            self._pending_tool_context = model_content
            self._pending_tool_history = [model_content]
            return LLMResponse(text="", tool_call=call)

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            self._pending_tool_context = None
            raise InvalidProviderResponseError("Gemini returned an invalid response")
        self._pending_tool_context = None
        return LLMResponse(text=text.strip())

    def _generate_after_tool_result(
        self,
        prompt: str,
        tool_call: ToolCall,
        tool_result: dict[str, Any],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        try:
            model_content = self._pending_tool_context
            if model_content is None:
                raise InvalidProviderResponseError("Gemini tool-call context is no longer available")

            function_response = types.Part.from_function_response(
                name=tool_call.name,
                response={"result": tool_result},
            )
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            contents.extend(self._pending_tool_history or [model_content])
            contents.append(types.Content(role="user", parts=[function_response]))
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
            )
            call = self._extract_tool_call(response, tools)
            if call is not None:
                model_content = self._get_model_content(response)
                if model_content is None:
                    raise InvalidProviderResponseError("Gemini returned a malformed function call")
                self._pending_tool_context = model_content
                self._pending_tool_history = [*(self._pending_tool_history or [model_content]), contents[-1], model_content]
                return LLMResponse(text="", tool_call=call)
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise InvalidProviderResponseError("Gemini returned an invalid response")
            self._pending_tool_context = None
            self._pending_tool_history = []
            return LLMResponse(text=text.strip())
        except InvalidProviderResponseError:
            self._pending_tool_context = None
            self._pending_tool_history = []
            raise
        except Exception as exc:
            self._pending_tool_context = None
            self._pending_tool_history = []
            raise LLMProviderError("Gemini request failed") from exc

    @staticmethod
    def _get_model_content(response: Any) -> Any | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        return getattr(candidates[0], "content", None)

    def _extract_tool_call(self, response: Any, tools: list[ToolDefinition] | None = None) -> ToolCall | None:
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                parts = getattr(candidates[0], "content", None)
                function_calls = getattr(parts, "parts", None) or [] if parts is not None else []
                for part in function_calls:
                    call = getattr(part, "function_call", None)
                    if call is not None:
                        name = getattr(call, "name", None)
                        if not isinstance(name, str) or not name.strip():
                            raise InvalidProviderResponseError("Gemini returned a malformed function call")
                        args = getattr(call, "args", None) or {}
                        if not isinstance(args, dict):
                            raise InvalidProviderResponseError("Gemini returned a malformed function call")
                        return ToolCall(name=name, arguments=args)
        except Exception as exc:
            raise InvalidProviderResponseError("Gemini returned a malformed function call") from exc
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return None
        return self._extract_textual_tool_call(text, tools)

    @staticmethod
    def _extract_textual_tool_call(text: str, tools: list[ToolDefinition] | None) -> ToolCall | None:
        """Normalize Gemini's observed textual tool-call fallback, never arbitrary prose."""
        match = re.fullmatch(r"TOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)", text.strip(), re.DOTALL)
        if match is None:
            return None
        name, raw_arguments = match.groups()
        allowed_names = {tool.name for tool in tools or []}
        if name not in allowed_names:
            return None
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise InvalidProviderResponseError("Gemini returned malformed textual tool-call JSON") from exc
        if not isinstance(arguments, dict):
            raise InvalidProviderResponseError("Gemini returned non-object textual tool-call arguments")
        return ToolCall(name=name, arguments=arguments)

    @staticmethod
    def _to_tool_declarations(tools: list[ToolDefinition]) -> list[Any]:
        declarations: list[Any] = []
        for tool in tools:
            schema = tool.parameters
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=types.Schema(**schema),
                )
            )
        return declarations
