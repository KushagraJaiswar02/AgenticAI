"""Core request orchestration for JARVIS V0.2."""

from __future__ import annotations

import json
import logging

from app.errors import ApplicationError, UnexpectedApplicationError
from app.input import normalize_input
from app.llm import LLMProvider, ToolCall
from app.tools import ToolExecutionResult, ToolRegistry


class Orchestrator:
    """Coordinate normalized text input with a provider-neutral LLM and tools."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry or ToolRegistry()
        self._logger = logger or logging.getLogger("jarvis")

    def process(self, user_message: str) -> str:
        normalized = normalize_input(user_message)
        if not normalized:
            raise ApplicationError("Message must not be empty")
        try:
            response = self._llm.generate(normalized, tools=self._tool_registry.definitions())
            if response.tool_call is None:
                return response.text

            tool_call = response.tool_call
            tool_result = self._tool_registry.execute(tool_call.name, tool_call.arguments)
            if not tool_result.success:
                return self._tool_failure_message(tool_call.name, tool_result.error)

            follow_up = self._llm.generate(
                normalized,
                tools=self._tool_registry.definitions(),
                tool_call=tool_call,
                tool_result=tool_result.data or {},
            )
            return follow_up.text.strip() or self._tool_success_summary(tool_result)
        except ApplicationError:
            raise
        except Exception as exc:
            self._logger.exception("Unexpected orchestration failure")
            raise UnexpectedApplicationError("Unable to process the request") from exc

    def _tool_failure_message(self, tool_name: str, error: str | None) -> str:
        if error:
            return f"I couldn't complete the {tool_name} request: {error}"
        return f"I couldn't complete the {tool_name} request right now."

    def _tool_success_summary(self, tool_result: ToolExecutionResult) -> str:
        if tool_result.data is None:
            return "I completed the request."
        if "temperature_c" in tool_result.data:
            temperature = tool_result.data.get("temperature_c")
            location = tool_result.data.get("location", "that location")
            if temperature is not None:
                return f"The current temperature in {location} is {temperature}°C."
        return "I completed the request."
