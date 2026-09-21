"""Core request orchestration for JARVIS V0.2."""

from __future__ import annotations

import json
import logging

from app.errors import ApplicationError, UnexpectedApplicationError
from app.input import normalize_input
from app.llm import LLMProvider
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
            response = self._llm.generate(normalized)
            if response.tool_call is None:
                return response.text

            tool_result = self._tool_registry.execute(response.tool_call.name, response.tool_call.arguments)
            if not tool_result.success:
                return self._tool_failure_message(response.tool_call.name, tool_result.error)

            final_prompt = (
                f"User request: {normalized}\n"
                f"Tool result: {json.dumps(tool_result.data, ensure_ascii=False)}\n"
                "Return a short, natural-language answer based on the tool result."
            )
            follow_up = self._llm.generate(final_prompt)
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
