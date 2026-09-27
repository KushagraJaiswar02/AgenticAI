"""Core request orchestration for JARVIS V0.2."""

from __future__ import annotations

import logging

from app.errors import ApplicationError, LLMProviderError, UnexpectedApplicationError
from app.input import normalize_input
from app.llm import LLMProvider, ToolCall
from app.local_router import LocalIntentRouter
from app.tools import ToolExecutionResult, ToolRegistry


class Orchestrator:
    """Coordinate normalized text input with a provider-neutral LLM and tools."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        logger: logging.Logger | None = None,
        local_router: LocalIntentRouter | None = None,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry or ToolRegistry()
        self._logger = logger or logging.getLogger("jarvis")
        self._local_router = local_router or LocalIntentRouter()

    def process(self, user_message: str) -> str:
        normalized = normalize_input(user_message)
        if not normalized:
            raise ApplicationError("Message must not be empty")
        try:
            try:
                response = self._llm.generate(normalized, tools=self._tool_registry.definitions())
            except LLMProviderError as exc:
                fallback_call = self._local_router.route(normalized)
                if fallback_call is None:
                    self._logger.warning(
                        "LLM provider unavailable and no local capability matched"
                    )
                    self._logger.debug("LLM provider failure details", exc_info=True)
                    raise
                self._logger.warning(
                    "LLM provider unavailable; attempting local fallback for '%s': %s",
                    fallback_call.name,
                    exc,
                )
                self._logger.debug("LLM provider failure details", exc_info=True)
                return self._execute_local_fallback(fallback_call)

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

    def _execute_local_fallback(self, tool_call: ToolCall) -> str:
        tool_result = self._tool_registry.execute(tool_call.name, tool_call.arguments)
        if not tool_result.success:
            return self._tool_failure_message(tool_call.name, tool_result.error)
        return self._tool_success_summary(tool_result)

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
        if "time" in tool_result.data:
            return f"The current local time is {tool_result.data['time']}."
        if "date" in tool_result.data:
            return f"Today is {tool_result.data['date']} ({tool_result.data.get('day', 'local time')})."
        if "result" in tool_result.data:
            return f"The result is {tool_result.data['result']}."
        if "python_version" in tool_result.data:
            return (
                f"You are running {tool_result.data.get('os', 'an unknown OS')} "
                f"with Python {tool_result.data['python_version']}."
            )
        if tool_result.data.get("status") == "launched":
            return f"I opened {tool_result.data.get('application', 'the application')}."
        return "I completed the request."
