"""Core request orchestration for JARVIS V0.2."""

from __future__ import annotations

import logging
import inspect
import uuid
from collections.abc import Callable

from app.errors import ApplicationError, LLMProviderError, UnexpectedApplicationError
from app.input import normalize_input
from app.llm import ConversationMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition
from app.local_router import LocalIntentRouter
from app.tools import ToolExecutionResult, ToolRegistry
from app.policy import ConfirmationRequiredError, PolicyDeniedError
from app.session import ConversationState, PendingConfirmation
from app.filesystem_context import AmbiguousFilesystemReferenceError, MissingFilesystemReferenceError


class Orchestrator:
    """Coordinate normalized text input with a provider-neutral LLM and tools."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        logger: logging.Logger | None = None,
        local_router: LocalIntentRouter | None = None,
        confirmation_callback: Callable[[str], str] | None = None,
        state: ConversationState | None = None,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry or ToolRegistry()
        self._logger = logger or logging.getLogger("jarvis")
        self._local_router = local_router or LocalIntentRouter()
        self._confirmation_callback = confirmation_callback
        self._state = state or ConversationState(filesystem_context=self._filesystem_context_from_registry())

    def process(self, user_message: str) -> str:
        normalized = normalize_input(user_message)
        if not normalized:
            raise ApplicationError("Message must not be empty")
        if self._state.pending_confirmation is not None and normalized.strip().lower() in {"y", "yes", "proceed", "no", "n"}:
            return self._resolve_pending_confirmation(normalized)
        request_id = str(uuid.uuid4())
        try:
            try:
                response = self._generate(normalized, tools=self._tool_registry.definitions(), conversation=self._state.recent_messages())
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
                return self._record_response(normalized, self._execute_local_fallback(fallback_call, request_id=request_id))

            if response.tool_call is None:
                return self._record_response(normalized, response.text)

            response = self._continue_tool_calls(response, normalized, request_id)
            return self._record_response(normalized, response)
        except ApplicationError:
            raise
        except Exception as exc:
            self._logger.exception("Unexpected orchestration failure")
            raise UnexpectedApplicationError("Unable to process the request") from exc

    def _filesystem_context_from_registry(self):
        for name in self._tool_registry.list():
            tool = self._tool_registry.get(name)
            context = getattr(tool, "context", None)
            if context is not None:
                return context
        return None

    def _continue_tool_calls(self, response, prompt: str, request_id: str) -> str:
        last_result: ToolExecutionResult | None = None
        for _ in range(8):
            if response.tool_call is None:
                text = response.text.strip()
                return text or (self._tool_success_summary(last_result) if last_result else "I completed the request.")
            tool_call = self._state.resolve_filesystem_intent(response.tool_call, prompt)
            self._state.record_tool_call(tool_call.name, tool_call.arguments)
            tool_result = self._execute_with_confirmation(tool_call.name, tool_call.arguments, request_id=request_id)
            self._state.record_tool_result(tool_call.name, self._tool_result_payload(tool_result))
            if not tool_result.success:
                return self._tool_failure_message(tool_call.name, tool_result.error)
            last_result = tool_result
            response = self._llm.generate(
                prompt,
                tools=self._tool_registry.definitions(),
                tool_call=tool_call,
                tool_result=self._tool_result_payload(tool_result),
            )
        return "I stopped after reaching the tool-step limit for this request."

    @staticmethod
    def _tool_result_payload(tool_result: ToolExecutionResult) -> dict:
        payload = dict(tool_result.data or {})
        if tool_result.tool_name not in {
            "create_file", "read_file", "write_file", "list_directory", "change_directory",
            "search_files", "create_directory", "copy_file", "move_file", "rename_file", "delete_file",
        }:
            return payload
        payload.update(tool_result.as_dict())
        return payload

    def _generate(
        self,
        prompt: str,
        *,
        tools: list[ToolDefinition] | None = None,
        conversation: list[ConversationMessage] | None = None,
    ) -> LLMResponse:
        """Pass history to providers while retaining compatibility with test/local adapters."""
        generate = self._llm.generate
        if "conversation" in inspect.signature(generate).parameters:
            return generate(prompt, tools=tools, conversation=conversation)
        return generate(prompt, tools=tools)

    def _record_response(self, user_message: str, response: str) -> str:
        self._state.record_turn(user_message, response)
        return response

    def _resolve_pending_confirmation(self, response: str) -> str:
        pending = self._state.pending_confirmation
        self._state.pending_confirmation = None
        if pending is None:
            return "There is no pending confirmation."
        if response.strip().lower() in {"no", "n"}:
            return "I did not execute the action."
        result = self._tool_registry.execute(
            pending.tool_name,
            pending.arguments,
            request_id=pending.request.action_id,
            confirmation_request=pending.request,
            confirmation_response=response,
            provider=type(self._llm).__name__,
        )
        if not result.success:
            return self._tool_failure_message(pending.tool_name, result.error)
        message = self._tool_success_summary(result)
        self._state.record_turn(response, message)
        return message

    def _execute_local_fallback(self, tool_call: ToolCall, *, request_id: str) -> str:
        tool_result = self._execute_with_confirmation(tool_call.name, tool_call.arguments, request_id=request_id)
        if not tool_result.success:
            return self._tool_failure_message(tool_call.name, tool_result.error)
        return self._tool_success_summary(tool_result)

    def _execute_with_confirmation(self, name: str, arguments: dict, *, request_id: str) -> ToolExecutionResult:
        try:
            return self._tool_registry.execute(name, arguments, request_id=request_id, provider=type(self._llm).__name__)
        except (AmbiguousFilesystemReferenceError, MissingFilesystemReferenceError) as exc:
            return ToolExecutionResult(success=False, error=str(exc))
        except ConfirmationRequiredError as exc:
            if self._confirmation_callback is None:
                self._state.pending_confirmation = PendingConfirmation(name, arguments, exc.request)
                return ToolExecutionResult(success=False, error=exc.request.prompt)
            response = self._confirmation_callback(exc.request.prompt)
            return self._tool_registry.execute(name, arguments, request_id=request_id, confirmation_response=response, confirmation_request=exc.request, provider=type(self._llm).__name__)
        except PolicyDeniedError:
            raise

    def _tool_failure_message(self, tool_name: str, error: str | None) -> str:
        if error:
            return f"I couldn't complete the {tool_name} request: {error}"
        return f"I couldn't complete the {tool_name} request right now."

    def _tool_success_summary(self, tool_result: ToolExecutionResult) -> str:
        if tool_result.data is None:
            return "I completed the request."
        if tool_result.data.get("status") == "created" and tool_result.data.get("path"):
            return f"I created {tool_result.data['path']}."
        if tool_result.data.get("status") == "written" and tool_result.data.get("path"):
            return f"I wrote the content to {tool_result.data['path']}."
        if "temperature_c" in tool_result.data:
            temperature = tool_result.data.get("temperature_c")
            location = tool_result.data.get("location", "that location")
            if temperature is not None:
                return f"The current temperature in {location} is {temperature}°C."
        if "time" in tool_result.data:
            return f"The current local time is {tool_result.data['time']}."
        if "date" in tool_result.data:
            return f"Today is {tool_result.data['date']} ({tool_result.data.get('day', 'local time')})."
        if "content" in tool_result.data:
            return f"I read {tool_result.data.get('path', 'the file')}:\n{tool_result.data['content']}"
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
