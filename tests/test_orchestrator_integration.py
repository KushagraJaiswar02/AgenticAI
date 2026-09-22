from __future__ import annotations

import pytest

from app.fake_provider import FakeLLMProvider
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.tools import ToolError, ToolExecutionResult, ToolRegistry, WeatherTool


class RecordingWeatherTool(WeatherTool):
    def __init__(self, result: ToolExecutionResult | None = None) -> None:
        self.arguments_seen: list[object] = []
        self.result = result

    def execute(self, arguments):
        self.arguments_seen.append(arguments)
        if self.result is not None:
            return self.result
        return ToolExecutionResult(
            success=True,
            data={"location": arguments.location, "temperature_c": 28},
        )


def test_normal_conversation_does_not_execute_tools() -> None:
    provider = FakeLLMProvider([LLMResponse(text="Hello from the fake provider.")])
    tool = RecordingWeatherTool()

    result = Orchestrator(provider, ToolRegistry([tool])).process("hello")

    assert result == "Hello from the fake provider."
    assert tool.arguments_seen == []
    assert len(provider.calls) == 1


def test_successful_tool_call_uses_real_orchestrator_and_registry() -> None:
    provider = FakeLLMProvider(
        [
            LLMResponse(
                tool_call=ToolCall(
                    name="weather",
                    arguments={"location": "Ujjain"},
                    call_id="test-call-1",
                )
            ),
            LLMResponse(text="The weather in Ujjain is currently 28°C."),
        ]
    )
    tool = RecordingWeatherTool()

    result = Orchestrator(provider, ToolRegistry([tool])).process("tell me ujjain's weather")

    assert result == "The weather in Ujjain is currently 28°C."
    assert len(tool.arguments_seen) == 1
    assert tool.arguments_seen[0].location == "Ujjain"
    assert len(provider.calls) == 2
    assert provider.calls[1].tool_call == ToolCall(
        name="weather",
        arguments={"location": "Ujjain"},
        call_id="test-call-1",
    )
    assert provider.calls[1].tool_result == {"location": "Ujjain", "temperature_c": 28}


def test_unknown_tool_uses_existing_error_path() -> None:
    provider = FakeLLMProvider(
        [LLMResponse(tool_call=ToolCall(name="missing", arguments={}))]
    )

    with pytest.raises(ToolError, match="not registered"):
        Orchestrator(provider, ToolRegistry()).process("do the missing thing")

    assert len(provider.calls) == 1


def test_tool_failure_returns_message_without_second_llm_call() -> None:
    provider = FakeLLMProvider(
        [LLMResponse(tool_call=ToolCall(name="weather", arguments={"location": "Ujjain"}))]
    )
    tool = RecordingWeatherTool(
        ToolExecutionResult(success=False, error="weather service unavailable")
    )

    result = Orchestrator(provider, ToolRegistry([tool])).process("check weather")

    assert result == "I couldn't complete the weather request: weather service unavailable"
    assert len(tool.arguments_seen) == 1
    assert len(provider.calls) == 1


def test_empty_final_model_response_keeps_fallback_behavior() -> None:
    provider = FakeLLMProvider(
        [
            LLMResponse(tool_call=ToolCall(name="weather", arguments={"location": "Ujjain"})),
            LLMResponse(text=""),
        ]
    )
    tool = RecordingWeatherTool()

    result = Orchestrator(provider, ToolRegistry([tool])).process("check weather")

    assert result == "The current temperature in Ujjain is 28°C."


def test_sequential_interactions_use_independent_fake_responses() -> None:
    provider = FakeLLMProvider(
        [
            LLMResponse(text="first answer"),
            LLMResponse(text="second answer"),
        ]
    )
    orchestrator = Orchestrator(provider, ToolRegistry())

    assert orchestrator.process("first") == "first answer"
    assert orchestrator.process("second") == "second answer"
    assert [call.prompt for call in provider.calls] == ["first", "second"]


def test_weather_tool_network_is_not_needed_for_orchestration() -> None:
    provider = FakeLLMProvider(
        [
            LLMResponse(tool_call=ToolCall(name="weather", arguments={"location": "Ujjain"})),
            LLMResponse(text="Weather received."),
        ]
    )
    tool = RecordingWeatherTool()

    assert Orchestrator(provider, ToolRegistry([tool])).process("weather") == "Weather received."
