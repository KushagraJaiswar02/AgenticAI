import pytest

from app.errors import LLMProviderError
from app.errors import UnexpectedApplicationError
from app.llm import LLMProvider
from app.orchestrator import Orchestrator
from app.tools import ToolExecutionResult, ToolRegistry, WeatherTool


class FailingProvider(LLMProvider):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def generate(self, prompt, tools=None, *, tool_call=None, tool_result=None):
        self.calls += 1
        raise self.error


class RecordingWeatherTool(WeatherTool):
    def __init__(self) -> None:
        self.locations: list[str] = []

    def execute(self, arguments):
        self.locations.append(arguments.location)
        return ToolExecutionResult(
            success=True,
            data={"location": arguments.location, "temperature_c": 28},
        )


def test_llm_provider_failure_uses_local_weather_fallback() -> None:
    provider = FailingProvider(LLMProviderError("provider unavailable"))
    weather = RecordingWeatherTool()

    result = Orchestrator(provider, ToolRegistry([weather])).process(
        "tell me Ujjain's weather"
    )

    assert result == "The current temperature in Ujjain is 28°C."
    assert weather.locations == ["Ujjain"]
    assert provider.calls == 1


def test_unrecognized_local_intent_reraises_provider_error() -> None:
    provider = FailingProvider(LLMProviderError("provider unavailable"))

    with pytest.raises(LLMProviderError, match="provider unavailable"):
        Orchestrator(provider).process("tell me a joke")


def test_non_provider_errors_do_not_trigger_local_fallback() -> None:
    provider = FailingProvider(RuntimeError("unexpected failure"))

    with pytest.raises(UnexpectedApplicationError, match="Unable to process"):
        Orchestrator(provider).process("weather in Ujjain")


def test_local_fallback_tool_failure_returns_safe_message() -> None:
    provider = FailingProvider(LLMProviderError("provider unavailable"))

    class FailedWeatherTool(WeatherTool):
        def execute(self, arguments):
            return ToolExecutionResult(success=False, error="weather unavailable")

    result = Orchestrator(
        provider,
        ToolRegistry([FailedWeatherTool()]),
    ).process("weather in Ujjain")

    assert result == "I couldn't complete the weather request: weather unavailable"
