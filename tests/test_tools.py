from __future__ import annotations

import pytest

from app.errors import ApplicationError
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.tools import ToolExecutionError, ToolRegistry, UnknownToolError, WeatherTool


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="I will check the weather for Paris.",
                tool_call=ToolCall(name="weather", arguments={"location": "Paris"}),
            )
        return LLMResponse(text="The current temperature in Paris is 18°C.")


def test_tool_registry_registers_and_looks_up_tools() -> None:
    registry = ToolRegistry()
    registry.register(WeatherTool())
    assert registry.get("weather") is not None
    assert registry.list() == ["weather"]


def test_tool_registry_rejects_duplicates() -> None:
    registry = ToolRegistry([WeatherTool()])
    with pytest.raises(ApplicationError, match="already registered"):
        registry.register(WeatherTool())


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(UnknownToolError, match="not registered"):
        ToolRegistry().execute("missing", {})


def test_weather_tool_returns_normalized_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url, params=None, timeout=None):
        if "geocoding-api" in url:
            return FakeResponse({"results": [{"latitude": 48.8566, "longitude": 2.3522}]})
        return FakeResponse(
            {
                "current": {
                    "temperature_2m": 18.5,
                    "apparent_temperature": 17.0,
                    "relative_humidity_2m": 68,
                    "wind_speed_10m": 12.2,
                    "weather_code": 1,
                }
            }
        )

    monkeypatch.setattr("app.tools.httpx.get", fake_get)

    result = WeatherTool().execute({"location": "Paris"})
    assert result.success is True
    assert result.data["location"] == "Paris"
    assert result.data["temperature_c"] == 18.5


def test_weather_tool_handles_external_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url, params=None, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.tools.httpx.get", fake_get)

    with pytest.raises(ToolExecutionError, match="Weather service unavailable"):
        WeatherTool().execute({"location": "Paris"})


def test_orchestrator_executes_tool_and_reformats_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url, params=None, timeout=None):
        if "geocoding-api" in url:
            return FakeResponse({"results": [{"latitude": 48.8566, "longitude": 2.3522}]})
        return FakeResponse(
            {
                "current": {
                    "temperature_2m": 18.5,
                    "apparent_temperature": 17.0,
                    "relative_humidity_2m": 68,
                    "wind_speed_10m": 12.2,
                    "weather_code": 1,
                }
            }
        )

    monkeypatch.setattr("app.tools.httpx.get", fake_get)

    llm = FakeLLM()
    orchestrator = Orchestrator(llm, tool_registry=ToolRegistry([WeatherTool()]))
    result = orchestrator.process("what is the weather in Paris")

    assert result == "The current temperature in Paris is 18°C."
    assert llm.calls == 2
