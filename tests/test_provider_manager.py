import logging

import pytest

from app.config import Settings
from app.errors import LLMProviderError
from app.llm import LLMProvider, LLMResponse
from app.provider_manager import ProviderManager, create_provider_manager


class StubProvider(LLMProvider):
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def generate(self, prompt, tools=None, *, tool_call=None, tool_result=None, think=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def test_manager_stops_after_first_success() -> None:
    first = StubProvider(LLMResponse(text="ok"))
    second = StubProvider(LLMResponse(text="unused"))
    result = ProviderManager([("first", first), ("second", second)]).generate("hello")
    assert result.text == "ok"
    assert first.calls == 1
    assert second.calls == 0


def test_manager_fails_over_in_order() -> None:
    first = StubProvider(error=LLMProviderError("down"))
    second = StubProvider(error=TimeoutError("timeout"))
    third = StubProvider(LLMResponse(text="ok"))
    result = ProviderManager([("first", first), ("second", second), ("third", third)]).generate("hello")
    assert result.text == "ok"
    assert [first.calls, second.calls, third.calls] == [1, 1, 1]


def test_manager_returns_controlled_error_when_all_fail() -> None:
    manager = ProviderManager([("first", StubProvider(error=LLMProviderError("down")))])
    with pytest.raises(LLMProviderError, match="All configured"):
        manager.generate("hello")


def test_factory_skips_missing_cloud_keys_and_keeps_ollama() -> None:
    settings = Settings(
        llm_provider="gemini",
        llm_provider_order=("gemini", "openai", "ollama"),
        gemini_api_key=None,
        openai_api_key=None,
    )
    manager = create_provider_manager(
        settings,
        constructors={"ollama": lambda _: StubProvider(LLMResponse(text="local"))},
    )
    assert manager.generate("hello").text == "local"
