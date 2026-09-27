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


def test_factory_includes_every_configured_provider_in_declared_order() -> None:
    order = ("gemini", "openai", "groq", "cerebras", "openrouter", "mistral", "cohere", "ollama")
    settings = Settings(
        llm_provider="gemini",
        llm_provider_order=order,
        gemini_api_key="gemini-key",
        gemini_model="gemini-model",
        openai_api_key="openai-key",
        openai_model="openai-model",
        groq_api_key=None,
        groq_model="groq-model",
        cerebras_api_key="cerebras-key",
        cerebras_model="cerebras-model",
        openrouter_api_key="openrouter-key",
        openrouter_model="openrouter-model",
        mistral_api_key="mistral-key",
        mistral_model="mistral-model",
        cohere_api_key="cohere-key",
        cohere_model="cohere-model",
    )
    constructors = {
        name: (lambda provider_name: lambda _: StubProvider(
            LLMResponse(text=provider_name)
        ))(name)
        for name in order
    }

    manager = create_provider_manager(settings, constructors=constructors)

    assert [name for name, _ in manager._providers] == [
        "gemini",
        "openai",
        "cerebras",
        "openrouter",
        "mistral",
        "cohere",
        "ollama",
    ]


def test_configured_provider_construction_failure_is_not_silent() -> None:
    settings = Settings(
        llm_provider="cerebras",
        llm_provider_order=("cerebras", "ollama"),
        cerebras_api_key="key",
        cerebras_model="model",
    )

    def broken_constructor(_):
        raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        create_provider_manager(
            settings,
            constructors={
                "cerebras": broken_constructor,
                "ollama": lambda _: StubProvider(LLMResponse(text="local")),
            },
        )
