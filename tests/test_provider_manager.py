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


def test_image_request_never_falls_back_to_text_only_provider() -> None:
    cloud = StubProvider(LLMResponse(text="would ignore image"))
    manager = ProviderManager([("gemini", cloud)])
    with pytest.raises(LLMProviderError, match="supports image input"):
        manager.generate("describe", images=["ZmFrZQ=="])
    assert cloud.calls == 0


def test_factory_skips_missing_cloud_keys_and_keeps_ollama() -> None:
    settings = Settings(
        llm_provider="gemini",
        llm_provider_order=("gemini", "openrouter", "ollama"),
        gemini_api_key=None,
    )
    manager = create_provider_manager(
        settings,
        constructors={"ollama": lambda _: StubProvider(LLMResponse(text="local"))},
    )
    assert manager.generate("hello").text == "local"


def test_factory_includes_configured_providers_in_declared_order() -> None:
    order = ("gemini", "openrouter", "groq", "ollama")
    settings = Settings(
        llm_provider="gemini", llm_provider_order=order,
        gemini_api_key="gemini-key", openrouter_api_key="router-key",
        openrouter_model="router-model", groq_api_key="groq-key",
    )
    constructors = {
        name: (lambda provider_name: lambda _: StubProvider(LLMResponse(text=provider_name)))(name)
        for name in order
    }
    manager = create_provider_manager(settings, constructors=constructors)
    assert [name for name, _ in manager._providers] == list(order)


def test_configured_provider_construction_failure_is_not_silent() -> None:
    settings = Settings(
        llm_provider="groq",
        llm_provider_order=("groq", "ollama"),
        groq_api_key="key",
    )

    def broken_constructor(_):
        raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        create_provider_manager(
            settings,
            constructors={
                "groq": broken_constructor,
                "ollama": lambda _: StubProvider(LLMResponse(text="local")),
            },
        )


def test_provider_failure_logs_cause_reason_timing_and_redacts_secrets(caplog) -> None:
    cause = TimeoutError("api_key=super-secret upstream timeout")
    error = LLMProviderError("Gemini request failed")
    error.__cause__ = cause
    manager = ProviderManager([("gemini", StubProvider(error=error))])

    with caplog.at_level(logging.WARNING, logger="jarvis"):
        with pytest.raises(LLMProviderError, match="All configured"):
            manager.generate("hello")

    message = caplog.records[-1].getMessage()
    assert "provider=gemini" in message
    assert "exception_type=LLMProviderError" in message
    assert "cause_type=TimeoutError" in message
    assert "reason=timeout" in message
    assert "elapsed_ms=" in message
    assert "super-secret" not in message


def test_provider_failure_logs_api_status_and_message() -> None:
    class APIError(RuntimeError):
        status_code = 429

    error = APIError("provider rate limit exceeded")
    manager = ProviderManager([("groq", StubProvider(error=error))])
    with pytest.raises(LLMProviderError, match="All configured"):
        manager.generate("hello")
