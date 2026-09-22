import pytest

from app.errors import LLMProviderError
from app.fake_provider import FakeLLMProvider
from app.llm import LLMResponse, ToolCall


def test_fake_provider_returns_responses_in_order() -> None:
    provider = FakeLLMProvider(
        [
            LLMResponse(text="first"),
            LLMResponse(text="second"),
        ]
    )

    assert provider.generate("one").text == "first"
    assert provider.generate("two").text == "second"


def test_fake_provider_records_provider_neutral_call_context() -> None:
    tool_call = ToolCall(name="weather", arguments={"location": "Ujjain"}, call_id="test-call-1")
    provider = FakeLLMProvider([LLMResponse(text="done")])

    provider.generate(
        "weather",
        tool_call=tool_call,
        tool_result={"temperature_c": 28},
    )

    assert provider.calls[0].prompt == "weather"
    assert provider.calls[0].tool_call == tool_call
    assert provider.calls[0].tool_result == {"temperature_c": 28}


def test_fake_provider_fails_clearly_when_responses_are_exhausted() -> None:
    provider = FakeLLMProvider([])

    with pytest.raises(LLMProviderError, match="no response configured"):
        provider.generate("unexpected")
