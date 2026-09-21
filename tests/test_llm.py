from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import LLMProvider, LLMResponse
from app.openai_provider import OpenAIProvider


class FakeLLM(LLMProvider):
    def generate(self, prompt: str, tools=None, *, tool_call=None, tool_result=None) -> LLMResponse:
        return LLMResponse(text=f"reply: {prompt}")


class FakeResponses:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def provider(responses) -> OpenAIProvider:
    client = SimpleNamespace(responses=responses)
    return OpenAIProvider(
        Settings(openai_api_key="test-key", openai_model="test-model"),
        client=client,
    )


def test_llm_interface_can_be_mocked() -> None:
    assert FakeLLM().generate("hello").text == "reply: hello"


def test_openai_response_is_normalized() -> None:
    responses = FakeResponses(SimpleNamespace(output_text=" hello "))
    result = provider(responses).generate("hello")
    assert result.text == "hello"
    assert responses.last_kwargs == {"model": "test-model", "input": "hello"}


def test_openai_failure_is_structured() -> None:
    with pytest.raises(LLMProviderError, match="OpenAI request failed"):
        provider(FakeResponses(error=RuntimeError("secret test-key"))).generate("hello")


def test_malformed_provider_response_is_rejected() -> None:
    with pytest.raises(InvalidProviderResponseError, match="invalid response"):
        provider(FakeResponses(SimpleNamespace(output_text=None))).generate("hello")
