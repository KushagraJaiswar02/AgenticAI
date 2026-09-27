from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ToolDefinition
from app.ollama_provider import OllamaProvider


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def settings() -> Settings:
    return Settings(
        ollama_base_url="http://ollama",
        ollama_simple_model="llama3.2:3b",
        ollama_complex_model="qwen3:4b",
        ollama_simple_timeout=20,
        ollama_complex_timeout=120,
    )


def test_ollama_simple_request_sets_think_false() -> None:
    client = FakeClient(FakeResponse({"message": {"content": "hello"}}))
    result = OllamaProvider(settings(), client=client).generate("what time is it")
    assert result.text == "hello"
    assert client.calls[0][1]["json"]["model"] == "llama3.2:3b"
    assert "think" not in client.calls[0][1]["json"]


def test_ollama_complex_request_sets_think_true() -> None:
    client = FakeClient(FakeResponse({"message": {"content": "analysis"}}))
    OllamaProvider(settings(), client=client).generate("analyze this architecture")
    assert client.calls[0][1]["json"]["model"] == "qwen3:4b"
    assert client.calls[0][1]["json"]["think"] is True


def test_ollama_normalizes_tool_call() -> None:
    client = FakeClient(
        FakeResponse(
            {"message": {"tool_calls": [{"id": "call-1", "function": {
                "name": "weather", "arguments": {"location": "Indore"}
            }}]}}
        )
    )
    result = OllamaProvider(settings(), client=client).generate(
        "weather in Indore",
        tools=[ToolDefinition("weather", "Weather", {"type": "object"})],
    )
    assert result.tool_call.name == "weather"
    assert result.tool_call.call_id == "call-1"
    payload = client.calls[0][1]["json"]
    assert payload["model"] == "llama3.2:3b"
    assert "Tools are optional capabilities" in payload["messages"][0]["content"]


def test_ollama_tool_continuation_reuses_selected_model() -> None:
    client = FakeClient(
        FakeResponse({"message": {"tool_calls": [{
            "id": "call-1",
            "function": {"name": "weather", "arguments": {"location": "Indore"}},
        }]}})
    )
    provider = OllamaProvider(settings(), client=client)
    call = provider.generate("what is the weather in Indore").tool_call
    client.response = FakeResponse({"message": {"content": "It is warm."}})

    result = provider.generate(
        "what is the weather in Indore",
        tool_call=call,
        tool_result={"temperature_c": 28},
    )

    assert result.text == "It is warm."
    assert client.calls[0][1]["json"]["model"] == "llama3.2:3b"
    assert client.calls[1][1]["json"]["model"] == "llama3.2:3b"
    assert client.calls[1][1]["json"]["messages"][-1]["role"] == "tool"


def test_ollama_failure_is_structured() -> None:
    client = FakeClient(FakeResponse({}, error=TimeoutError("timed out")))
    with pytest.raises(LLMProviderError, match="Ollama request failed"):
        OllamaProvider(settings(), client=client).generate("hello")


def test_ollama_malformed_response_is_rejected() -> None:
    client = FakeClient(FakeResponse({"message": {}}))
    with pytest.raises(InvalidProviderResponseError):
        OllamaProvider(settings(), client=client).generate("hello")
