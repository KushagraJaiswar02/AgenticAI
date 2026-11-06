from types import SimpleNamespace

import pytest
from copy import deepcopy
from threading import Lock, Thread
from time import sleep

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
        self.responses = list(response) if isinstance(response, list) else [response]
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, deepcopy(kwargs)))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


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
    assert payload["model"] == "qwen3:4b"
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
    client.responses = [FakeResponse({"message": {"content": "It is warm."}})]

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


def test_ollama_sets_keep_alive_and_context_window():
    configured = Settings(ollama_keep_alive="3m", ollama_context_window=2048)
    client = FakeClient(FakeResponse({"message": {"content": "ok"}}))
    OllamaProvider(configured, client=client).generate("hello")
    payload = client.calls[0][1]["json"]
    assert payload["keep_alive"] == "3m"
    assert payload["options"]["num_ctx"] == 2048


def test_ollama_routes_images_only_to_configured_vision_model():
    client = FakeClient(FakeResponse({"message": {"content": "a screenshot"}}))
    provider = OllamaProvider(settings(), client=client)
    provider.generate("describe the image", images=["ZmFrZQ=="])
    payload = client.calls[0][1]["json"]
    assert payload["model"] == "gemma3:4b"
    assert payload["messages"][-1]["images"] == ["ZmFrZQ=="]


def test_missing_text_model_uses_one_configured_local_fallback():
    class MissingModelError(RuntimeError):
        response = SimpleNamespace(status_code=404)

    client = FakeClient([
        FakeResponse({}, error=MissingModelError("not found")),
        FakeResponse({"message": {"content": "fallback answer"}}),
    ])
    result = OllamaProvider(settings(), client=client).generate("debug this code")
    assert result.text == "fallback answer"
    assert client.calls[0][1]["json"]["model"] == "qwen3:4b"
    assert client.calls[1][1]["json"]["model"] == "llama3.2:3b"


def test_missing_vision_model_does_not_fallback_to_text():
    class MissingModelError(RuntimeError):
        response = SimpleNamespace(status_code=404)

    client = FakeClient(FakeResponse({}, error=MissingModelError("not found")))
    with pytest.raises(LLMProviderError, match="gemma3:4b"):
        OllamaProvider(settings(), client=client).generate("describe image", images=["ZmFrZQ=="])
    assert len(client.calls) == 1


def test_ollama_unloads_previous_model_before_switching():
    client = FakeClient(FakeResponse({"message": {"content": "ok"}}))
    provider = OllamaProvider(settings(), client=client)
    provider.generate("hello")
    provider.generate("debug this code")
    assert client.calls[1][0].endswith("/api/generate")
    assert client.calls[1][1]["json"] == {"model": "llama3.2:3b", "keep_alive": 0}
    assert client.calls[2][1]["json"]["model"] == "qwen3:4b"


def test_ollama_serializes_concurrent_requests():
    class SlowClient:
        def __init__(self):
            self.active = 0
            self.maximum = 0
            self.guard = Lock()

        def post(self, url, **kwargs):
            with self.guard:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            sleep(0.02)
            with self.guard:
                self.active -= 1
            return FakeResponse({"message": {"content": "ok"}})

    client = SlowClient()
    provider = OllamaProvider(settings(), client=client)
    threads = [Thread(target=provider.generate, args=("hello",)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert client.maximum == 1
