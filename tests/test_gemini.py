from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.gemini_provider import GeminiProvider
from app.llm import ToolCall, ToolDefinition


class FakeModels:
    def __init__(self, result=None, error=None, results=None):
        self.result = result
        self.error = error
        self.results = list(results or [])
        self.last_kwargs = None
        self.calls = []

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.results:
            return self.results.pop(0)
        return self.result


def provider(models) -> GeminiProvider:
    client = SimpleNamespace(models=models)
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test-key",
        gemini_model="test-model",
    )
    return GeminiProvider(settings, client=client)


def test_gemini_provider_creation() -> None:
    assert provider(FakeModels(SimpleNamespace(text="ok")))._model == "test-model"


def test_gemini_response_is_normalized() -> None:
    models = FakeModels(SimpleNamespace(text=" hello "))
    result = provider(models).generate("hello")
    assert result.text == "hello"
    assert models.last_kwargs["model"] == "test-model"
    assert models.last_kwargs["contents"] == "hello"


def test_gemini_native_function_call_is_normalized() -> None:
    function_call = SimpleNamespace(name="weather", args={"location": "Ujjain"})
    part = SimpleNamespace(function_call=function_call, thought_signature=b"signature")
    model_content = SimpleNamespace(role="model", parts=[part])
    candidate = SimpleNamespace(content=model_content)
    models = FakeModels(SimpleNamespace(text=None, candidates=[candidate]))
    llm = provider(models)

    result = llm.generate(
        "what's the weather in ujjain?",
        tools=[
            ToolDefinition(
                name="weather",
                description="Get weather.",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            )
        ],
    )

    assert result.tool_call == ToolCall(name="weather", arguments={"location": "Ujjain"})
    assert result.text == ""
    assert models.last_kwargs["config"] is not None
    assert llm._pending_tool_context is model_content


def test_gemini_malformed_function_call_is_rejected() -> None:
    function_call = SimpleNamespace(name="weather", args="location=Ujjain")
    part = SimpleNamespace(function_call=function_call)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    with pytest.raises(InvalidProviderResponseError, match="malformed function call"):
        provider(FakeModels(SimpleNamespace(text=None, candidates=[candidate]))).generate("hello")


def test_gemini_tool_result_uses_native_content_sequence() -> None:
    function_call = SimpleNamespace(name="weather", args={"location": "Ujjain"})
    original_part = SimpleNamespace(function_call=function_call, thought_signature=b"signature")
    original_model_content = SimpleNamespace(role="model", parts=[original_part])
    first_response = SimpleNamespace(
        text=None,
        candidates=[SimpleNamespace(content=original_model_content)],
    )
    final_response = SimpleNamespace(text="It is sunny in Ujjain.")
    models = FakeModels(results=[first_response, final_response])
    llm = provider(models)
    initial = llm.generate(
        "what's the weather in ujjain?",
        tools=[ToolDefinition(name="weather", description="Get weather.", parameters={})],
    )
    assert initial.tool_call == ToolCall(name="weather", arguments={"location": "Ujjain"})
    assert llm._pending_tool_context is original_model_content

    result = llm.generate(
        "what's the weather in ujjain?",
        tool_call=ToolCall(name="weather", arguments={"location": "Ujjain"}),
        tool_result={"location": "Ujjain", "temperature_c": 27},
    )

    assert result.text == "It is sunny in Ujjain."
    contents = models.last_kwargs["contents"]
    assert isinstance(contents, list)
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1] is original_model_content
    assert contents[2].role == "user"
    assert contents[2].parts[0].function_response.name == "weather"
    assert llm._pending_tool_context is None


def test_gemini_pending_tool_context_is_not_reused_for_unrelated_request() -> None:
    function_call = SimpleNamespace(name="weather", args={"location": "Ujjain"})
    model_content = SimpleNamespace(
        role="model",
        parts=[SimpleNamespace(function_call=function_call, thought_signature=b"signature")],
    )
    first_response = SimpleNamespace(
        text=None,
        candidates=[SimpleNamespace(content=model_content)],
    )
    second_response = SimpleNamespace(text="Hello there.")
    models = FakeModels(results=[first_response, second_response])
    llm = provider(models)

    initial = llm.generate("weather in ujjain")
    assert initial.tool_call is not None
    unrelated = llm.generate("hello")

    assert unrelated.text == "Hello there."
    with pytest.raises(InvalidProviderResponseError, match="context is no longer available"):
        llm.generate(
            "weather in ujjain",
            tool_call=initial.tool_call,
            tool_result={"temperature_c": 27},
        )


def test_gemini_failure_is_structured() -> None:
    with pytest.raises(LLMProviderError, match="Gemini request failed"):
        provider(FakeModels(error=RuntimeError("quota exhausted"))).generate("hello")


def test_gemini_malformed_response_is_rejected() -> None:
    with pytest.raises(InvalidProviderResponseError, match="invalid response"):
        provider(FakeModels(SimpleNamespace(text=None))).generate("hello")
