from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.gemini_provider import GeminiProvider
from app.llm import ToolCall, ToolDefinition


class FakeModels:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
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
    part = SimpleNamespace(function_call=function_call)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    models = FakeModels(SimpleNamespace(text=None, candidates=[candidate]))

    result = provider(models).generate(
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


def test_gemini_malformed_function_call_is_rejected() -> None:
    function_call = SimpleNamespace(name="weather", args="location=Ujjain")
    part = SimpleNamespace(function_call=function_call)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    with pytest.raises(InvalidProviderResponseError, match="malformed function call"):
        provider(FakeModels(SimpleNamespace(text=None, candidates=[candidate]))).generate("hello")


def test_gemini_tool_result_uses_native_content_sequence() -> None:
    models = FakeModels(SimpleNamespace(text="It is sunny in Ujjain."))
    llm = provider(models)
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
    assert contents[1].role == "model"
    assert contents[2].role == "user"


def test_gemini_failure_is_structured() -> None:
    with pytest.raises(LLMProviderError, match="Gemini request failed"):
        provider(FakeModels(error=RuntimeError("quota exhausted"))).generate("hello")


def test_gemini_malformed_response_is_rejected() -> None:
    with pytest.raises(InvalidProviderResponseError, match="invalid response"):
        provider(FakeModels(SimpleNamespace(text=None))).generate("hello")
