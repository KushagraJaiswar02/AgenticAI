from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.gemini_provider import GeminiProvider


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
    assert models.last_kwargs == {"model": "test-model", "contents": "hello"}


def test_gemini_failure_is_structured() -> None:
    with pytest.raises(LLMProviderError, match="Gemini request failed"):
        provider(FakeModels(error=RuntimeError("quota exhausted"))).generate("hello")


def test_gemini_malformed_response_is_rejected() -> None:
    with pytest.raises(InvalidProviderResponseError, match="invalid response"):
        provider(FakeModels(SimpleNamespace(text=None))).generate("hello")
