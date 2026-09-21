from types import SimpleNamespace

from app.config import Settings
from app.gemini_provider import GeminiProvider
from app.openai_provider import OpenAIProvider
from app.provider_factory import create_llm_provider


def test_factory_selects_gemini(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.provider_factory.GeminiProvider",
        lambda settings: SimpleNamespace(provider="gemini"),
    )
    provider = create_llm_provider(Settings(llm_provider="gemini", gemini_api_key="key"))
    assert provider.provider == "gemini"


def test_factory_selects_openai(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.provider_factory.OpenAIProvider",
        lambda settings: SimpleNamespace(provider="openai"),
    )
    provider = create_llm_provider(Settings(llm_provider="openai", openai_api_key="key"))
    assert provider.provider == "openai"
