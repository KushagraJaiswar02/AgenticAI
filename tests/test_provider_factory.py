from types import SimpleNamespace

from app.config import Settings
from app.gemini_provider import GeminiProvider
from app.provider_factory import create_llm_provider


def test_factory_selects_gemini(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.provider_factory.GeminiProvider",
        lambda settings: SimpleNamespace(provider="gemini"),
    )
    provider = create_llm_provider(Settings(llm_provider="gemini", gemini_api_key="key"))
    assert provider.provider == "gemini"


def test_factory_selects_groq(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.provider_factory.GroqProvider",
        lambda settings: SimpleNamespace(provider="groq"),
    )
    provider = create_llm_provider(Settings(llm_provider="groq", groq_api_key="key"))
    assert provider.provider == "groq"


def test_application_factory_path_constructs_configured_provider_chain() -> None:
    settings = Settings(
        llm_provider="gemini",
        llm_provider_order=("gemini", "openrouter", "groq", "ollama"),
        gemini_api_key="gemini-key",
        openrouter_api_key="router-key",
        openrouter_model="router-model",
        groq_api_key="groq-key",
    )
    provider = create_llm_provider(settings)
    assert [name for name, _ in provider._providers] == ["gemini", "openrouter", "groq", "ollama"]
