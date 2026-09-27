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


def test_application_factory_path_constructs_configured_provider_chain() -> None:
    settings = Settings(
        llm_provider="gemini",
        llm_provider_order=(
            "gemini",
            "openai",
            "groq",
            "cerebras",
            "openrouter",
            "mistral",
            "cohere",
            "ollama",
        ),
        gemini_api_key="gemini-key",
        openai_api_key="openai-key",
        groq_api_key=None,
        groq_model="groq-model",
        cerebras_api_key="cerebras-key",
        cerebras_model="cerebras-model",
        openrouter_api_key="openrouter-key",
        openrouter_model="openrouter-model",
        mistral_api_key="mistral-key",
        mistral_model="mistral-model",
        cohere_api_key="cohere-key",
        cohere_model="cohere-model",
    )

    provider = create_llm_provider(settings)

    assert [name for name, _ in provider._providers] == [
        "gemini",
        "openai",
        "cerebras",
        "openrouter",
        "mistral",
        "cohere",
        "ollama",
    ]
