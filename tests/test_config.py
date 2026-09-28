from pathlib import Path

import pytest

from app.config import ConfigurationError, load_settings


def test_configuration_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    settings = load_settings(dotenv_path=str(Path("missing.env")))
    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "test-model"


def test_missing_api_key_is_deferred_to_provider_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings(dotenv_path=str(Path("missing.env")))
    assert settings.gemini_api_key is None


def test_default_model_is_configurable_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert load_settings(dotenv_path=str(Path("missing.env"))).gemini_model == "gemini-3.8-flash"


def test_openai_configuration_remains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    settings = load_settings(dotenv_path=str(Path("missing.env")))
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "test-model"


def test_workspace_configuration_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("JARVIS_WORKSPACE", str(workspace))
    monkeypatch.setenv("JARVIS_ALLOWED_ROOTS", str(tmp_path))
    settings = load_settings(dotenv_path=str(Path("missing.env")))
    assert settings.jarvis_workspace == str(workspace)
    assert settings.filesystem_allowed_roots == (str(tmp_path),)


def test_ollama_models_and_timeouts_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_SIMPLE_MODEL", "fast-model")
    monkeypatch.setenv("OLLAMA_COMPLEX_MODEL", "deep-model")
    monkeypatch.setenv("OLLAMA_SIMPLE_TIMEOUT", "7")
    monkeypatch.setenv("OLLAMA_COMPLEX_TIMEOUT", "70")

    settings = load_settings(dotenv_path=str(Path("missing.env")))

    assert settings.ollama_simple_model == "fast-model"
    assert settings.ollama_complex_model == "deep-model"
    assert settings.ollama_simple_timeout == 7
    assert settings.ollama_complex_timeout == 70


@pytest.mark.parametrize("provider", ["groq", "cerebras", "openrouter", "mistral", "cohere"])
def test_all_cloud_providers_are_valid_preferred_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert load_settings(dotenv_path=str(Path("missing.env"))).llm_provider == provider
