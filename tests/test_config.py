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


def test_missing_api_key_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is required"):
        load_settings(dotenv_path=str(Path("missing.env")))


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
