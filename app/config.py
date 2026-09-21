"""Runtime configuration for JARVIS V0.1."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-luna"
    database_url: str = "sqlite:///jarvis.db"


def load_settings(*, dotenv_path: str | None = None) -> Settings:
    """Load settings from the environment without exposing secret values."""
    load_dotenv(dotenv_path=dotenv_path)
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "openai"}:
        raise ConfigurationError("LLM_PROVIDER must be 'gemini' or 'openai'")

    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()

    if provider == "gemini" and (not gemini_key or not gemini_key.strip()):
        raise ConfigurationError("GEMINI_API_KEY is required for Gemini")
    if provider == "openai" and (not openai_key or not openai_key.strip()):
        raise ConfigurationError("OPENAI_API_KEY is required for OpenAI")
    if provider == "gemini" and not gemini_model:
        raise ConfigurationError("GEMINI_MODEL must not be empty")
    if provider == "openai" and not openai_model:
        raise ConfigurationError("OPENAI_MODEL must not be empty")

    return Settings(
        llm_provider=provider,
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        openai_api_key=openai_key,
        openai_model=openai_model,
    )
