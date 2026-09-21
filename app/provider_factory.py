"""Construction of the configured provider."""

from __future__ import annotations

from app.config import ConfigurationError, Settings
from app.gemini_provider import GeminiProvider
from app.llm import LLMProvider
from app.openai_provider import OpenAIProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the configured provider without leaking provider choice to callers."""
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    raise ConfigurationError("Unsupported LLM provider")
