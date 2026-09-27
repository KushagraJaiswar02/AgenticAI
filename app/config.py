"""Runtime configuration for JARVIS V0.1."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


SUPPORTED_PROVIDERS = (
    "gemini",
    "openai",
    "groq",
    "cerebras",
    "openrouter",
    "mistral",
    "cohere",
    "ollama",
)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    llm_provider_order: tuple[str, ...] = (
        "gemini", "openai", "groq", "cerebras", "openrouter", "mistral", "cohere", "ollama"
    )
    llm_request_timeout: float = 30.0
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout: float = 30.0
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-luna"
    openai_timeout: float = 30.0
    groq_api_key: str | None = None
    groq_model: str = ""
    groq_timeout: float = 30.0
    cerebras_api_key: str | None = None
    cerebras_model: str = ""
    cerebras_timeout: float = 30.0
    openrouter_api_key: str | None = None
    openrouter_model: str = ""
    openrouter_timeout: float = 30.0
    cohere_api_key: str | None = None
    cohere_model: str = ""
    cohere_timeout: float = 30.0
    mistral_api_key: str | None = None
    mistral_model: str = ""
    mistral_timeout: float = 30.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_simple_model: str = "llama3.2:3b"
    ollama_complex_model: str = "qwen3:4b"
    ollama_model: str | None = None
    ollama_simple_timeout: float = 20.0
    ollama_complex_timeout: float = 120.0
    ollama_request_timeout: float = 120.0
    database_url: str = "sqlite:///jarvis.db"


def load_settings(*, dotenv_path: str | None = None) -> Settings:
    """Load settings from the environment without exposing secret values."""
    load_dotenv(dotenv_path=dotenv_path)
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    order_value = os.getenv(
        "LLM_PROVIDER_ORDER",
        "gemini,openai,groq,cerebras,openrouter,mistral,cohere,ollama",
    )
    provider_order = tuple(item.strip().lower() for item in order_value.split(",") if item.strip())
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"LLM_PROVIDER must be one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    if not provider_order:
        raise ConfigurationError("LLM_PROVIDER_ORDER must contain at least one provider")

    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
    cloud_values = {
        "GEMINI_MODEL": gemini_model,
        "OPENAI_MODEL": openai_model,
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "").strip(),
        "CEREBRAS_MODEL": os.getenv("CEREBRAS_MODEL", "").strip(),
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL", "").strip(),
        "COHERE_MODEL": os.getenv("COHERE_MODEL", "").strip(),
        "MISTRAL_MODEL": os.getenv("MISTRAL_MODEL", "").strip(),
    }
    simple_timeout_value = os.getenv("OLLAMA_SIMPLE_TIMEOUT", "20")
    complex_timeout_value = os.getenv("OLLAMA_COMPLEX_TIMEOUT", "120")
    try:
        request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
        cloud_timeouts = {
            name: float(os.getenv(name, "30"))
            for name in (
                "GEMINI_TIMEOUT", "OPENAI_TIMEOUT", "GROQ_TIMEOUT",
                "CEREBRAS_TIMEOUT", "OPENROUTER_TIMEOUT",
                "COHERE_TIMEOUT", "MISTRAL_TIMEOUT",
            )
        }
        ollama_simple_timeout = float(simple_timeout_value)
        ollama_complex_timeout = float(complex_timeout_value)
    except ValueError as exc:
        raise ConfigurationError("Provider timeouts must be numeric") from exc
    if any(value <= 0 for value in (request_timeout, ollama_simple_timeout, ollama_complex_timeout, *cloud_timeouts.values())):
        raise ConfigurationError("Provider timeouts must be positive")

    return Settings(
        llm_provider=provider,
        llm_provider_order=provider_order,
        llm_request_timeout=request_timeout,
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        gemini_timeout=cloud_timeouts["GEMINI_TIMEOUT"],
        openai_api_key=openai_key,
        openai_model=openai_model,
        openai_timeout=cloud_timeouts["OPENAI_TIMEOUT"],
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=cloud_values["GROQ_MODEL"],
        groq_timeout=cloud_timeouts["GROQ_TIMEOUT"],
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY"),
        cerebras_model=cloud_values["CEREBRAS_MODEL"],
        cerebras_timeout=cloud_timeouts["CEREBRAS_TIMEOUT"],
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=cloud_values["OPENROUTER_MODEL"],
        openrouter_timeout=cloud_timeouts["OPENROUTER_TIMEOUT"],
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        cohere_model=cloud_values["COHERE_MODEL"],
        cohere_timeout=cloud_timeouts["COHERE_TIMEOUT"],
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        mistral_model=cloud_values["MISTRAL_MODEL"],
        mistral_timeout=cloud_timeouts["MISTRAL_TIMEOUT"],
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_simple_model=os.getenv("OLLAMA_SIMPLE_MODEL", "llama3.2:3b").strip(),
        ollama_complex_model=os.getenv("OLLAMA_COMPLEX_MODEL", "qwen3:4b").strip(),
        ollama_simple_timeout=ollama_simple_timeout,
        ollama_complex_timeout=ollama_complex_timeout,
        ollama_request_timeout=ollama_complex_timeout,
    )
