"""Construction of the configured provider."""

from __future__ import annotations

from app.config import Settings
from app.gemini_provider import GeminiProvider
from app.groq_provider import GroqProvider
from app.cerebras_provider import CerebrasProvider
from app.openrouter_provider import OpenRouterProvider
from app.cohere_provider import CohereProvider
from app.mistral_provider import MistralProvider
from app.llm import LLMProvider
from app.ollama_provider import OllamaProvider
from app.openai_provider import OpenAIProvider
from app.provider_manager import create_provider_manager


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the request-scoped failover service."""
    return create_provider_manager(
        settings,
        constructors={
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "groq": GroqProvider,
            "cerebras": CerebrasProvider,
            "openrouter": OpenRouterProvider,
            "cohere": CohereProvider,
            "mistral": MistralProvider,
            "ollama": OllamaProvider,
        },
    )
