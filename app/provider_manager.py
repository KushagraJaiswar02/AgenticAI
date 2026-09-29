"""Request-scoped failover across configured LLM providers."""

from __future__ import annotations

import logging
import inspect
import re
import time
from typing import Any, Callable

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ConversationMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition


class ProviderManager(LLMProvider):
    """Try configured providers in priority order for each request."""

    def __init__(
        self,
        providers: list[tuple[str, LLMProvider]],
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._providers = providers
        self._logger = logger or logging.getLogger("jarvis")
        self._active_provider: str | None = None
        self.provider = providers[0][0] if providers else None

    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
        think: bool | None = None,
        conversation: list[ConversationMessage] | None = None,
    ) -> LLMResponse:
        continuation = tool_call is not None and tool_result is not None
        candidates = self._ordered_candidates(continuation=continuation)
        failures: list[Exception] = []
        for name, provider in candidates:
            self._logger.info("LLM request starting with provider='%s'", name)
            started = time.perf_counter()
            try:
                kwargs = {
                    "tools": tools,
                    "tool_call": tool_call,
                    "tool_result": tool_result,
                    "think": think,
                }
                if "conversation" in inspect.signature(provider.generate).parameters:
                    kwargs["conversation"] = conversation
                response = provider.generate(prompt, **kwargs)
                self._active_provider = name if response.tool_call is not None else None
                self._logger.info("LLM provider '%s' succeeded", name)
                return response
            except (LLMProviderError, InvalidProviderResponseError) as exc:
                failures.append(exc)
                self._log_failure(name, exc, started)
                continue
            except Exception as exc:
                failures.append(exc)
                self._log_failure(name, exc, started)
                self._logger.debug("Provider failure details", exc_info=True)
                continue

        self._active_provider = None
        raise LLMProviderError("All configured LLM providers are currently unavailable") from (
            failures[-1] if failures else None
        )

    def _ordered_candidates(self, *, continuation: bool) -> list[tuple[str, LLMProvider]]:
        if continuation and self._active_provider is not None:
            return [
                (name, provider)
                for name, provider in self._providers
                if name == self._active_provider
            ]
        if not continuation:
            return self._providers
        return self._providers

    @staticmethod
    def _safe_reason(error: Exception) -> str:
        text = str(error).lower()
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "rate" in text or "429" in text:
            return "rate limited"
        if "auth" in text or "401" in text or "403" in text:
            return "authentication failed"
        if "invalid" in text or "malformed" in text or "empty" in text:
            return "invalid response"
        if "unavailable" in text or "connection" in text or "dns" in text:
            return "unavailable"
        return error.__class__.__name__

    @classmethod
    def _failure_reason(cls, error: Exception) -> str:
        reason = cls._safe_reason(error)
        cause = error.__cause__ or error.__context__
        if reason == error.__class__.__name__ and cause is not None:
            return cls._safe_reason(cause)
        return reason

    def _log_failure(self, name: str, error: Exception, started: float) -> None:
        cause = error.__cause__ or error.__context__
        status_code = self._status_code(error) or (self._status_code(cause) if cause else None)
        self._logger.warning(
            "provider=%s status=failed exception_type=%s message=%s cause_type=%s cause=%s reason=%s status_code=%s elapsed_ms=%.1f",
            name,
            error.__class__.__name__,
            self._safe_text(str(error)),
            cause.__class__.__name__ if cause else "none",
            self._safe_text(str(cause)) if cause else "none",
            self._failure_reason(error),
            status_code if status_code is not None else "none",
            (time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _status_code(error: Exception | None) -> int | str | None:
        if error is None:
            return None
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) or getattr(error, "status_code", None)

    @staticmethod
    def _safe_text(value: str) -> str:
        text = re.sub(r"(?i)(api[_ -]?key|authorization|token|password|secret)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", value)
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
        return text[:1000]


def create_provider_manager(
    settings: Settings,
    *,
    logger: logging.Logger | None = None,
    constructors: dict[str, Callable[[Settings], LLMProvider]] | None = None,
) -> ProviderManager:
    logger = logger or logging.getLogger("jarvis")
    from app.gemini_provider import GeminiProvider
    from app.ollama_provider import OllamaProvider
    from app.openai_provider import OpenAIProvider

    available: dict[str, Callable[[Settings], LLMProvider]] = constructors or {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "groq": __import__("app.groq_provider", fromlist=["GroqProvider"]).GroqProvider,
        "cerebras": __import__("app.cerebras_provider", fromlist=["CerebrasProvider"]).CerebrasProvider,
        "openrouter": __import__("app.openrouter_provider", fromlist=["OpenRouterProvider"]).OpenRouterProvider,
        "cohere": __import__("app.cohere_provider", fromlist=["CohereProvider"]).CohereProvider,
        "mistral": __import__("app.mistral_provider", fromlist=["MistralProvider"]).MistralProvider,
        "ollama": OllamaProvider,
    }
    configured_keys = {
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
        "groq": settings.groq_api_key and settings.groq_model,
        "cerebras": settings.cerebras_api_key and settings.cerebras_model,
        "openrouter": settings.openrouter_api_key and settings.openrouter_model,
        "cohere": settings.cohere_api_key and settings.cohere_model,
        "mistral": settings.mistral_api_key and settings.mistral_model,
        "ollama": True,
    }
    ordered_names = list(dict.fromkeys(settings.llm_provider_order))
    providers: list[tuple[str, LLMProvider]] = []
    for name in ordered_names:
        constructor = available.get(name)
        if constructor is None:
            logger.info("provider=%s status=skipped reason=unknown_provider", name)
            continue
        if not configured_keys.get(name):
            logger.info("provider=%s status=skipped reason=missing_configuration", name)
            continue
        try:
            provider = constructor(settings)
        except (LLMProviderError, InvalidProviderResponseError) as exc:
            logger.warning(
                "provider=%s status=skipped reason=construction_unavailable type=%s",
                name,
                exc.__class__.__name__,
            )
            continue
        except Exception:
            logger.exception("provider=%s status=construction_failed", name)
            raise
        providers.append((name, provider))
    if not providers:
        raise LLMProviderError("No configured LLM providers are available")
    logger.info("configured provider order=%s", list(settings.llm_provider_order))
    logger.info("constructed providers=%s", [name for name, _ in providers])
    return ProviderManager(providers, logger=logger)
