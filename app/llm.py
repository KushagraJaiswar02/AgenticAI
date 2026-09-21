"""Provider-neutral LLM contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.errors import LLMProviderError


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """Structured tool request chosen by the LLM for a specific prompt."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by any LLM provider."""

    text: str = ""
    tool_call: ToolCall | None = None


class LLMProvider(ABC):
    """Interface owned by JARVIS and consumed by the orchestrator."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response for a prompt, optionally with available tools or a tool result."""
        raise NotImplementedError


def provider_failure(message: str, cause: Exception) -> LLMProviderError:
    """Build a safe provider failure without exposing provider details."""
    return LLMProviderError(message)
