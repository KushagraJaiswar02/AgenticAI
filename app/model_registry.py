"""Central model catalogue and configuration-aware registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings


TaskCategory = Literal["general", "coding", "tools", "vision", "reasoning"]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    identifier: str
    capabilities: frozenset[str]
    input_types: frozenset[str]
    task_categories: frozenset[TaskCategory]
    local: bool
    available: bool | None
    default: bool = False


class ModelRegistry:
    """Small declarative model catalogue; availability means configured locally."""

    def __init__(self, settings: Settings, *, ollama_models: set[str] | None = None) -> None:
        installed = ollama_models
        def local_available(identifier: str) -> bool | None:
            return None if installed is None else identifier in installed

        self._models = (
            ModelSpec("gemini", settings.gemini_model, frozenset({"chat", "tools"}), frozenset({"text"}), frozenset({"general", "coding", "tools", "reasoning"}), False, bool(settings.gemini_api_key), True),
            ModelSpec("openrouter", settings.openrouter_model, frozenset({"chat", "tools"}), frozenset({"text"}), frozenset({"general", "coding", "tools", "reasoning"}), False, bool(settings.openrouter_api_key and settings.openrouter_model)),
            ModelSpec("grok", settings.grok_model, frozenset({"chat", "tools"}), frozenset({"text"}), frozenset({"general", "coding", "tools", "reasoning"}), False, bool(settings.grok_api_key)),
            ModelSpec("ollama", settings.ollama_general_model, frozenset({"chat"}), frozenset({"text"}), frozenset({"general"}), True, local_available(settings.ollama_general_model), True),
            ModelSpec("ollama", settings.ollama_coding_model, frozenset({"chat", "tools"}), frozenset({"text"}), frozenset({"coding", "tools", "reasoning"}), True, local_available(settings.ollama_coding_model)),
            ModelSpec("ollama", settings.ollama_tool_model, frozenset({"chat", "tools"}), frozenset({"text"}), frozenset({"tools"}), True, local_available(settings.ollama_tool_model)),
            ModelSpec("ollama", settings.ollama_vision_model, frozenset({"chat", "vision"}), frozenset({"text", "image"}), frozenset({"vision"}), True, local_available(settings.ollama_vision_model)),
        )

    def all(self) -> tuple[ModelSpec, ...]:
        return self._models

    def find(self, provider: str, identifier: str) -> ModelSpec | None:
        return next((model for model in self._models if model.provider == provider and model.identifier == identifier), None)
