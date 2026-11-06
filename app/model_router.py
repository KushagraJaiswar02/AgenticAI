"""Deterministic task classification and configurable local model selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings
from app.model_registry import ModelRegistry, TaskCategory


@dataclass(frozen=True)
class RoutingDecision:
    category: TaskCategory
    provider: str
    model: str


class ModelRouter:
    def __init__(self, settings: Settings, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)

    def classify(self, prompt: str, *, has_tools: bool = False, has_image: bool = False) -> TaskCategory:
        if has_image:
            return "vision"
        if has_tools:
            return "tools"
        value = prompt.lower()
        if re.search(r"\b(code|coding|program|python|debug|implement|compile|function|class|bug|script)\b", value):
            return "coding"
        if re.search(r"\b(analy[sz]e|reason|compare|evaluate|design|architecture|why|plan|investigate|derive)\b", value):
            return "reasoning"
        return "general"

    def local_model(self, prompt: str, *, has_tools: bool = False, has_image: bool = False) -> RoutingDecision:
        category = self.classify(prompt, has_tools=has_tools, has_image=has_image)
        configured = {
            "general": self.settings.ollama_general_model,
            "coding": self.settings.ollama_coding_model,
            "tools": self.settings.ollama_tool_model,
            "vision": self.settings.ollama_vision_model,
            "reasoning": self.settings.ollama_coding_model,
        }
        return RoutingDecision(category, "ollama", configured.get(category, self.settings.ollama_general_model))
