"""Fast deterministic task-complexity classification for Ollama only."""

from __future__ import annotations

import re
from enum import Enum


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class TaskComplexityRouter:
    """Classify requests without network access or another model call."""

    _COMPLEX = (
        "analyze", "debug", "diagnose", "compare", "design", "architect",
        "plan", "investigate", "explain why", "reason", "optimize",
        "evaluate", "troubleshoot", "derive",
    )
    _SIMPLE = (
        "weather", "time", "date", "calculate", "lookup", "what is", "who is",
        "open", "close", "launch", "play", "stop", "search",
    )

    def classify(self, prompt: str) -> TaskComplexity:
        normalized = " ".join(prompt.lower().split())
        complex_score = sum(1 for signal in self._COMPLEX if signal in normalized)
        simple_score = sum(1 for signal in self._SIMPLE if signal in normalized)
        if len(normalized.split()) >= 35:
            complex_score += 1
        if re.search(r"\b(first|then|after that|and compare|multiple steps)\b", normalized):
            complex_score += 2
        if complex_score > 0 and complex_score >= simple_score:
            return TaskComplexity.COMPLEX
        return TaskComplexity.SIMPLE

    def should_think(self, prompt: str) -> bool:
        return self.classify(prompt) is TaskComplexity.COMPLEX
