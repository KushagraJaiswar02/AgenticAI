"""Deterministic local intent recognition and fallback tool-call routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.llm import ToolCall


@dataclass(frozen=True)
class IntentDefinition:
    """Stable metadata for one locally supported intent."""

    name: str
    tool_name: str
    examples: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentMatch:
    """Provider-independent result of local intent recognition."""

    intent: str
    confidence: float
    entities: dict[str, Any]


class IntentRecognizer(Protocol):
    """Extension point for future semantic or local-model recognizers."""

    def recognize(self, normalized_message: str) -> IntentMatch | None:
        ...


class RuleIntentRecognizer:
    """Small, deterministic recognizer for the currently supported intents."""

    definitions = (
        IntentDefinition("weather", "weather", ("weather in a city", "is it raining")),
        IntentDefinition("time", "time", ("what time is it", "current time")),
        IntentDefinition("date", "date", ("today's date", "what date is it")),
        IntentDefinition("calculator", "calculator", ("calculate 2 plus 2",)),
        IntentDefinition("system_info", "system_info", ("what os am i running",)),
        IntentDefinition("open_application", "open_application", ("open brave",)),
    )

    _WEATHER_WITH_LOCATION = (
        re.compile(r"\b(?:weather|forecast)\s+(?:in|at|for)\s+(?P<location>.+)$"),
        re.compile(r"\b(?:is|will)\s+it\s+rain(?:ing)?\s+(?:in|at)\s+(?P<location>.+)$"),
        re.compile(r"\bhow\s+(?:hot|cold)\s+is\s+it\s+(?:in|at)\s+(?P<location>.+)$"),
        re.compile(r"\b(?:what\s+is|what's)\s+the\s+temperature\s+(?:in|at)\s+(?P<location>.+)$"),
        re.compile(r"\bshould\s+i\s+take\s+an?\s+umbrella\s+(?:in|at)\s+(?P<location>.+)$"),
        re.compile(r"\bwhat\s+is\s+it\s+like\s+outside\s+in\s+(?P<location>.+)$"),
    )
    _WEATHER_LOCATION_FIRST = (
        re.compile(r"^(?:(?:tell me|show me|what is)\s+)?(?P<location>.+?)['’]s\s+weather(?:\s+like)?$"),
        re.compile(r"^(?:how is\s+)?(?P<location>.+?)\s+weather(?:\s+like)?$"),
    )
    _WEATHER_WORDS = (
        "weather",
        "forecast",
        "rain",
        "raining",
        "temperature",
        "hot",
        "cold",
        "umbrella",
        "outside",
    )
    _WEATHER_FALSE_POSITIVES = ("laptop", "cpu", "computer", "engine", "battery")

    _TIME_PATTERNS = (
        re.compile(r"^what time is it(?: right now)?$"),
        re.compile(r"^(?:tell me )?(?:the )?(?:current )?time$"),
        re.compile(r"^can you tell me the time$"),
    )
    _DATE_PATTERNS = (
        re.compile(r"^(?:what is|what's) today'?s date$"),
        re.compile(r"^what date is it$"),
        re.compile(r"^(?:tell me )?today'?s date$"),
        re.compile(r"^what day is today$"),
    )
    _SYSTEM_PATTERNS = (
        re.compile(r"^what system am i running$"),
        re.compile(r"^what os am i running$"),
        re.compile(r"^what (?:os|operating system) is this$"),
        re.compile(r"^what operating system do i have$"),
        re.compile(r"^(?:show me|tell me about) (?:my )?(?:system information|this computer)$"),
        re.compile(r"^what python version am i using$"),
    )
    _OPEN_APPLICATION_PATTERN = re.compile(
        r"^(?:open|launch|start)\s+(?P<name>[a-z0-9_-]+)$"
    )

    def recognize(self, normalized_message: str) -> IntentMatch | None:
        weather = self._recognize_weather(normalized_message)
        if weather is not None:
            return weather
        if any(pattern.fullmatch(normalized_message) for pattern in self._TIME_PATTERNS):
            return IntentMatch("time", 0.98, {})
        if any(pattern.fullmatch(normalized_message) for pattern in self._DATE_PATTERNS):
            return IntentMatch("date", 0.98, {})
        if any(pattern.fullmatch(normalized_message) for pattern in self._SYSTEM_PATTERNS):
            return IntentMatch("system_info", 0.97, {})
        application = self._OPEN_APPLICATION_PATTERN.fullmatch(normalized_message)
        if application is not None:
            return IntentMatch(
                "open_application",
                0.96 if application.group("name") == "brave" else 0.9,
                {"name": application.group("name")},
            )
        calculator = self._recognize_calculator(normalized_message)
        if calculator is not None:
            return calculator
        return None

    def _recognize_weather(self, message: str) -> IntentMatch | None:
        if any(word in message for word in self._WEATHER_FALSE_POSITIVES):
            return None
        if not any(word in message for word in self._WEATHER_WORDS):
            return None

        for pattern in self._WEATHER_WITH_LOCATION + self._WEATHER_LOCATION_FIRST:
            match = pattern.search(message)
            if not match:
                continue
            location = self._clean_location(match.group("location"))
            if location:
                return IntentMatch("weather", 0.94, {"location": location})
        return IntentMatch("weather", 0.55, {})

    @staticmethod
    def _recognize_calculator(message: str) -> IntentMatch | None:
        expression = message
        for prefix in ("calculate ", "what is ", "what's ", "how much is "):
            if expression.startswith(prefix):
                expression = expression[len(prefix):]
                break
        expression = re.sub(r"\bmultiplied by\b|\btimes\b", "*", expression)
        expression = re.sub(r"\bdivided by\b", "/", expression)
        expression = re.sub(r"\bplus\b", "+", expression)
        expression = re.sub(r"\bminus\b", "-", expression)
        expression = re.sub(r"\b(?P<number>\d+(?:\.\d+)?)\s+percent\s+of\s+(?P<base>\d+(?:\.\d+)?)\b",
                            r"(\g<number> / 100) * \g<base>", expression)
        if not re.fullmatch(r"[\d\s.+*/%()\-]+", expression):
            return None
        if not re.search(r"\d", expression) or not re.search(r"[+\-*/%]", expression):
            return None
        return IntentMatch("calculator", 0.96, {"expression": expression.strip()})

    @staticmethod
    def _clean_location(location: str) -> str | None:
        value = re.sub(r"[?.!,]+$", "", location).strip(" '\"")
        value = re.sub(r"\b(?:please|today|tonight|right now|now)\b$", "", value).strip()
        if not value or value in {"today", "tonight", "now", "outside", "what is the", "tell me"}:
            return None
        return " ".join(word.capitalize() for word in value.split())


class LocalIntentRouter:
    """Convert high-confidence local intent matches into provider-neutral calls."""

    def __init__(
        self,
        recognizer: IntentRecognizer | None = None,
        *,
        minimum_confidence: float = 0.8,
    ) -> None:
        self._recognizer = recognizer or RuleIntentRecognizer()
        self._minimum_confidence = minimum_confidence

    def normalize(self, message: str) -> str:
        """Normalize harmless surface variation without changing meaning."""
        normalized = message.strip().lower()
        replacements = {
            "what's": "what is",
            "whats": "what is",
            "i'm": "i am",
            "how's": "how is",
            "rn": "right now",
        }
        for source, target in replacements.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        normalized = re.sub(r"[!?]+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def recognize(self, message: str, context: Any | None = None) -> IntentMatch | None:
        """Recognize an intent; context is reserved for future contextual routing."""
        del context
        return self._recognizer.recognize(self.normalize(message))

    def route(self, message: str, context: Any | None = None) -> ToolCall | None:
        match = self.recognize(message, context=context)
        if match is None or match.confidence < self._minimum_confidence:
            return None
        definition = next(
            definition
            for definition in RuleIntentRecognizer.definitions
            if definition.name == match.intent
        )
        return ToolCall(name=definition.tool_name, arguments=match.entities)
