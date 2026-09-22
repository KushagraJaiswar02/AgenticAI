"""Deterministic local intent routing used only after provider failure."""

from __future__ import annotations

import re

from app.llm import ToolCall


class LocalIntentRouter:
    """Recognize a deliberately small set of obvious local tool intents."""

    _WEATHER_PATTERNS = (
        re.compile(
            r"\b(?:weather|temperature|raining|rain)\s+(?:in|at|for)\s+"
            r"(?P<location>[A-Za-z][A-Za-z .'-]*?)(?:\?|!|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:(?:tell|show)\s+me\s+)?"
            r"(?P<location>[A-Za-z][A-Za-z .'-]*?)['’]s\s+"
            r"(?:weather|temperature)(?:\?|!|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:how\s+is|how's|what(?:'s| is)\s+it\s+like)\s+"
            r"(?P<location>[A-Za-z][A-Za-z .'-]*?)\s+weather(?:\?|!|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat(?:'s| is)\s+it\s+like\s+outside\s+in\s+"
            r"(?P<location>[A-Za-z][A-Za-z .'-]*?)(?:\?|!|$)",
            re.IGNORECASE,
        ),
    )
    _NON_LOCATIONS = {"today", "tonight", "now", "outside", "there"}

    def route(self, text: str) -> ToolCall | None:
        """Return a weather ToolCall for an obvious request, otherwise None."""
        for pattern in self._WEATHER_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            location = self._normalize_location(match.group("location"))
            if location:
                return ToolCall(name="weather", arguments={"location": location})
        return None

    @classmethod
    def _normalize_location(cls, location: str) -> str | None:
        normalized = " ".join(location.strip(" .,'\"").split())
        if normalized.lower() in cls._NON_LOCATIONS:
            return None
        return normalized or None
