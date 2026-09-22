"""Deterministic provider for offline application tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.errors import LLMProviderError
from app.llm import LLMProvider, LLMResponse, ToolCall, ToolDefinition


@dataclass(frozen=True)
class FakeGeneration:
    """Recorded arguments from one fake-provider call."""

    prompt: str
    tools: list[ToolDefinition] | None
    tool_call: ToolCall | None
    tool_result: dict[str, Any] | None


class FakeLLMProvider(LLMProvider):
    """Return predetermined responses without interpreting user language."""

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[FakeGeneration] = []

    def generate(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        *,
        tool_call: ToolCall | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            FakeGeneration(
                prompt=prompt,
                tools=tools,
                tool_call=tool_call,
                tool_result=tool_result,
            )
        )
        if self._index >= len(self._responses):
            raise LLMProviderError("FakeLLMProvider has no response configured for this call")
        response = self._responses[self._index]
        self._index += 1
        return response
