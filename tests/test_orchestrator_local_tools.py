from app.errors import LLMProviderError
from app.llm import LLMProvider
from app.orchestrator import Orchestrator
from app.tools import CalculatorTool, DateTool, SystemInfoTool, TimeTool, ToolRegistry


class FailingProvider(LLMProvider):
    def generate(self, prompt, tools=None, *, tool_call=None, tool_result=None):
        raise LLMProviderError("provider unavailable")


def test_provider_failure_falls_back_to_time() -> None:
    result = Orchestrator(FailingProvider(), ToolRegistry([TimeTool()])).process(
        "what time is it?"
    )
    assert result.startswith("The current local time is ")


def test_provider_failure_falls_back_to_date() -> None:
    result = Orchestrator(FailingProvider(), ToolRegistry([DateTool()])).process(
        "what's today's date?"
    )
    assert result.startswith("Today is ")


def test_provider_failure_falls_back_to_calculator() -> None:
    result = Orchestrator(FailingProvider(), ToolRegistry([CalculatorTool()])).process(
        "calculate 25 * 17"
    )
    assert result == "The result is 425."


def test_provider_failure_falls_back_to_system_info() -> None:
    result = Orchestrator(
        FailingProvider(),
        ToolRegistry([SystemInfoTool()]),
    ).process("what OS am I running?")
    assert result.startswith("You are running ")
