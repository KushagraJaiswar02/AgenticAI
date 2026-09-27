from app.errors import LLMProviderError
from app.llm import LLMProvider, LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.provider_manager import ProviderManager
from app.tools import ToolExecutionResult, ToolRegistry, Tool
from pydantic import BaseModel


class InputModel(BaseModel):
    value: str = "ok"


class CountingTool(Tool):
    name = "count"
    description = "Count executions."
    input_model = InputModel

    def __init__(self):
        self.count = 0

    def execute(self, arguments):
        self.count += 1
        return ToolExecutionResult(success=True, data={"result": self.count})


class ContinuationProvider(LLMProvider):
    def __init__(self, *, fail_on_continuation=False):
        self.fail_on_continuation = fail_on_continuation
        self.calls = []

    def generate(self, prompt, tools=None, *, tool_call=None, tool_result=None, think=None):
        self.calls.append((tool_call, tool_result))
        if tool_call is not None and self.fail_on_continuation:
            raise LLMProviderError("continuation failed")
        if tool_call is None:
            return LLMResponse(tool_call=ToolCall("count", {}))
        return LLMResponse(text="done")


def test_tool_is_not_reexecuted_when_continuation_fails():
    tool = CountingTool()
    first = ContinuationProvider(fail_on_continuation=True)
    second = ContinuationProvider()
    manager = ProviderManager([("first", first), ("second", second)])
    orchestrator = Orchestrator(manager, ToolRegistry([tool]))

    try:
        orchestrator.process("count")
    except Exception:
        pass

    assert tool.count == 1
    assert len(second.calls) == 0
