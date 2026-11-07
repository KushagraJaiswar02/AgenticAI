from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.fake_provider import FakeLLMProvider
from app.filesystem_security import FilesystemBoundary
from app.filesystem_tools import CreateFileTool
from app.gemini_provider import GeminiProvider
from app.groq_provider import GroqProvider
from app.llm import ConversationMessage, LLMResponse, ToolCall
from app.ollama_provider import OllamaProvider
from app.openrouter_provider import OpenRouterProvider
from app.orchestrator import Orchestrator
from app.policy import RiskLevel
from app.provider_manager import ProviderManager
from app.session import ConversationState
from app.tools import Tool, ToolExecutionResult, ToolRegistry


class CountInput(BaseModel):
    value: int


class CountingTool(Tool):
    name = "count"
    description = "Count one invocation."
    input_model = CountInput

    def __init__(self, risk=RiskLevel.SAFE):
        self.risk_level = risk
        self.values = []

    def execute(self, arguments):
        self.values.append(arguments.value)
        return ToolExecutionResult(True, {"result": len(self.values)})


def call(value, call_id=None):
    return LLMResponse(tool_call=ToolCall("count", {"value": value}, call_id))


@pytest.mark.parametrize(("risk", "ids", "count"), [
    (RiskLevel.SAFE, ("a", "a"), 1),
    (RiskLevel.SAFE, ("a", "b"), 2),
    (RiskLevel.SAFE, (None, None), 2),
    (RiskLevel.LOW, ("a", "b"), 1),
    (RiskLevel.LOW, (None, None), 1),
    (RiskLevel.CONFIRM, ("a", "b"), 1),
])
def test_turn_replay_and_repeated_reads(risk, ids, count):
    tool = CountingTool(risk)
    provider = FakeLLMProvider([call(1, ids[0]), call(1, ids[1]), LLMResponse(text="done")])
    confirmations = []

    def confirm(prompt):
        confirmations.append(prompt)
        return "yes"

    response = Orchestrator(provider, ToolRegistry([tool]), confirmation_callback=confirm).process("count")
    assert len(tool.values) == count
    if ids == ("a", "a"):
        assert provider.calls[1].tool_result == provider.calls[2].tool_result == {"result": 1}
    if risk != RiskLevel.SAFE:
        assert "already completed" in response
        assert len(provider.calls) == 2
    assert len(confirmations) == (1 if risk == RiskLevel.CONFIRM else 0)


def test_completed_id_cannot_be_reused_with_different_arguments():
    tool = CountingTool()
    provider = FakeLLMProvider([call(1, "same"), call(2, "same")])
    result = Orchestrator(provider, ToolRegistry([tool])).process("count")
    assert "different arguments" in result
    assert tool.values == [1]


def test_file_create_replay_matches_semantic_and_explicit_paths(tmp_path):
    target = tmp_path / "once.txt"
    tool = CreateFileTool(FilesystemBoundary([tmp_path]))
    registry = ToolRegistry([tool])
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall("create_file", {"name": "once.txt", "content": "once"}, "a")),
        LLMResponse(tool_call=ToolCall("create_file", {"path": str(target), "content": "once"}, "b")),
    ])
    assert "already completed" in Orchestrator(provider, registry).process("create once.txt")
    assert target.read_text() == "once"
    assert [record["execution_result"] for record in registry.audit_logger.records] == [
        "completed", "not_executed",
    ]


def test_provider_continuation_uses_original_arguments_after_local_resolution(monkeypatch):
    tool = CountingTool()
    provider = FakeLLMProvider([call(1, "one"), LLMResponse(text="done")])
    state = ConversationState()
    monkeypatch.setattr(state, "resolve_filesystem_intent",
                        lambda *args, **kwargs: ToolCall("count", {"value": 2}, "one"))
    assert Orchestrator(provider, ToolRegistry([tool]), state=state).process("count") == "done"
    assert tool.values == [2]
    assert provider.calls[1].tool_call.arguments == {"value": 1}


def test_same_id_can_correct_validation_and_be_used_in_a_new_turn():
    tool = CountingTool(RiskLevel.LOW)
    provider = FakeLLMProvider([
        call("invalid", "same"), call(1, "same"), LLMResponse(text="fixed"),
        call(1, "same"), LLMResponse(text="new turn"),
    ])
    agent = Orchestrator(provider, ToolRegistry([tool]))
    assert agent.process("count") == "fixed"
    assert agent.process("count again") == "new turn"
    assert provider.calls[1].tool_result["success"] is False
    assert tool.values == [1, 1]


class ScriptedProvider:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def generate(self, prompt, tools=None, *, tool_call=None, tool_result=None, think=None, conversation=None):
        self.calls.append((prompt, tool_call, tool_result, conversation))
        return self.respond(prompt, tool_call)


def test_interleaved_requests_keep_owners_even_with_identical_call_ids():
    def primary(prompt, pending):
        if prompt == "second":
            raise LLMProviderError("primary down")
        return LLMResponse(text="primary") if pending else call(1, "same")

    first = ScriptedProvider(primary)
    second = ScriptedProvider(lambda prompt, pending: LLMResponse(text="backup") if pending else call(2, "same"))
    manager = ProviderManager([("primary", first), ("backup", second)])
    a = manager.generate("first").tool_call
    b = manager.generate("second").tool_call
    assert manager.generate("first", tool_call=a, tool_result={}).text == "primary"
    assert manager.generate("second", tool_call=b, tool_result={}).text == "backup"
    with pytest.raises(LLMProviderError, match="no active provider owner"):
        manager.generate("first", tool_call=a, tool_result={})
    with pytest.raises(LLMProviderError, match="no active provider owner"):
        manager.generate("first", tool_call=ToolCall("count", {}, "same"), tool_result={})
    assert len(first.calls) == 3
    assert len(second.calls) == 2


class RateLimitError(Exception):
    code = 429


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),
    RateLimitError(),
    InvalidProviderResponseError("malformed tool arguments"),
])
def test_failover_passes_same_history_and_executes_only_backup_call(error):
    def fail(prompt, pending):
        raise error

    first = ScriptedProvider(fail)
    second = ScriptedProvider(lambda prompt, pending: LLMResponse(text="done") if pending else call(7, "backup"))
    manager = ProviderManager([("primary", first), ("backup", second)])
    state = ConversationState()
    state.record_turn("prior question", "prior answer")
    history = state.recent_messages()
    tool = CountingTool()
    assert Orchestrator(manager, ToolRegistry([tool]), state=state).process("count") == "done"
    assert first.calls[0][3] == second.calls[0][3] == second.calls[1][3] == history
    assert tool.values == [7]


def test_failure_finalizes_events_under_original_turn():
    def respond(prompt, pending):
        if prompt == "next":
            return LLMResponse(text="ok")
        if pending:
            raise LLMProviderError("provider down")
        return call(3, "one")

    provider = ScriptedProvider(respond)
    manager = ProviderManager([("primary", provider)])
    state = ConversationState()
    agent = Orchestrator(manager, ToolRegistry([CountingTool()]), state=state)
    assert "tool completed" in agent.process("count")
    assert state.pending_tool_events == []
    assert agent.process("next") == "ok"
    prior = provider.calls[-1][3]
    assert prior[0] == ConversationMessage("user", "count")
    assert [m.content for m in prior if m.content.startswith("TOOL_RESULT")] == [
        'TOOL_RESULT count {"result": 1}',
    ]


class Queue:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return next(self.responses)

    generate_content = create

    def post(self, url, *, json, timeout):
        self.calls.append(json)
        result = next(self.responses)

        def raise_for_status():
            if isinstance(result, Exception):
                raise result

        return SimpleNamespace(raise_for_status=raise_for_status, json=lambda: result)


def chat_response(*ids, model="backend"):
    calls = [
        SimpleNamespace(id=identifier, function=SimpleNamespace(name="count", arguments='{"value": 1}'))
        for identifier in ids
    ]
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=SimpleNamespace(
        tool_calls=calls, content=None if calls else "done",
    ))])


def gemini_response(*ids):
    parts = [
        SimpleNamespace(function_call=SimpleNamespace(name="count", args={"value": 1}, id=identifier),
                        thought_signature=b"native-signature")
        for identifier in ids
    ]
    return SimpleNamespace(
        text=None if parts else "done",
        candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=parts))] if parts else [],
    )


def ollama_response(*ids):
    return {"message": {
        "tool_calls": [
            {"id": identifier, "function": {"name": "count", "arguments": {"value": 1}}}
            for identifier in ids
        ],
        "content": "" if ids else "done",
        "thinking": "native reasoning",
    }}


@pytest.fixture(params=["gemini", "openrouter", "groq", "ollama"])
def adapter(request):
    name = request.param
    settings = Settings(gemini_api_key="test", openrouter_api_key="test", openrouter_model="openrouter/free",
                        groq_api_key="test")

    def build(*batches):
        factory = {"gemini": gemini_response, "ollama": ollama_response}.get(name, chat_response)
        queue = Queue([factory(*batch) for batch in batches])
        if name == "gemini":
            provider = GeminiProvider(settings, client=SimpleNamespace(models=queue))
        elif name == "ollama":
            provider = OllamaProvider(settings, client=queue)
        else:
            cls = OpenRouterProvider if name == "openrouter" else GroqProvider
            provider = cls(settings, client=SimpleNamespace(chat=SimpleNamespace(completions=queue)))
        return name, provider, queue

    return build


def test_adapter_preserves_history_and_successive_native_tool_results(adapter):
    name, provider, queue = adapter(("one",), ("two",), ())
    state = ConversationState()
    state.record_turn("earlier question", "earlier answer")
    tool = CountingTool()
    manager = ProviderManager([(name, provider)])
    assert Orchestrator(manager, ToolRegistry([tool]), state=state).process("count") == "done"
    assert tool.values == [1, 1]
    assert len({payload["model"] for payload in queue.calls}) == 1
    last = queue.calls[-1]
    if name == "gemini":
        contents = last["contents"]
        assert [part.role for part in contents] == ["user", "model", "user", "model", "user", "model", "user"]
        assert contents[1].parts[0].text == "earlier answer"
        assert contents[3].parts[0].thought_signature == b"native-signature"
        assert [contents[i].parts[0].function_response.id for i in (4, 6)] == ["one", "two"]
        assert last["config"].tools[0].function_declarations[0].name == "count"
    else:
        messages = last["messages"]
        assert [message["role"] for message in messages] == [
            "system", "user", "assistant", "user", "assistant", "tool", "assistant", "tool",
        ]
        assert messages[2]["content"] == "earlier answer"
        assert [messages[i]["content"] for i in (5, 7)] == ['{"result": 1}', '{"result": 2}']
        if name == "ollama":
            assert [messages[i]["tool_name"] for i in (5, 7)] == ["count", "count"]
            assert messages[4]["thinking"] == "native reasoning"
        else:
            assert [messages[i]["tool_call_id"] for i in (5, 7)] == ["one", "two"]


def test_adapter_rejects_multiple_calls_before_dispatch(adapter):
    name, provider, _ = adapter(("one", "two"))
    tool = CountingTool()
    with pytest.raises(InvalidProviderResponseError, match="multiple tool calls"):
        Orchestrator(provider, ToolRegistry([tool])).process("count")
    assert tool.values == []


def test_interleaved_native_transcripts_remain_separate(adapter):
    name, provider, queue = adapter(("same",), ("same",), (), ())
    manager = ProviderManager([(name, provider)])
    first = manager.generate("first request").tool_call
    second = manager.generate("second request").tool_call
    assert manager.generate("first request", tool_call=first, tool_result={"result": 1}).text == "done"
    assert manager.generate("second request", tool_call=second, tool_result={"result": 2}).text == "done"
    if name == "gemini":
        assert [payload["contents"][0].parts[0].text for payload in queue.calls[2:]] == [
            "first request", "second request",
        ]
    else:
        assert [payload["messages"][1]["content"] for payload in queue.calls[2:]] == [
            "first request", "second request",
        ]


def test_invalid_batch_after_execution_does_not_dispatch_or_fail_over(adapter):
    name, provider, _ = adapter(("one",), ("two", "three"))
    backup = ScriptedProvider(lambda prompt, pending: LLMResponse(text="backup"))
    manager = ProviderManager([(name, provider), ("backup", backup)])
    tool = CountingTool()
    assert "tool completed" in Orchestrator(manager, ToolRegistry([tool])).process("count")
    assert tool.values == [1]
    assert backup.calls == []


def test_openrouter_backend_change_is_rejected_without_switching_router():
    queue = Queue([chat_response("one", model="first"), chat_response("two", model="second")])
    provider = OpenRouterProvider(
        Settings(openrouter_api_key="test", openrouter_model="openrouter/free"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=queue)),
    )
    first = provider.generate("count")
    with pytest.raises(InvalidProviderResponseError, match="changed models"):
        provider.generate("count", tool_call=first.tool_call, tool_result={"result": 1})
    assert [payload["model"] for payload in queue.calls] == ["openrouter/free", "openrouter/free"]


def test_openrouter_preserves_native_assistant_metadata():
    initial = chat_response("one")
    native_message = initial.choices[0].message
    native_message.content = "I will use the tool."
    native_message.reasoning_details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    queue = Queue([initial, chat_response()])
    provider = OpenRouterProvider(
        Settings(openrouter_api_key="test", openrouter_model="openrouter/free"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=queue)),
    )
    first = provider.generate("count")
    provider.generate("count", tool_call=first.tool_call, tool_result={"result": 1})
    assistant = queue.calls[-1]["messages"][-2]
    assert assistant["content"] == native_message.content
    assert assistant["reasoning_details"] == native_message.reasoning_details
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"value": 1}'


def test_ollama_missing_continuation_model_does_not_retry_on_general_model():
    missing = RuntimeError("model unavailable")
    missing.response = SimpleNamespace(status_code=404)
    queue = Queue([ollama_response("one"), missing])
    provider = OllamaProvider(Settings(), client=queue)
    tool = CountingTool()
    manager = ProviderManager([("ollama", provider)])
    result = Orchestrator(manager, ToolRegistry([tool])).process("calculate using count")
    assert "tool completed" in result
    assert tool.values == [1]
    assert [payload["model"] for payload in queue.calls] == ["qwen3:4b", "qwen3:4b"]


def test_gemini_textual_tool_result_uses_text_transcript():
    queue = Queue([SimpleNamespace(text='TOOL_CALL count {"value":1}'), gemini_response()])
    provider = GeminiProvider(Settings(gemini_api_key="test"), client=SimpleNamespace(models=queue))
    tool = CountingTool()
    assert Orchestrator(provider, ToolRegistry([tool])).process("count") == "done"
    parts = [content.parts[0] for content in queue.calls[-1]["contents"]]
    assert [part.text for part in parts] == [
        "count", 'TOOL_CALL count {"value":1}', 'TOOL_RESULT count {"result": 1}',
    ]
