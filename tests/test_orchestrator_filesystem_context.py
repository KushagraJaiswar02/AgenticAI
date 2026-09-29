from pathlib import Path

import pytest

from app.fake_provider import FakeLLMProvider
from app.filesystem_context import FilesystemContext
from app.filesystem_security import FilesystemBoundary
from app.filesystem_tools import create_filesystem_tools
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.session import ConversationState
from app.tools import ToolRegistry


def setup_orchestrator(tmp_path: Path, responses: list[LLMResponse]):
    workspace = tmp_path / "JARVIS"
    workspace.mkdir()
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary)
    provider = FakeLLMProvider(responses)
    return Orchestrator(provider, ToolRegistry(create_filesystem_tools(boundary, context))), provider, context


def text_response(text: str = "done") -> LLMResponse:
    return LLMResponse(text=text)


def test_multiturn_create_write_read_rename_write_read(tmp_path: Path):
    responses = [
        LLMResponse(tool_call=ToolCall("create_file", {"path": "hello.txt"})), text_response(),
        LLMResponse(tool_call=ToolCall("write_file", {"path": "wrong.txt", "content": "hallucinated"})),
        LLMResponse(tool_call=ToolCall("list_directory", {"path": "wrong.txt"})), text_response("hello"),
        LLMResponse(tool_call=ToolCall("list_directory", {"path": "wrong.txt"})),
        LLMResponse(tool_call=ToolCall("write_file", {"path": "wrong.txt", "content": "Goodbye!"})),
        LLMResponse(tool_call=ToolCall("list_directory", {"path": "wrong.txt"})), text_response("new content"),
    ]
    orchestrator, provider, context = setup_orchestrator(tmp_path, responses)

    orchestrator.process("create hello.txt")
    assert orchestrator.process('write "GODDAMIT" to it').startswith("I couldn't complete")
    assert "hello.txt" in orchestrator._state.pending_confirmation.request.prompt
    assert "GODDAMIT" in orchestrator._state.pending_confirmation.request.prompt
    orchestrator.process("y")
    assert (context.current_workspace / "hello.txt").read_text(encoding="utf-8") == "GODDAMIT"

    orchestrator.process("read it")
    assert provider.calls[4].tool_result["content"] == "GODDAMIT"
    assert orchestrator.process("rename it to goodbye.txt").startswith("I couldn't complete")
    orchestrator.process("y")
    assert context.last_modified_file == context.current_workspace / "goodbye.txt"

    orchestrator.process('write "new content" to it')
    assert orchestrator._state.pending_confirmation.request.arguments["path"].endswith("goodbye.txt")
    assert orchestrator._state.pending_confirmation.request.arguments["content"] == "new content"
    orchestrator.process("y")
    orchestrator.process("read it")
    assert provider.calls[-1].tool_result["content"] == "new content"


def test_ambiguous_context_does_not_guess(tmp_path: Path):
    responses = [
        LLMResponse(tool_call=ToolCall("create_file", {"path": "a.txt"})), text_response(),
        LLMResponse(tool_call=ToolCall("create_file", {"path": "b.txt"})), text_response(),
        LLMResponse(tool_call=ToolCall("write_file", {"path": "x", "content": "wrong"})),
    ]
    orchestrator, _, context = setup_orchestrator(tmp_path, responses)
    orchestrator.process("create a.txt")
    orchestrator.process("create b.txt")
    result = orchestrator.process('write "x" to it')
    assert "Which filesystem object do you mean" in result
    assert not (context.current_workspace / "a.txt").read_text(encoding="utf-8") if (context.current_workspace / "a.txt").exists() else True
    assert not (context.current_workspace / "b.txt").read_text(encoding="utf-8") if (context.current_workspace / "b.txt").exists() else True
    assert orchestrator._state.pending_confirmation is None


def test_missing_context_asks_for_target(tmp_path: Path):
    orchestrator, _, _ = setup_orchestrator(
        tmp_path,
        [LLMResponse(tool_call=ToolCall("write_file", {"path": "x", "content": "wrong"}))],
    )
    result = orchestrator.process('write "x" to it')
    assert "Which filesystem object do you mean" in result


def test_missing_context_rename_asks_instead_of_selecting_directory(tmp_path: Path):
    orchestrator, _, context = setup_orchestrator(
        tmp_path,
        [LLMResponse(tool_call=ToolCall("list_directory", {"path": "goodbye.txt"}))],
    )
    result = orchestrator.process("rename it to goodbye.txt")
    assert "Which filesystem object do you mean" in result
    assert orchestrator._state.pending_confirmation is None
    assert list(context.current_workspace.iterdir()) == []


def test_contextual_delete_uses_current_file(tmp_path: Path):
    orchestrator, _, context = setup_orchestrator(
        tmp_path,
        [
            LLMResponse(tool_call=ToolCall("create_file", {"path": "green.txt"})), text_response(),
            LLMResponse(tool_call=ToolCall("list_directory", {"path": "wrong"})),
        ],
    )
    orchestrator.process("create green.txt")
    orchestrator.process("delete it")
    assert "green.txt" in orchestrator._state.pending_confirmation.request.prompt
    orchestrator.process("y")
    assert not (context.current_workspace / "green.txt").exists()


def test_contextual_move_uses_current_file_and_destination(tmp_path: Path):
    orchestrator, _, context = setup_orchestrator(
        tmp_path,
        [
            LLMResponse(tool_call=ToolCall("create_file", {"path": "green.txt"})), text_response(),
            LLMResponse(tool_call=ToolCall("list_directory", {"path": "wrong"})),
        ],
    )
    orchestrator.process("create green.txt")
    orchestrator.process("move it to archive")
    pending = orchestrator._state.pending_confirmation
    assert pending is not None
    assert pending.request.arguments["source"].endswith("green.txt")
    assert pending.request.arguments["destination"].endswith("archive")
    orchestrator.process("y")
    assert not (context.current_workspace / "green.txt").exists()
    assert (context.current_workspace / "archive").exists()


def test_follow_up_search_uses_last_created_workspace_directory(tmp_path: Path):
    workspace = tmp_path / "WorkSpace"
    project = workspace / "JARVIS"
    project.mkdir(parents=True)
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary, project)
    target = workspace / "square.py"
    target.write_text("square", encoding="utf-8")
    context.remember_file(target, created=True)
    state = ConversationState(filesystem_context=context)

    resolved = state.prepare_tool_call(
        ToolCall("search_files", {"root": str(project), "pattern": "square.py"}),
        "search for square.py",
    )

    assert resolved.arguments["root"] == str(workspace)


def test_explicit_search_location_wins_over_context(tmp_path: Path):
    workspace = tmp_path / "WorkSpace"
    project = workspace / "JARVIS"
    project.mkdir(parents=True)
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary, project)
    state = ConversationState(filesystem_context=context)

    resolved = state.prepare_tool_call(
        ToolCall("search_files", {"root": str(project), "pattern": "config.json"}),
        "search for config.json in my project",
    )

    assert resolved.arguments["root"] == "my project"


def test_two_turn_create_then_write_receives_prior_history_and_uses_context(tmp_path: Path):
    responses = [
        LLMResponse(tool_call=ToolCall("create_file", {"name": "green lanterns.txt", "location": "workspace"})), text_response(),
        LLMResponse(tool_call=ToolCall("write_file", {"path": "example.txt", "content": "wrong"})), text_response(),
    ]
    workspace = tmp_path / "JARVIS"
    workspace.mkdir()
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary)
    provider = FakeLLMProvider(responses)
    orchestrator = Orchestrator(
        provider,
        ToolRegistry(create_filesystem_tools(boundary, context)),
        confirmation_callback=lambda _: "yes",
    )

    orchestrator.process("create a file named green lanterns.txt in the workspace")
    orchestrator.process('write "hello" into the file we just created')

    target = workspace / "green lanterns.txt"
    assert target.read_text(encoding="utf-8") == "hello"
    second_turn = provider.calls[2]
    assert second_turn.conversation is not None
    assert [item.role for item in second_turn.conversation] == ["user", "assistant", "assistant", "assistant"]
    assert second_turn.conversation[0].content == "create a file named green lanterns.txt in the workspace"
    assert "TOOL_CALL create_file" in second_turn.conversation[1].content
    assert "TOOL_RESULT create_file" in second_turn.conversation[2].content
    assert second_turn.conversation[3].content == "done"
    assert context.last_created_path == target


def test_history_is_bounded_and_session_state_isolated():
    first = ConversationState()
    for index in range(60):
        first.record_turn(f"user {index}", f"assistant {index}")
    assert len(first.history) == 50
    assert [message.content for message in first.recent_messages()] == [
        value for index in range(50, 60) for value in (f"user {index}", f"assistant {index}")
    ]
    second = ConversationState()
    assert second.history == []
    assert second.recent_messages() == []


def test_golden_create_write_read_returns_actual_content(tmp_path: Path):
    workspace = tmp_path / "JARVIS"
    workspace.mkdir()
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary)
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall("create_file", {"path": "green.txt"})), text_response(),
        LLMResponse(tool_call=ToolCall("write_file", {"path": "wrong.txt", "content": "hello"})), text_response(),
        LLMResponse(tool_call=ToolCall("read_file", {"path": "wrong.txt"})), LLMResponse(text="hello"),
    ])
    orchestrator = Orchestrator(
        provider,
        ToolRegistry(create_filesystem_tools(boundary, context)),
        confirmation_callback=lambda _: "yes",
    )

    orchestrator.process("create green.txt")
    orchestrator.process("write hello in it")
    final = orchestrator.process("read it")

    target = workspace / "green.txt"
    assert target.read_text(encoding="utf-8") == "hello"
    assert final == "hello"
    assert provider.calls[5].tool_result["path"] == str(target)
    assert provider.calls[5].tool_result["content"] == "hello"
