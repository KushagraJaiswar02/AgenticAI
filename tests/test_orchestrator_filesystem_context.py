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
