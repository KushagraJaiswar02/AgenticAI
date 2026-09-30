from pathlib import Path

import pytest

from app.fake_provider import FakeLLMProvider
from app.filesystem_context import FilesystemContext
from app.filesystem_security import FilesystemBoundary, FilesystemBoundaryError
from app.filesystem_tools import PathInput, create_filesystem_tools
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.tools import ToolRegistry


def setup_orchestrator(tmp_path: Path, responses: list[LLMResponse]):
    workspace = tmp_path / "JARVIS"
    workspace.mkdir()
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary)
    provider = FakeLLMProvider(responses)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    return Orchestrator(provider, registry), provider, workspace


def test_path_input_schema_describes_exact_path_key():
    schema = PathInput.model_json_schema()

    assert schema["properties"]["path"]["description"]
    assert "path" in schema["properties"]["path"]["description"]


def test_invalid_file_path_argument_is_retried_with_validation_feedback(tmp_path: Path):
    orchestrator, provider, workspace = setup_orchestrator(
        tmp_path,
        [
            LLMResponse(tool_call=ToolCall("read_file", {"file_path": "notes.txt"})),
            LLMResponse(tool_call=ToolCall("read_file", {"path": "notes.txt"})),
            LLMResponse(text="done"),
        ],
    )
    (workspace / "notes.txt").write_text("hello", encoding="utf-8")

    assert orchestrator.process("read notes.txt") == "done"
    assert len(provider.calls) == 3
    assert provider.calls[1].tool_result["missing_keys"] == ["path"]
    assert provider.calls[1].tool_result["unexpected_keys"] == ["file_path"]
    assert provider.calls[1].tool_result["expected_schema"]["properties"]["path"]["description"]


def test_boundary_error_does_not_trigger_validation_retry(tmp_path: Path):
    orchestrator, provider, _ = setup_orchestrator(
        tmp_path,
        [LLMResponse(tool_call=ToolCall("read_file", {"path": r"C:\outside.txt"}))],
    )

    with pytest.raises(FilesystemBoundaryError):
        orchestrator.process("read the outside file")

    assert len(provider.calls) == 1


def test_schema_validation_retries_stop_after_two(tmp_path: Path):
    bad_response = LLMResponse(tool_call=ToolCall("read_file", {"file_path": "notes.txt"}))
    orchestrator, provider, _ = setup_orchestrator(
        tmp_path,
        [bad_response, bad_response, bad_response],
    )

    result = orchestrator.process("read notes.txt")

    assert "couldn't complete the read_file request" in result
    assert len(provider.calls) == 3
