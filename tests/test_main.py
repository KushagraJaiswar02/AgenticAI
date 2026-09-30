from pathlib import Path

from app.fake_provider import FakeLLMProvider
from app.filesystem_context import FilesystemContext
from app.filesystem_security import FilesystemBoundary
from app.filesystem_tools import create_filesystem_tools
from app.llm import LLMResponse, ToolCall
from app.main import _run_cli_loop
from app.orchestrator import Orchestrator
from app.tools import ToolRegistry


def test_cli_skips_blank_input_after_successful_create_file(tmp_path: Path, monkeypatch, capsys):
    workspace = tmp_path / "JARVIS"
    workspace.mkdir()
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary)
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall(
            "create_file",
            {"name": "square.py", "location": "workspace", "content": "print('square')"},
        )),
        LLMResponse(text="Created square.py with the requested Python code."),
    ])
    orchestrator = Orchestrator(provider, ToolRegistry(create_filesystem_tools(boundary, context)))
    inputs = iter([
        "create a file called square.py and write python code for square pattern printing in it",
        "",
        "exit",
    ])
    monkeypatch.setattr("app.main.read_input", lambda: next(inputs))

    assert _run_cli_loop(orchestrator, orchestrator._logger) == 0

    output = capsys.readouterr().out
    assert "Created square.py" in output
    assert "could not process that request" not in output
    assert (workspace / "square.py").read_text(encoding="utf-8") == "print('square')"
    assert len(provider.calls) == 2
