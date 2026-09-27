import subprocess

from app.tools import OpenApplicationTool


def test_brave_is_allowlisted_and_launched(monkeypatch) -> None:
    launched = []
    monkeypatch.setattr(OpenApplicationTool, "_find_executable", staticmethod(lambda _: "brave.exe"))
    monkeypatch.setattr(subprocess, "Popen", lambda args, shell: launched.append((args, shell)))

    result = OpenApplicationTool().call({"name": "brave"})

    assert result.success is True
    assert launched == [(["brave.exe"], False)]


def test_unknown_application_is_rejected() -> None:
    result = OpenApplicationTool().call({"name": "powershell"})

    assert result.success is False
    assert "not allowlisted" in result.error


def test_executable_paths_are_not_accepted() -> None:
    result = OpenApplicationTool().call({"name": "C:\\Windows\\System32\\cmd.exe"})

    assert result.success is False
    assert "not allowlisted" in result.error
