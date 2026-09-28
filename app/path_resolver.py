"""Deterministic path interpretation, kept separate from filesystem security."""

from __future__ import annotations

from pathlib import Path

from app.filesystem_context import FilesystemContext


class PathResolver:
    def __init__(self, context: FilesystemContext) -> None:
        self.context = context

    def resolve(self, value: str | Path) -> Path:
        raw = str(value).strip()
        key = " ".join(raw.lower().split())
        semantic = " ".join(part for part in key.split() if part not in {"my", "the"})
        if semantic in {"workspace folder"}:
            semantic = "workspace"
        if semantic in {"project folder"}:
            semantic = "project"
        if semantic in {"workspace", "current workspace"}:
            return self.context.workspace_root
        if semantic in {"project", "jarvis project"}:
            return self.context.project_root
        if semantic in {"current", "current directory", "here", "this folder"}:
            return self.context.current_directory
        if semantic in {"this project"}:
            return self.context.project_root
        if semantic in {"file i just created", "file we just created"}:
            return self.context.conversational_file()
        if semantic in {"this file", "that file", "it", "file", "previous file", "file we just renamed"}:
            return self.context.conversational_path(kind="file")
        if semantic in {"this folder", "that folder", "it folder", "previously created directory", "previously created folder", "folder i just created", "folder we just created"}:
            return self.context.conversational_path(kind="directory")

        compact = key.replace("\\", "/").lstrip("/")
        if compact in {"jarvis/project", "the/jarvis/project"}:
            return self.context.workspace_root

        # Resolve a semantic first path component such as workspace/square.py.
        parts = Path(raw).parts
        if parts and " ".join(part for part in parts[0].lower().split() if part not in {"my", "the"}) in {"workspace", "project"}:
            base = self.context.workspace_root if "workspace" in parts[0].lower() else self.context.project_root
            return base.joinpath(*parts[1:])
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.context.current_directory / candidate
