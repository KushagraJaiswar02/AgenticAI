"""Exact in-memory filesystem state for one JARVIS session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.filesystem_security import FilesystemBoundary
from app.errors import ApplicationError


class AmbiguousFilesystemReferenceError(ApplicationError):
    """Raised when a conversational reference has multiple plausible targets."""


class MissingFilesystemReferenceError(ApplicationError):
    """Raised when a conversational file reference has no target."""


@dataclass
class FilesystemContext:
    workspace_root: Path
    project_root: Path
    current_directory: Path
    recent_files: list[Path] = field(default_factory=list)
    recent_directories: list[Path] = field(default_factory=list)
    last_created_file: Path | None = None
    last_read_file: Path | None = None
    last_modified_file: Path | None = None
    last_selected_file: Path | None = None
    last_created_path: Path | None = None
    last_created_directory: Path | None = None
    last_created_location: str | None = None
    last_relevant_path: Path | None = None
    recent_paths: list[Path] = field(default_factory=list)

    @classmethod
    def create(cls, workspace: str | Path, boundary: FilesystemBoundary, project_root: str | Path | None = None) -> "FilesystemContext":
        workspace_path = boundary.resolve(workspace, must_exist=True)
        project_path = boundary.resolve(project_root or workspace_path, must_exist=True)
        if not workspace_path.is_dir() or not project_path.is_dir():
            raise ApplicationError("JARVIS workspace must be an existing directory")
        return cls(workspace_path, project_path, workspace_path, recent_directories=[workspace_path])

    @property
    def current_workspace(self) -> Path:
        """Backward-compatible alias for the configured workspace root."""
        return self.workspace_root

    @property
    def current_base_path(self) -> Path:
        """Canonical name for the conversational filesystem base path."""
        return self.current_directory

    @property
    def cwd(self) -> Path:
        """Short alias used by filesystem context consumers."""
        return self.current_directory

    @property
    def last_file(self) -> Path | None:
        """Most recently relevant file operation target."""
        return self.last_selected_file or self.last_created_file

    def remember_file(self, path: str | Path, *, created: bool = False, read: bool = False, modified: bool = False) -> None:
        resolved = Path(path)
        self.last_created_path = resolved if created else self.last_created_path
        self.last_created_directory = resolved.parent if created else self.last_created_directory
        self.last_created_location = self.semantic_location(resolved) if created else self.last_created_location
        self.last_relevant_path = resolved
        if resolved not in self.recent_paths:
            self.recent_paths.insert(0, resolved)
        if resolved not in self.recent_files:
            self.recent_files.insert(0, resolved)
        if len(self.recent_files) > 20:
            del self.recent_files[20:]
        if len(self.recent_paths) > 20:
            del self.recent_paths[20:]
        self.last_selected_file = resolved
        if created:
            self.last_created_file = resolved
        if read:
            self.last_read_file = resolved
        if modified:
            self.last_modified_file = resolved

    def remember_directory(self, path: str | Path, *, created: bool = False) -> None:
        resolved = Path(path)
        self.current_directory = resolved
        self.last_created_path = resolved if created else self.last_created_path
        self.last_created_directory = resolved if created else self.last_created_directory
        self.last_created_location = self.semantic_location(resolved) if created else self.last_created_location
        self.last_relevant_path = resolved
        if resolved not in self.recent_paths and created:
            self.recent_paths.insert(0, resolved)
        if resolved not in self.recent_directories:
            self.recent_directories.insert(0, resolved)
        if len(self.recent_directories) > 20:
            del self.recent_directories[20:]

    def forget_file(self, path: str | Path) -> None:
        resolved = Path(path)
        self.recent_paths = [item for item in self.recent_paths if item != resolved]
        self.recent_files = [item for item in self.recent_files if item != resolved]
        for name in ("last_created_file", "last_read_file", "last_modified_file", "last_selected_file"):
            if getattr(self, name) == resolved:
                setattr(self, name, self.recent_files[0] if self.recent_files else None)
        if self.last_created_path == resolved:
            self.last_created_path = self.recent_paths[0] if self.recent_paths else None
            self.last_created_directory = self.last_created_path.parent if self.last_created_path else None
            self.last_created_location = self.semantic_location(self.last_created_path) if self.last_created_path else None
        if self.last_relevant_path == resolved:
            self.last_relevant_path = self.recent_paths[0] if self.recent_paths else None

    def conversational_path(self, *, kind: str | None = None) -> Path:
        candidates = [item for item in self.recent_paths if item.exists()]
        if kind == "file":
            candidates = [item for item in candidates if item.is_file()]
        elif kind == "directory":
            candidates = [item for item in candidates if item.is_dir()]
        if not candidates:
            raise MissingFilesystemReferenceError("Which filesystem object do you mean? I don't have a recent object in context.")
        if len(candidates) > 1:
            names = ", ".join(item.name for item in candidates[:3])
            raise AmbiguousFilesystemReferenceError(f"Which filesystem object do you mean: {names}?")
        return candidates[0]

    def conversational_file(self) -> Path:
        if self.last_created_file is not None and self.last_created_file.exists() and self.last_created_file.is_file():
            return self.last_created_file
        return self.conversational_path(kind="file")

    def search_root(self) -> Path:
        """Choose a bounded deterministic root for an unqualified follow-up search."""
        if self.last_created_directory is not None and self.last_created_directory.exists():
            return self.last_created_directory
        if self.last_relevant_path is not None:
            candidate = self.last_relevant_path if self.last_relevant_path.is_dir() else self.last_relevant_path.parent
            if candidate.exists() and candidate.is_dir():
                return candidate
        return self.workspace_root

    def semantic_location(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == self.workspace_root or self.workspace_root in resolved.parents:
            return "workspace"
        if resolved == self.project_root or self.project_root in resolved.parents:
            return "project"
        return "current directory"

    def set_current_directory(self, path: str | Path) -> Path:
        """Move the conversational base without changing the security boundary."""
        resolved = Path(path).expanduser().resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError("Current directory must remain inside the workspace boundary")
        if not resolved.is_dir():
            raise ValueError("Current directory must exist and be a directory")
        self.current_directory = resolved
        self.last_relevant_path = resolved
        if resolved not in self.recent_directories:
            self.recent_directories.insert(0, resolved)
            del self.recent_directories[20:]
        return resolved
