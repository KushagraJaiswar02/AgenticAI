"""Bounded filesystem tools. No tool accepts arbitrary commands or recursion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.filesystem_security import FilesystemBoundary
from app.filesystem_context import FilesystemContext
from app.path_resolver import PathResolver
from app.tools import Tool, ToolArgumentError, ToolExecutionError, ToolExecutionResult
from app.policy import RiskLevel


class PathInput(BaseModel):
    path: str = Field(description="Absolute or workspace-relative path to the target. Use the parameter name 'path' exactly.")


class CreateFileInput(BaseModel):
    name: str | None = Field(default=None, description="The file name to create, without the workspace or project directory.")
    location: str | None = Field(default=None, description="Semantic location such as workspace, project, current directory, or a previously referenced directory.")
    content: str | None = Field(default=None, description="Optional text content for the new file.")
    path: str | None = Field(default=None, description="An existing path-compatible form; it is still resolved and security-checked by JARVIS.")

    @model_validator(mode="after")
    def require_target(self) -> "CreateFileInput":
        if not self.name and not self.path:
            raise ValueError("create_file requires name/location or path")
        return self


class CreateDirectoryInput(BaseModel):
    name: str | None = Field(default=None, description="The directory name to create, without the workspace or project directory.")
    location: str | None = Field(default=None, description="Semantic parent location such as workspace, project, current directory, or a previously referenced directory.")
    path: str | None = Field(default=None, description="An existing path-compatible form; it is still resolved and security-checked by JARVIS.")

    @model_validator(mode="after")
    def require_target(self) -> "CreateDirectoryInput":
        if not self.name and not self.path:
            raise ValueError("create_directory requires name/location or path")
        return self


class SearchInput(BaseModel):
    root: str | None = Field(default=None, description="Absolute or workspace-relative directory to search.")
    pattern: str = Field(description="File-name pattern to search for.")


class ChangeDirectoryInput(BaseModel):
    location: str | None = Field(default=None, description="Directory to make the conversational working directory.")
    path: str | None = Field(default=None, description="Compatibility path for the directory.")

    @model_validator(mode="after")
    def require_target(self) -> "ChangeDirectoryInput":
        if not self.location and not self.path:
            raise ValueError("change_directory requires location or path")
        return self


class FilePairInput(BaseModel):
    source: str = Field(description="Absolute or workspace-relative source file path.")
    destination: str = Field(description="Absolute or workspace-relative destination file path.")


class WriteInput(PathInput):
    content: str = Field(description="Text content to write to the target file.")


class FilesystemTool(Tool):
    boundary: FilesystemBoundary

    def __init__(self, boundary: FilesystemBoundary, context: FilesystemContext | None = None) -> None:
        self.boundary = boundary
        self.context = context or FilesystemContext.create(boundary.roots[0], boundary)
        self.resolver = PathResolver(self.context)

    def _path(self, value: str, *, exists: bool = False) -> Path:
        return self.boundary.resolve(self.resolver.resolve(value), must_exist=exists)

    def normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if normalized.get("name"):
            location = normalized.get("location") or normalized.get("directory") or "current directory"
            location_path = self.resolver.resolve(location)
            semantic_path = location_path / str(normalized["name"]).strip()
            if normalized.get("path"):
                existing_path = self.resolver.resolve(normalized["path"])
                if existing_path != semantic_path:
                    raise ToolArgumentError("Conflicting semantic and path arguments")
            normalized["path"] = str(semantic_path)
        if self.name == "write_file" and normalized.get("path") == normalized.get("content") and self.context.last_created_file:
            normalized["path"] = str(self.context.last_created_file)
        if self.name == "write_file" and normalized.get("path") == normalized.get("content") and not self.context.last_created_file:
            raise ToolArgumentError("Please specify which file to write; there is no recent file in context")
        for key in ("path", "root", "source", "destination"):
            if key in normalized:
                normalized[key] = str(self._path(normalized[key]))
        return normalized


class ChangeDirectoryTool(FilesystemTool):
    name = "change_directory"
    description = "Change the conversational working directory inside the configured workspace."
    input_model = ChangeDirectoryInput
    risk_level = RiskLevel.LOW

    def normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        value = normalized.get("location") or normalized.get("path")
        if not value:
            raise ToolArgumentError("change_directory requires a location")
        normalized["path"] = str(self.resolver.resolve_directory_reference(value))
        return normalized

    def execute(self, arguments: ChangeDirectoryInput | dict[str, Any]) -> ToolExecutionResult:
        value = arguments.path if isinstance(arguments, ChangeDirectoryInput) else arguments["path"]
        path = self._path(value, exists=True)
        if not path.is_dir():
            raise ToolArgumentError("change_directory requires an existing directory")
        self.context.set_current_directory(path)
        return ToolExecutionResult(True, {"path": str(path), "cwd": str(path), "workspace_root": str(self.context.workspace_root), "status": "changed"})


class ListDirectoryTool(FilesystemTool):
    name = "list_directory"
    description = "List entries in an existing directory. Provide the directory path in the 'path' field."
    input_model = PathInput
    risk_level = RiskLevel.SAFE

    def execute(self, arguments: PathInput | dict[str, Any]) -> ToolExecutionResult:
        path = self._path(arguments.path if isinstance(arguments, PathInput) else arguments["path"], exists=True)
        if not path.is_dir():
            raise ToolArgumentError("Path is not a directory")
        self.context.remember_directory(path)
        return ToolExecutionResult(True, {"path": str(path), "entries": sorted(item.name for item in path.iterdir())})


class SearchFilesTool(FilesystemTool):
    name = "search_files"
    description = "Search file names under an allowed directory without recursive deletion or execution."
    input_model = SearchInput
    risk_level = RiskLevel.SAFE

    def execute(self, arguments: SearchInput | dict[str, Any]) -> ToolExecutionResult:
        raw_root = arguments.root if isinstance(arguments, SearchInput) else arguments.get("root")
        if not raw_root:
            raw_root = str(self.context.search_root())
        root = self._path(raw_root, exists=True)
        pattern = arguments.pattern if isinstance(arguments, SearchInput) else arguments["pattern"]
        if not root.is_dir() or not pattern or len(pattern) > 200:
            raise ToolArgumentError("Search requires a directory and a bounded pattern")
        matches = [
            {"name": item.name, "path": str(self.boundary.resolve(item)), "is_file": True}
            for item in root.rglob(pattern)
            if item.is_file() and self.boundary.resolve(item).is_file()
        ]
        self.context.remember_directory(root)
        return ToolExecutionResult(True, {"root": str(root), "matches": matches[:1000]})


class ReadFileTool(FilesystemTool):
    name = "read_file"
    description = "Read text from an existing file. Provide the file path in the 'path' field."
    input_model = PathInput
    risk_level = RiskLevel.SAFE

    def execute(self, arguments: PathInput | dict[str, Any]) -> ToolExecutionResult:
        path = self._path(arguments.path if isinstance(arguments, PathInput) else arguments["path"], exists=True)
        if not path.is_file() or path.stat().st_size > 1_000_000:
            raise ToolArgumentError("Only files up to 1 MB can be read")
        try:
            self.context.remember_file(path, read=True)
            return ToolExecutionResult(True, {"path": str(path), "content": path.read_text(encoding="utf-8")})
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolExecutionError("Unable to read the requested text file") from exc


class CreateDirectoryTool(FilesystemTool):
    name = "create_directory"
    description = "Create one directory using a semantic name and location, or a compatibility path."
    input_model = CreateDirectoryInput
    risk_level = RiskLevel.LOW

    def execute(self, arguments: CreateDirectoryInput | dict[str, Any]) -> ToolExecutionResult:
        path = self._path(arguments.path if isinstance(arguments, CreateDirectoryInput) else arguments["path"])
        path.mkdir(exist_ok=False)
        self.context.remember_directory(path, created=True)
        return ToolExecutionResult(True, {"path": str(path), "status": "created"})


class CreateFileTool(FilesystemTool):
    name = "create_file"
    description = "Create a file using name, semantic location, and optional content; do not construct an absolute path."
    input_model = CreateFileInput
    risk_level = RiskLevel.LOW

    def execute(self, arguments: CreateFileInput | dict[str, Any]) -> ToolExecutionResult:
        path = self._path(arguments.path if isinstance(arguments, CreateFileInput) else arguments["path"])
        content = arguments.content if isinstance(arguments, CreateFileInput) else arguments.get("content")
        try:
            if content is None:
                path.touch(exist_ok=False)
            else:
                path.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            return ToolExecutionResult(False, data=None, error=f"Unable to create file '{path}': {exc}")
        if not path.exists() or not path.is_file():
            return ToolExecutionResult(False, data=None, error=f"File creation could not be verified at '{path}'")
        self.context.remember_file(path, created=True)
        return ToolExecutionResult(True, {"path": str(path), "status": "created", "created": True, "verified": True})


class WriteFileTool(FilesystemTool):
    name = "write_file"
    description = "Overwrite or create a bounded text file inside an allowed root."
    input_model = WriteInput
    risk_level = RiskLevel.CONFIRM

    def preflight(self, arguments: dict[str, Any]) -> None:
        path = self._path(arguments["path"])
        if not path.parent.is_dir():
            raise ToolArgumentError("The destination parent directory does not exist")

    def confirmation_prompt(self, arguments: dict[str, Any]) -> str:
        return f"This will write to:\n    {arguments['path']}\n\nContent:\n    {arguments['content']}\n\nProceed? [y/N]"

    def execute(self, arguments: WriteInput | dict[str, Any]) -> ToolExecutionResult:
        path = self._path(arguments.path if isinstance(arguments, WriteInput) else arguments["path"])
        content = arguments.content if isinstance(arguments, WriteInput) else arguments["content"]
        if len(content.encode("utf-8")) > 1_000_000:
            raise ToolArgumentError("Files are limited to 1 MB")
        path.write_text(content, encoding="utf-8")
        self.context.remember_file(path, modified=True)
        return ToolExecutionResult(True, {"path": str(path), "status": "written"})


class CopyFileTool(FilesystemTool):
    name = "copy_file"
    description = "Copy one file to another path inside an allowed root."
    input_model = FilePairInput
    risk_level = RiskLevel.LOW

    def execute(self, arguments: FilePairInput | dict[str, Any]) -> ToolExecutionResult:
        source = arguments.source if isinstance(arguments, FilePairInput) else arguments["source"]
        destination = arguments.destination if isinstance(arguments, FilePairInput) else arguments["destination"]
        source_path = self._path(source, exists=True)
        destination_path = self._path(destination)
        if not source_path.is_file() or destination_path.exists():
            raise ToolArgumentError("Copy requires an existing source file and a new destination")
        shutil.copyfile(source_path, destination_path)
        self.context.remember_file(destination_path, created=True)
        return ToolExecutionResult(True, {"source": str(source_path), "destination": str(destination_path), "status": "copied"})


class MoveFileTool(FilesystemTool):
    name = "move_file"
    description = "Move one file inside an allowed root."
    input_model = FilePairInput
    risk_level = RiskLevel.CONFIRM

    def preflight(self, arguments: dict[str, Any]) -> None:
        self._path(arguments["source"], exists=True)
        if Path(arguments["destination"]).exists():
            raise ToolArgumentError("The destination already exists")

    def execute(self, arguments: FilePairInput | dict[str, Any]) -> ToolExecutionResult:
        source = arguments.source if isinstance(arguments, FilePairInput) else arguments["source"]
        destination = arguments.destination if isinstance(arguments, FilePairInput) else arguments["destination"]
        source_path = self._path(source, exists=True)
        destination_path = self._path(destination)
        if not source_path.is_file() or destination_path.exists():
            raise ToolArgumentError("Move requires an existing source file and a new destination")
        shutil.move(source_path, destination_path)
        self.context.forget_file(source_path)
        self.context.remember_file(destination_path, modified=True)
        return ToolExecutionResult(True, {"source": str(source_path), "destination": str(destination_path), "status": "moved"})


class RenameFileTool(MoveFileTool):
    name = "rename_file"
    description = "Rename one file inside an allowed root."


class DeleteFileTool(FilesystemTool):
    name = "delete_file"
    description = "Delete one file inside an allowed root; directories are never recursively deleted."
    input_model = PathInput
    risk_level = RiskLevel.CONFIRM

    def preflight(self, arguments: dict[str, Any]) -> None:
        self._path(arguments["path"], exists=True)

    def confirmation_prompt(self, arguments: dict[str, Any]) -> str:
        return f"This will permanently delete:\n    {arguments['path']}\n\nProceed? [y/N]"

    def execute(self, arguments: PathInput | dict[str, Any]) -> ToolExecutionResult:
        path = self._path(arguments.path if isinstance(arguments, PathInput) else arguments["path"], exists=True)
        if not path.is_file():
            raise ToolArgumentError("delete_file only accepts a regular file")
        path.unlink()
        self.context.forget_file(path)
        return ToolExecutionResult(True, {"path": str(path), "status": "deleted"})


def create_filesystem_tools(boundary: FilesystemBoundary, context: FilesystemContext | None = None) -> list[Tool]:
    return [
        ChangeDirectoryTool(boundary, context), ListDirectoryTool(boundary, context), SearchFilesTool(boundary, context), ReadFileTool(boundary, context),
        CreateDirectoryTool(boundary, context), CreateFileTool(boundary, context), WriteFileTool(boundary, context),
        CopyFileTool(boundary, context), MoveFileTool(boundary, context), RenameFileTool(boundary, context), DeleteFileTool(boundary, context),
    ]
