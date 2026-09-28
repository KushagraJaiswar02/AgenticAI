from pathlib import Path

import pytest

from app.filesystem_context import AmbiguousFilesystemReferenceError, FilesystemContext
from app.filesystem_security import FilesystemBoundary, FilesystemBoundaryError
from app.filesystem_tools import DeleteFileTool, CreateFileTool, ReadFileTool, RenameFileTool, WriteFileTool, create_filesystem_tools
from app.path_resolver import PathResolver
from app.policy import ConfirmationRequiredError
from app.tools import ToolRegistry


def make_context(tmp_path: Path) -> tuple[FilesystemBoundary, FilesystemContext]:
    workspace = tmp_path / "JARVIS"
    workspace.mkdir()
    boundary = FilesystemBoundary([workspace])
    return boundary, FilesystemContext.create(workspace, boundary)


def test_workspace_and_natural_references_resolve_inside_allowlist(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    resolver = PathResolver(context)
    assert resolver.resolve("workspace") == context.current_workspace
    assert resolver.resolve("JARVIS project") == context.current_workspace
    assert resolver.resolve("/JARVIS/project") == context.current_workspace
    assert resolver.resolve("current directory") == context.current_directory
    assert resolver.resolve("hello.txt") == context.current_workspace / "hello.txt"
    assert resolver.resolve("Workspace") == context.workspace_root
    assert resolver.resolve("my workspace folder") == context.workspace_root
    assert resolver.resolve("project") == context.project_root
    assert resolver.resolve("workspace/square.py") == context.workspace_root / "square.py"
    with pytest.raises(FilesystemBoundaryError):
        boundary.resolve(resolver.resolve(Path("..") / "outside.txt"))


def test_create_then_write_content_uses_last_created_file(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    created = registry.execute("create_file", {"path": "hello.txt"})
    target = Path(created.data["path"])
    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("write_file", {"path": "GODDAMIT", "content": "GODDAMIT"})
    request = pending.value.request
    assert str(target) in request.prompt
    registry.execute("write_file", {"path": "GODDAMIT", "content": "GODDAMIT"}, confirmation_request=request, confirmation_response="yes")
    assert target.read_text(encoding="utf-8") == "GODDAMIT"
    assert context.last_modified_file == target
    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("delete_file", {"path": "that file"})
    registry.execute("delete_file", {"path": "that file"}, confirmation_request=pending.value.request, confirmation_response="yes")
    assert not target.exists()


def test_read_rename_move_and_delete_update_context(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    target = context.current_workspace / "hello.txt"
    target.write_text("hello", encoding="utf-8")
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    registry.execute("read_file", {"path": "hello.txt"})
    assert context.last_read_file == target
    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("rename_file", {"source": "hello.txt", "destination": "renamed.txt"})
    registry.execute("rename_file", {"source": "hello.txt", "destination": "renamed.txt"}, confirmation_request=pending.value.request, confirmation_response="yes")
    renamed = context.current_workspace / "renamed.txt"
    assert context.last_modified_file == renamed
    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("move_file", {"source": "renamed.txt", "destination": "moved.txt"})
    registry.execute("move_file", {"source": "renamed.txt", "destination": "moved.txt"}, confirmation_request=pending.value.request, confirmation_response="yes")
    moved = context.current_workspace / "moved.txt"
    assert context.last_modified_file == moved
    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("delete_file", {"path": "moved.txt"})
    registry.execute("delete_file", {"path": "moved.txt"}, confirmation_request=pending.value.request, confirmation_response="yes")
    assert context.last_selected_file is None


def test_ambiguous_recent_files_are_not_guessed(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry([CreateFileTool(boundary, context)])
    registry.execute("create_file", {"path": "a.txt"})
    registry.execute("create_file", {"path": "b.txt"})
    with pytest.raises(AmbiguousFilesystemReferenceError):
        PathResolver(context).resolve("that file")


def test_rejected_write_does_not_update_context(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    registry.execute("create_file", {"path": "green.txt"})
    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("write_file", {"path": "green.txt", "content": "new content"})
    with pytest.raises(Exception):
        registry.execute("write_file", {"path": "green.txt", "content": "new content"}, confirmation_request=pending.value.request, confirmation_response="no")
    assert context.last_modified_file is None
    assert context.last_created_file == context.current_workspace / "green.txt"


def test_directory_context_is_available_for_nested_file_creation(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    registry.execute("create_directory", {"name": "experiments", "location": "workspace"})
    result = registry.execute("create_file", {"name": "hello.py", "location": "that folder"})
    assert Path(result.data["path"]) == context.workspace_root / "experiments" / "hello.py"


def test_create_file_schema_preserves_name_and_location_and_resolves_workspace(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    definition = next(item for item in registry.definitions() if item.name == "create_file")
    assert {"name", "location", "content", "path"} <= set(definition.parameters["properties"])
    result = registry.execute("create_file", {"name": "square.py", "location": "workspace"})
    assert Path(result.data["path"]) == context.workspace_root / "square.py"


def test_create_file_physically_verifies_content_and_context(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    content = "def square(x):\n    return x * x\n"

    result = registry.execute("create_file", {"name": "square.py", "location": "workspace", "content": content})

    target = context.workspace_root / "square.py"
    assert result.success is True
    assert result.data == {"path": str(target), "status": "created", "created": True, "verified": True}
    assert target.exists() and target.is_file()
    assert target.read_text(encoding="utf-8") == content
    assert context.last_created_path == target
    assert context.last_created_directory == context.workspace_root
    assert context.last_created_location == "workspace"


def test_search_uses_workspace_context_and_returns_canonical_matches(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    target = context.workspace_root / "square.py"
    target.write_text("square", encoding="utf-8")
    context.remember_file(target, created=True)

    result = registry.execute("search_files", {"root": str(context.search_root()), "pattern": "square.py"})

    assert result.data["root"] == str(context.workspace_root)
    assert result.data["matches"] == [{"name": "square.py", "path": str(target), "is_file": True}]


def test_last_created_file_resolves_without_search(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    target = context.workspace_root / "square.py"
    target.write_text("square", encoding="utf-8")
    context.remember_file(target, created=True)

    assert PathResolver(context).resolve("file I just created") == target


def test_conflicting_create_file_arguments_are_rejected(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    with pytest.raises(Exception, match="Conflicting"):
        registry.execute("create_file", {"name": "square.py", "location": "workspace", "path": "project/other.py"})
