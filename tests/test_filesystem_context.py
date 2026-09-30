from pathlib import Path

import pytest

from app.filesystem_context import AmbiguousFilesystemReferenceError, FilesystemContext
from app.filesystem_security import FilesystemBoundary, FilesystemBoundaryError
from app.filesystem_tools import DeleteFileTool, CreateFileTool, ReadFileTool, RenameFileTool, WriteFileTool, create_filesystem_tools
from app.path_resolver import PathResolver
from app.policy import ConfirmationRequiredError
from app.session import ConversationState
from app.llm import ToolCall
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
    context.last_selected_file = None
    context.last_created_file = None
    context.last_modified_file = None
    context.last_read_file = None
    with pytest.raises(AmbiguousFilesystemReferenceError):
        PathResolver(context).resolve("that file")


def test_authoritative_last_file_wins_over_older_recent_files(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    registry.execute("create_file", {"path": "alpha.txt"})
    registry.execute("create_file", {"path": "beta.txt"})

    with pytest.raises(ConfirmationRequiredError) as pending:
        registry.execute("write_file", {"path": "it", "content": "BETA"})
    registry.execute("write_file", {"path": "it", "content": "BETA"}, confirmation_request=pending.value.request, confirmation_response="yes")
    read = registry.execute("read_file", {"path": "it"})

    assert Path(read.data["path"]).name == "beta.txt"
    assert read.data["content"] == "BETA"


def test_create_directory_and_list_do_not_change_cwd(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    original = context.cwd

    registry.execute("create_directory", {"name": "TestFolder"})
    registry.execute("list_directory", {"path": "TestFolder"})

    assert context.cwd == original


def test_windows_target_after_preserves_case_and_backslashes():
    prompt = r"read ARCHITECTURE.md from C:\PROJECTSPACE\WorkSpace\JARVIS\docs"
    tokens = ConversationState._tokens(prompt)
    assert ConversationState._target_after(tokens, {"from"}) == r"C:\PROJECTSPACE\WorkSpace\JARVIS\docs"


def test_read_absolute_llm_path_is_not_overwritten_by_prompt_directory():
    directory = r"C:\PROJECTSPACE\WorkSpace\JARVIS\docs"
    absolute = directory + r"\ARCHITECTURE.md"
    state = ConversationState()
    result = state.prepare_tool_call(ToolCall("read_file", {"path": absolute}), f"read ARCHITECTURE.md from {directory}")
    assert result.arguments["path"] == absolute


def test_read_bare_llm_name_is_joined_to_explicit_directory():
    directory = r"C:\PROJECTSPACE\WorkSpace\JARVIS\docs"
    state = ConversationState()
    result = state.prepare_tool_call(ToolCall("read_file", {"path": "ARCHITECTURE.md"}), f"read ARCHITECTURE.md from {directory}")
    assert result.arguments["path"] == str(Path(directory) / "ARCHITECTURE.md")


def test_read_bare_name_join_rejects_traversal_and_drive_paths():
    directory = r"C:\PROJECTSPACE\WorkSpace\JARVIS\docs"
    state = ConversationState()
    for name in (r"..\..\secret.txt", r"D:\x.txt"):
        result = state.prepare_tool_call(ToolCall("read_file", {"path": name}), f"read file from {directory}")
        assert result.arguments["path"] == name


def test_rename_destination_preserves_case():
    state = ConversationState()
    result = state.prepare_tool_call(ToolCall("rename_file", {"source": "a.txt", "destination": "wrong.md"}), "rename a.txt to Report.md")
    assert result.arguments["destination"] == "Report.md"


def test_read_it_still_resolves_from_filesystem_context(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    target = context.workspace_root / "created.txt"
    target.write_text("content", encoding="utf-8")
    context.remember_file(target, created=True)
    state = ConversationState(filesystem_context=context)
    result = state.prepare_tool_call(ToolCall("read_file", {"path": "placeholder.txt"}), "read it")
    assert PathResolver(context).resolve(result.arguments["path"]) == target


def test_quoted_directory_with_spaces_is_preserved():
    directory = r"C:\My Docs\x"
    state = ConversationState()
    result = state.prepare_tool_call(ToolCall("read_file", {"path": "notes.txt"}), r'read "notes.txt" from "C:\My Docs\x"')
    assert result.arguments["path"] == str(Path(directory) / "notes.txt")


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


def test_relative_create_and_read_share_workspace_base_path(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))

    created = registry.execute("create_file", {"name": "encore.txt", "content": "hello"})
    read = registry.execute("read_file", {"path": "encore.txt"})

    assert Path(created.data["path"]) == context.workspace_root / "encore.txt"
    assert Path(read.data["path"]) == Path(created.data["path"])


def test_explicit_project_location_updates_shared_base_path(tmp_path: Path):
    workspace = tmp_path / "WorkSpace"
    project = workspace / "JARVIS"
    project.mkdir(parents=True)
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary, project)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))

    result = registry.execute("create_file", {"name": "test.py", "location": "project"})

    assert Path(result.data["path"]) == project / "test.py"
    assert context.current_directory == workspace


def test_base_path_update_cannot_escape_workspace(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    with pytest.raises(ValueError):
        context.set_current_directory(tmp_path.parent)


def test_change_directory_updates_cwd_without_changing_workspace_root(tmp_path: Path):
    workspace = tmp_path / "WorkSpace"
    project = workspace / "JARVIS"
    project.mkdir(parents=True)
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary, project)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))

    result = registry.execute("change_directory", {"location": "JARVIS"})

    assert result.success is True
    assert context.cwd == project
    assert context.workspace_root == workspace


def test_change_directory_rejects_outside_workspace_and_preserves_cwd(tmp_path: Path):
    workspace = tmp_path / "WorkSpace"
    project = workspace / "JARVIS"
    project.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary, project)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))

    with pytest.raises(Exception):
        registry.execute("change_directory", {"path": str(outside)})

    assert context.cwd == workspace


def test_change_directory_reports_ambiguous_directory_names(tmp_path: Path):
    workspace = tmp_path / "WorkSpace"
    (workspace / "one" / "JARVIS").mkdir(parents=True)
    (workspace / "two" / "JARVIS").mkdir(parents=True)
    boundary = FilesystemBoundary([workspace])
    context = FilesystemContext.create(workspace, boundary)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))

    with pytest.raises(AmbiguousFilesystemReferenceError):
        registry.execute("change_directory", {"location": "JARVIS"})

    assert context.cwd == workspace


def test_conflicting_create_file_arguments_are_rejected(tmp_path: Path):
    boundary, context = make_context(tmp_path)
    registry = ToolRegistry(create_filesystem_tools(boundary, context))
    with pytest.raises(Exception, match="Conflicting"):
        registry.execute("create_file", {"name": "square.py", "location": "workspace", "path": "project/other.py"})
