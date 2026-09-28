from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from app.audit import AuditLogger
from app.confirmation import ConfirmationManager
from app.filesystem_security import FilesystemBoundary, FilesystemBoundaryError
from app.filesystem_tools import DeleteFileTool, ReadFileTool
from app.policy import ConfirmationRequiredError, PolicyDeniedError, RiskLevel
from app.tools import Tool, ToolExecutionResult, ToolRegistry


class Empty(BaseModel):
    pass


class ConfirmedTool(Tool):
    name = "confirmed"
    description = "A test consequential action."
    input_model = Empty
    risk_level = RiskLevel.CONFIRM

    def __init__(self):
        self.calls = 0

    def execute(self, arguments):
        self.calls += 1
        return ToolExecutionResult(True, {"called": self.calls})


class ForbiddenTool(ConfirmedTool):
    name = "forbidden"
    risk_level = RiskLevel.FORBIDDEN


def test_safe_tools_are_allowed_and_metadata_is_explicit():
    from app.tools import TimeTool
    registry = ToolRegistry([TimeTool()])
    assert registry.definitions()[0].risk_level == "SAFE"
    assert registry.execute("time", {}).success


def test_confirmed_tool_cannot_execute_until_exact_request_is_approved():
    tool = ConfirmedTool()
    registry = ToolRegistry([tool])
    with pytest.raises(ConfirmationRequiredError) as caught:
        registry.execute("confirmed", {})
    request = caught.value.request
    with pytest.raises(PolicyDeniedError):
        registry.execute("confirmed", {}, confirmation_request=request, confirmation_response="no")
    assert tool.calls == 0
    with pytest.raises(ConfirmationRequiredError) as caught:
        registry.execute("confirmed", {})
    request = caught.value.request
    assert registry.execute("confirmed", {}, confirmation_request=request, confirmation_response="yes").success
    assert tool.calls == 1
    with pytest.raises(PolicyDeniedError):
        registry.execute("confirmed", {}, confirmation_request=request, confirmation_response="yes")


def test_confirmation_cannot_be_reused_for_modified_arguments():
    tool = ConfirmedTool()
    registry = ToolRegistry([tool])
    with pytest.raises(ConfirmationRequiredError) as caught:
        registry.execute("confirmed", {})
    with pytest.raises(PolicyDeniedError):
        registry.execute("confirmed", {"unexpected": "change"}, confirmation_request=caught.value.request, confirmation_response="yes")
    assert tool.calls == 0


def test_forbidden_is_denied_without_confirmation():
    tool = ForbiddenTool()
    with pytest.raises(PolicyDeniedError, match="forbidden"):
        ToolRegistry([tool]).execute("forbidden", {})
    assert tool.calls == 0


def test_high_confirmation_requires_strong_phrase():
    manager = ConfirmationManager()
    request = manager.create_request("x", {}, "a", RiskLevel.HIGH, strong=True)
    assert not manager.approve(request, "yes", tool_name="x", arguments={}, action_id="a", risk_level=RiskLevel.HIGH)
    request = manager.create_request("x", {}, "b", RiskLevel.HIGH, strong=True)
    assert manager.approve(request, "CONFIRM DELETE", tool_name="x", arguments={}, action_id="b", risk_level=RiskLevel.HIGH)


def test_confirmation_expires():
    now = [100.0]
    manager = ConfirmationManager(ttl_seconds=5, clock=lambda: now[0])
    request = manager.create_request("x", {}, "a", RiskLevel.CONFIRM)
    now[0] = 106.0
    assert not manager.approve(request, "yes", tool_name="x", arguments={}, action_id="a", risk_level=RiskLevel.CONFIRM)


def test_filesystem_boundary_blocks_traversal_and_outside_paths(tmp_path: Path):
    root = tmp_path / "allowed"
    root.mkdir()
    boundary = FilesystemBoundary([root])
    assert boundary.resolve(".") == root
    with pytest.raises(FilesystemBoundaryError):
        boundary.resolve(root / ".." / "outside.txt")


def test_filesystem_tools_use_boundary_and_confirmation(tmp_path: Path):
    root = tmp_path / "allowed"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("hello", encoding="utf-8")
    boundary = FilesystemBoundary([root])
    registry = ToolRegistry([ReadFileTool(boundary), DeleteFileTool(boundary)])
    assert registry.execute("read_file", {"path": str(target)}).data["content"] == "hello"
    with pytest.raises(ConfirmationRequiredError) as caught:
        registry.execute("delete_file", {"path": str(target)})
    assert target.exists()
    registry.execute("delete_file", {"path": str(target)}, confirmation_request=caught.value.request, confirmation_response="yes")
    assert not target.exists()


def test_audit_records_redact_secrets():
    audit = AuditLogger()
    audit.record(request_id="r1", tool_name="x", arguments={"api_key": "secret"}, execution_result="denied")
    assert audit.records[0]["arguments"]["api_key"] == "[REDACTED]"


def test_registry_returns_canonical_result_and_audits_failed_execution(tmp_path):
    from app.filesystem_context import FilesystemContext
    from app.filesystem_security import FilesystemBoundary
    from app.filesystem_tools import CreateFileTool

    boundary = FilesystemBoundary([tmp_path])
    context = FilesystemContext.create(tmp_path, boundary)
    audit = AuditLogger()
    registry = ToolRegistry([CreateFileTool(boundary, context)], audit_logger=audit)
    first = registry.execute("create_file", {"path": "created.txt"})
    assert first.success is True
    assert first.tool_name == "create_file"
    assert first.result["status"] == "created"
    failed = registry.execute("create_file", {"path": "created.txt"})
    assert failed.success is False
    assert failed.result is None
    assert failed.error
    assert audit.records[-1]["execution_result"] == "failed"
