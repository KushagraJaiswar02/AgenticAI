"""Configurable filesystem boundary for JARVIS filesystem tools."""

from __future__ import annotations

import os
from pathlib import Path

from app.errors import ApplicationError


class FilesystemBoundaryError(ApplicationError):
    """Raised when a resolved path is outside an allowed root."""


class FilesystemBoundary:
    def __init__(self, allowed_roots: list[str | Path]) -> None:
        if not allowed_roots:
            raise ValueError("At least one filesystem root is required")
        self._roots = tuple(self._real(path) for path in allowed_roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def resolve(self, value: str | Path, *, must_exist: bool = False) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self._roots[0] / candidate
        resolved = self._real(candidate)
        if must_exist and not resolved.exists():
            raise FilesystemBoundaryError(f"Path does not exist: {value}")
        if not any(self._is_within(resolved, root) for root in self._roots):
            raise FilesystemBoundaryError("The requested path is outside JARVIS's allowed filesystem area")
        return resolved

    @staticmethod
    def _real(path: str | Path) -> Path:
        return Path(os.path.realpath(os.path.abspath(os.fspath(path))))

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
