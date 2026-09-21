"""Application-level error types."""

from __future__ import annotations


class ApplicationError(RuntimeError):
    """Base class for safe, user-facing application failures."""


class DatabaseError(ApplicationError):
    """Raised when SQLite/SQLAlchemy initialization or access fails."""


class LLMProviderError(ApplicationError):
    """Raised when the configured LLM provider cannot complete a request."""


class InvalidProviderResponseError(ApplicationError):
    """Raised when a provider returns an unexpected response shape."""


class UnexpectedApplicationError(ApplicationError):
    """Raised for unexpected failures that should be reported safely."""
