"""Terminal input normalization."""

from __future__ import annotations


def normalize_input(value: str) -> str:
    """Normalize terminal input without changing its substantive content."""
    return " ".join(value.strip().split())


def read_input(prompt: str = "User: ") -> str:
    """Read one terminal message."""
    return normalize_input(input(prompt))
