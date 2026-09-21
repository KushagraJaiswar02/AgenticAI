"""Terminal response layer."""

from __future__ import annotations


def render_response(text: str) -> str:
    """Format a normalized assistant response for terminal output."""
    return f"JARVIS: {text}"
