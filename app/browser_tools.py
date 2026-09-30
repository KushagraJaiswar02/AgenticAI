"""Small, policy-enforced browser tools for Phase 2 V0.1."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.browser_context import BrowserContext
from app.policy import RiskLevel
from app.tools import Tool, ToolExecutionResult


class OpenBrowserInput(BaseModel):
    browser: str = Field(default="brave", description="Allowlisted browser to open. Use 'brave'.")


class OpenUrlInput(BaseModel):
    url: str = Field(description="An absolute HTTP or HTTPS URL to open.")


class EmptyBrowserInput(BaseModel):
    """Input schema for browser state queries."""


def _valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        not any(character.isspace() for character in value)
        and parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
    )


class BrowserTool(Tool):
    context: BrowserContext

    def __init__(self, context: BrowserContext) -> None:
        self.context = context


class OpenBrowserTool(BrowserTool):
    name = "open_browser"
    description = "Open the allowlisted Brave browser for controlled navigation."
    input_model = OpenBrowserInput
    risk_level = RiskLevel.LOW

    def execute(self, arguments: OpenBrowserInput | dict[str, Any]) -> ToolExecutionResult:
        browser = arguments.browser if isinstance(arguments, OpenBrowserInput) else arguments.get("browser", "brave")
        if browser != "brave":
            data = {"status": "failed", "error": f"Browser '{browser}' is not allowlisted"}
        else:
            data = self.context.open_browser(browser)
        return ToolExecutionResult(data.get("status") == "opened", data=data, error=data.get("error"))


class OpenUrlTool(BrowserTool):
    name = "open_url"
    description = "Navigate the controlled browser to an absolute HTTP or HTTPS URL."
    input_model = OpenUrlInput
    risk_level = RiskLevel.LOW

    def execute(self, arguments: OpenUrlInput | dict[str, Any]) -> ToolExecutionResult:
        url = arguments.url if isinstance(arguments, OpenUrlInput) else str(arguments["url"])
        if not _valid_url(url):
            data = {"status": "invalid_url", "error": "URL must be an absolute HTTP or HTTPS URL without credentials"}
            return ToolExecutionResult(False, data=data, error=data["error"])
        data = self.context.open_url(url)
        return ToolExecutionResult(data.get("status") == "navigated", data=data, error=data.get("error"))


class GetCurrentPageTool(BrowserTool):
    name = "get_current_page"
    description = "Return the current controlled browser page URL and title."
    input_model = EmptyBrowserInput
    risk_level = RiskLevel.SAFE

    def execute(self, arguments: EmptyBrowserInput | dict[str, Any]) -> ToolExecutionResult:
        del arguments
        data = self.context.current_page()
        return ToolExecutionResult(data.get("status") == "available", data=data, error=data.get("error"))


def create_browser_tools(context: BrowserContext | None = None) -> list[Tool]:
    context = context or BrowserContext()
    return [OpenBrowserTool(context), OpenUrlTool(context), GetCurrentPageTool(context)]
