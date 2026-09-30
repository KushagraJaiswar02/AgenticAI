"""Small, policy-enforced browser tools for Phase 2 V0.1."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus, urlparse

from pydantic import BaseModel, Field, model_validator

from app.browser_context import BrowserContext
from app.policy import RiskLevel
from app.tools import Tool, ToolExecutionResult


class OpenBrowserInput(BaseModel):
    browser: str = Field(default="brave", description="Allowlisted browser to open. Use 'brave'.")


class OpenUrlInput(BaseModel):
    url: str = Field(description="An absolute HTTP or HTTPS URL to open. Use the parameter name 'url' exactly.")


class SearchWebInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=300,
        description="Non-empty search terms, up to 300 characters. Use the parameter name 'query' exactly.",
    )
    engine: str = Field(
        default="google",
        description="Search engine, either 'google' or 'duckduckgo'. Use the parameter name 'engine' exactly.",
    )

    @model_validator(mode="after")
    def require_non_whitespace_query(self) -> "SearchWebInput":
        if not self.query.strip():
            raise ValueError("Search query must not be empty")
        return self


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


def normalize_url(value: str) -> str:
    """Add HTTPS only to scheme-less dotted hostnames."""
    parsed = urlparse(value)
    if not any(character.isspace() for character in value) and not parsed.scheme and "." in value:
        return f"https://{value}"
    return value


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
        raw_url = arguments.url if isinstance(arguments, OpenUrlInput) else str(arguments["url"])
        url = normalize_url(raw_url)
        if not _valid_url(url):
            data = {"status": "invalid_url", "error": "URL must be an absolute HTTP or HTTPS URL without credentials"}
            return ToolExecutionResult(False, data=data, error=data["error"])
        data = self.context.open_url(url)
        return ToolExecutionResult(data.get("status") == "navigated", data=data, error=data.get("error"))


class SearchWebTool(BrowserTool):
    name = "search_web"
    description = "Search the web with Google or DuckDuckGo using the controlled browser."
    input_model = SearchWebInput
    risk_level = RiskLevel.LOW
    _SEARCH_URLS = {
        "google": "https://www.google.com/search?q={}",
        "duckduckgo": "https://duckduckgo.com/?q={}",
    }

    def execute(self, arguments: SearchWebInput | dict[str, Any]) -> ToolExecutionResult:
        query = arguments.query if isinstance(arguments, SearchWebInput) else str(arguments["query"])
        engine = arguments.engine if isinstance(arguments, SearchWebInput) else str(arguments.get("engine", "google"))
        template = self._SEARCH_URLS.get(engine)
        if template is None:
            error = "Search engine must be 'google' or 'duckduckgo'"
            data = {"status": "invalid_engine", "error": error}
            return ToolExecutionResult(False, data=data, error=error)
        data = self.context.open_url(template.format(quote_plus(query)))
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
    return [OpenBrowserTool(context), OpenUrlTool(context), GetCurrentPageTool(context), SearchWebTool(context)]
