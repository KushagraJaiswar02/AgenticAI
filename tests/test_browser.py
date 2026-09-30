from typing import Any

import pytest

from app.browser_context import BrowserContext
from app.browser_tools import create_browser_tools
from app.policy import PolicyAction, PolicyDecision, PolicyDeniedError, PolicyEngine, RiskLevel
from app.tools import ToolRegistry


class FakeBrowserBackend:
    def __init__(self, *, open_error: Exception | None = None, navigate_error: Exception | None = None):
        self.open_error = open_error
        self.navigate_error = navigate_error
        self.opened = False
        self.url = ""
        self.title = ""

    def open(self, browser: str) -> None:
        if self.open_error:
            raise self.open_error
        self.opened = True

    def navigate(self, url: str) -> tuple[str, str]:
        if self.navigate_error:
            raise self.navigate_error
        self.url = url
        self.title = "Example title"
        return self.url, self.title

    def current_url(self) -> str:
        return self.url

    def current_title(self) -> str:
        return self.title


class DenyPolicy(PolicyEngine):
    def evaluate(self, tool: Any, arguments: dict[str, Any] | None = None) -> PolicyDecision:
        del arguments
        return PolicyDecision(PolicyAction.DENY, RiskLevel.FORBIDDEN, "test denial")


def make_registry(backend: FakeBrowserBackend | None = None, **kwargs) -> tuple[ToolRegistry, BrowserContext]:
    context = BrowserContext(_backend=backend or FakeBrowserBackend())
    return ToolRegistry(create_browser_tools(context), **kwargs), context


def test_browser_tools_register_with_simple_schemas():
    registry, _ = make_registry()

    assert registry.list() == ["get_current_page", "open_browser", "open_url"]
    definitions = {definition.name: definition for definition in registry.definitions()}
    assert definitions["open_url"].parameters["properties"]["url"]["description"]
    assert definitions["open_browser"].parameters["properties"]["browser"]["description"]


def test_browser_context_has_only_initial_facts():
    context = BrowserContext(_backend=FakeBrowserBackend())

    assert context.browser_running is False
    assert context.current_url is None
    assert context.current_title is None
    assert context.last_opened_url is None


def test_valid_url_navigation_updates_context_and_current_page():
    registry, context = make_registry()

    opened = registry.execute("open_browser", {})
    navigated = registry.execute("open_url", {"url": "https://github.com"})
    current = registry.execute("get_current_page", {})

    assert opened.success
    assert navigated.success
    assert navigated.data == {"status": "navigated", "url": "https://github.com", "title": "Example title"}
    assert current.data == {"status": "available", "url": "https://github.com", "title": "Example title"}
    assert context.last_opened_url == "https://github.com"


@pytest.mark.parametrize("url", ["github.com", "javascript:alert(1)", "https://user:pass@example.com", "https://"])
def test_malformed_url_is_rejected_without_navigation(url: str):
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    registry.execute("open_browser", {})

    result = registry.execute("open_url", {"url": url})

    assert not result.success
    assert result.data["status"] == "invalid_url"
    assert backend.url == ""


def test_browser_execution_failure_is_structured():
    registry, _ = make_registry(FakeBrowserBackend(navigate_error=RuntimeError("network down")))
    registry.execute("open_browser", {})

    result = registry.execute("open_url", {"url": "https://example.com"})

    assert not result.success
    assert result.data["status"] == "failed"
    assert "network down" in result.error


def test_current_page_without_browser_is_structured():
    registry, _ = make_registry()

    result = registry.execute("get_current_page", {})

    assert not result.success
    assert result.data["status"] == "unavailable"


def test_policy_engine_remains_authoritative():
    registry, context = make_registry(policy_engine=DenyPolicy())

    with pytest.raises(PolicyDeniedError):
        registry.execute("open_browser", {})

    assert context.browser_running is False
