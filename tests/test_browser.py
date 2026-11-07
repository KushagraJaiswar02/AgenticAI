from typing import Any

import pytest

from app.browser_context import BrowserContext
from app.browser_tools import create_browser_tools, normalize_url
from app.fake_provider import FakeLLMProvider
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.policy import PolicyAction, PolicyDecision, PolicyDeniedError, PolicyEngine, RiskLevel
from app.tools import ToolArgumentError, ToolRegistry


class FakeBrowserBackend:
    def __init__(self, *, open_error: Exception | None = None, navigate_error: Exception | None = None):
        self.open_error = open_error
        self.navigate_error = navigate_error
        self.opened = False
        self.open_attempts = 0
        self.navigations = []
        self.url = ""
        self.title = ""

    def open(self, browser: str) -> None:
        self.open_attempts += 1
        if self.open_error:
            raise self.open_error
        self.opened = True

    def navigate(self, url: str) -> tuple[str, str]:
        self.navigations.append(url)
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

    assert registry.list() == ["get_current_page", "get_page_content", "open_browser", "open_url", "search_web"]
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


def test_first_navigation_initializes_browser_and_audits_both_tools():
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    result = registry.execute("open_url", {"url": "https://react.dev"}, request_id="first")
    assert result.success and result.data["url"] == "https://react.dev"
    assert backend.open_attempts == 1 and backend.navigations == ["https://react.dev"]
    assert [(record["tool_name"], record["request_id"]) for record in registry.audit_logger.records] == [
        ("open_browser", "first"), ("open_url", "first"),
    ]


@pytest.mark.parametrize("explicit_open", [False, True])
def test_multiple_navigations_reuse_same_browser(explicit_open):
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    if explicit_open:
        assert registry.execute("open_browser", {}).success
    for url in ("https://react.dev", "https://react.dev/reference/react"):
        assert registry.execute("open_url", {"url": url}).success
    assert backend.open_attempts == 1
    assert backend.navigations == ["https://react.dev", "https://react.dev/reference/react"]


@pytest.mark.parametrize(("name", "arguments"), [
    ("open_url", {"url": "https://react.dev"}),
    ("search_web", {"query": "jarvis"}),
])
def test_browser_launch_failure_stops_navigation_without_retry(name, arguments):
    backend = FakeBrowserBackend(open_error=RuntimeError("raw launch details"))
    registry, context = make_registry(backend)
    result = registry.execute(name, arguments)
    assert not result.success
    assert result.data == {"status": "browser_launch_failed", "error": "Browser could not be opened"}
    assert not context.browser_running
    assert backend.open_attempts == 1 and backend.navigations == []


def test_initial_navigation_failure_is_not_retried_after_opening_browser():
    backend = FakeBrowserBackend(navigate_error=RuntimeError("navigation failed"))
    registry, context = make_registry(backend)
    result = registry.execute("open_url", {"url": "https://react.dev"})
    assert not result.success and result.data["status"] == "failed"
    assert context.browser_running
    assert backend.open_attempts == 1 and backend.navigations == ["https://react.dev"]


@pytest.mark.parametrize("url", ["https://[", "http://127.0.0.1"])
def test_invalid_or_blocked_url_cannot_launch_browser(url):
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    assert not registry.execute("open_url", {"url": url}).success
    assert backend.open_attempts == 0 and backend.navigations == []


@pytest.mark.parametrize(("name", "arguments"), [
    ("open_url", {"url": "https://react.dev"}),
    ("search_web", {"query": "jarvis"}),
])
@pytest.mark.parametrize("deny_launch", [True, False])
def test_initial_navigation_honors_both_tool_policies(name, arguments, deny_launch):
    denied_tool = "open_browser" if deny_launch else name
    class BrowserPolicy(PolicyEngine):
        def evaluate(self, tool, arguments):
            if tool.name == denied_tool:
                return PolicyDecision(PolicyAction.DENY, RiskLevel.FORBIDDEN, "test denial")
            return super().evaluate(tool, arguments)

    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend, policy_engine=BrowserPolicy())
    with pytest.raises(PolicyDeniedError, match="test denial"):
        registry.execute(name, arguments)
    assert backend.open_attempts == 0 and backend.navigations == []


def test_implicit_browser_open_does_not_bypass_confirmation():
    class BrowserPolicy(PolicyEngine):
        def evaluate(self, tool, arguments):
            if tool.name == "open_browser":
                return PolicyDecision(PolicyAction.REQUIRE_CONFIRMATION, RiskLevel.CONFIRM, "test confirmation")
            return super().evaluate(tool, arguments)

    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend, policy_engine=BrowserPolicy())
    result = registry.execute("open_url", {"url": "https://react.dev"})
    assert not result.success and "requires confirmation" in result.error
    assert backend.open_attempts == 0 and backend.navigations == []


def test_scripted_orchestrator_can_navigate_before_requesting_browser_open():
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall("open_url", {"url": "https://react.dev"}, "one")),
        LLMResponse(tool_call=ToolCall("open_browser", {}, "two")),
        LLMResponse(tool_call=ToolCall("open_url", {"url": "https://react.dev/reference/react"}, "three")),
        LLMResponse(text="Opened React docs."),
    ])
    assert Orchestrator(provider, registry).process("Open the React docs") == "Opened React docs."
    assert provider.calls[1].tool_result["status"] == "navigated"
    assert provider.calls[2].tool_result["status"] == "opened"
    assert provider.calls[3].tool_result["status"] == "navigated"
    assert backend.open_attempts == 1
    assert backend.navigations == ["https://react.dev", "https://react.dev/reference/react"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("github.com", "https://github.com"), ("https://github.com", "https://github.com")],
)
def test_normalize_url_adds_https_only_to_scheme_less_hostnames(value: str, expected: str):
    assert normalize_url(value) == expected


@pytest.mark.parametrize(
    "url",
    [
        "www.x.com/a b",
        "javascript:alert(1)",
        "file:///C:/x",
        "data:text/html,x",
        "chrome://settings",
        "about:blank",
        "ftp://x.com",
        "localhost",
    ],
)
def test_open_url_rejects_malformed_or_unsafe_urls(url: str):
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    registry.execute("open_browser", {})

    result = registry.execute("open_url", {"url": url})

    assert not result.success
    assert result.data["status"] == "invalid_url"
    assert backend.url == ""


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        ("google", "https://www.google.com/search?q=c%2B%2B+%26+rust%3F"),
        ("duckduckgo", "https://duckduckgo.com/?q=c%2B%2B+%26+rust%3F"),
    ],
)
def test_search_web_builds_encoded_engine_url(engine: str, expected: str):
    backend = FakeBrowserBackend()
    registry, context = make_registry(backend)
    registry.execute("open_browser", {})

    result = registry.execute("search_web", {"query": "c++ & rust?", "engine": engine})

    assert result.success
    assert backend.url == expected
    assert result.data["url"] == expected.split("?")[0]
    assert context.current_url == result.data["url"]


def test_search_web_quotes_unicode_and_defaults_to_duckduckgo():
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    registry.execute("open_browser", {})

    result = registry.execute("search_web", {"query": "東京"})

    assert result.success
    assert backend.url == "https://duckduckgo.com/?q=%E6%9D%B1%E4%BA%AC"
    assert result.data["url"] == "https://duckduckgo.com/"


def test_search_web_success_includes_content_unavailable_note():
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)
    registry.execute("open_browser", {})

    result = registry.execute("search_web", {"query": "jarvis"})

    assert result.success
    assert result.data["status"] == "navigated"
    assert result.data["note"] == "Results page opened. Use get_page_content to read visible content if needed."


def test_search_web_blocks_google_verification_page():
    class VerificationBackend(FakeBrowserBackend):
        def navigate(self, url: str) -> tuple[str, str]:
            self.url = "https://www.google.com/sorry/index"
            self.title = "Before you continue"
            return self.url, self.title

    registry, _ = make_registry(VerificationBackend())
    registry.execute("open_browser", {})

    result = registry.execute("search_web", {"query": "jarvis", "engine": "google"})

    assert not result.success
    assert result.data["status"] == "blocked"
    assert result.error == "The search engine showed a verification page instead of results"


@pytest.mark.parametrize(("url", "title", "expected"), [
    ("https://consent.google.com/", "Google", "consent_required"),
    ("https://example.com/privacy/consent", "Privacy", "consent_required"),
    ("https://example.com/search", "Before you continue to Google", "consent_required"),
    ("https://example.com/search", "Verify you are human", "blocked"),
    ("https://example.com/search", "Cookie Consent - Google Search", "navigated"),
])
def test_search_interstitials_use_url_or_title_without_misclassifying_search_titles(url, title, expected):
    class InterstitialBackend(FakeBrowserBackend):
        def navigate(self, requested_url):
            return url, title

    registry, _ = make_registry(InterstitialBackend())
    registry.execute("open_browser", {})
    result = registry.execute("search_web", {"query": "browser verification", "engine": "google"})
    assert result.data["status"] == expected
    assert result.success == (expected == "navigated")


@pytest.mark.parametrize("provider_available", [True, False])
def test_orchestrator_reports_blocked_search_once_and_keeps_accurate_user_message(provider_available):
    class ChallengeBackend(FakeBrowserBackend):
        def navigate(self, requested_url):
            self.url = "https://www.google.com/sorry/index?q=opaque-token"
            return self.url, "Verify you are human"

    registry, _ = make_registry(ChallengeBackend())
    registry.execute("open_browser", {})
    initial = LLMResponse(tool_call=ToolCall("search_web", {"query": "jarvis", "engine": "google"}))
    responses = [initial, LLMResponse(text="Results found")] if provider_available else [initial]
    provider = FakeLLMProvider(responses)
    answer = Orchestrator(provider, registry).process("Search for jarvis")
    assert "verification page instead of results" in answer
    assert len(provider.calls) == 2
    payload = provider.calls[1].tool_result
    assert not payload["success"] and payload["result"]["status"] == "blocked"
    assert payload["result"]["url"] == "https://www.google.com/sorry/index"
    assert "opaque-token" not in str(payload)
    assert [record["tool_name"] for record in registry.audit_logger.records].count("search_web") == 1


@pytest.mark.parametrize("arguments", [{"query": ""}, {"query": " "}, {"query": "x", "engine": "bing"}, {"query": "x" * 301}])
def test_search_web_rejects_invalid_arguments(arguments: dict[str, str]):
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)

    result = registry.execute("search_web", arguments) if arguments.get("engine") == "bing" else None
    if result is not None:
        assert not result.success
        assert result.data["status"] == "invalid_engine"
    else:
        with pytest.raises(ToolArgumentError):
            registry.execute("search_web", arguments)
    assert backend.open_attempts == 0 and backend.navigations == []


def test_search_web_starts_browser_then_reuses_it_for_later_searches():
    backend = FakeBrowserBackend()
    registry, _ = make_registry(backend)

    for query in ("avengers", "hulk"):
        assert registry.execute("search_web", {"query": query}, request_id=query).success

    assert backend.open_attempts == 1
    assert backend.navigations == ["https://duckduckgo.com/?q=avengers", "https://duckduckgo.com/?q=hulk"]
    assert [(record["tool_name"], record["request_id"]) for record in registry.audit_logger.records] == [
        ("open_browser", "avengers"), ("search_web", "avengers"), ("search_web", "hulk"),
    ]


def test_search_web_is_registered_with_described_parameters():
    registry, _ = make_registry()

    definition = next(item for item in registry.definitions() if item.name == "search_web")

    assert definition.parameters["properties"]["query"]["description"]
    assert definition.parameters["properties"]["engine"]["description"]
    assert "returns no result text" in definition.description.lower()
    assert "get_page_content" in definition.description


@pytest.mark.parametrize("url", ["javascript:alert(1)", "https://user:pass@example.com", "https://"])
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
    assert result.error == "Navigation failed or the destination was blocked"


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
