import logging
import os
import socket
from contextvars import Context
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import browser_context, browser_security
from app.browser_context import BrowserContext, PlaywrightBrowserBackend
from app.browser_navigation import BrowserNavigation, BrowserRequest, NavigationDenied, active_browser_request, browser_origin
from app.browser_security import BlockedBrowserDestination, BrowserGateway, connect_public, safe_browser_url
from app.browser_tools import create_browser_tools
from app.fake_provider import FakeLLMProvider
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.research_budget import ResearchBudget, ResearchLimitReached, ResearchLimits
from app.session import ConversationState
from app.tools import ToolRegistry


@pytest.mark.parametrize("url", [
    "https://[", "https://[::1]junk", "https://[v1.test]/",
    "https://example.com:bad/", "https://example.com:65536/", "https://example.com:0/",
    "https://example.com:/", "https://:443/", "https://a..example.com/", "https://u:p@example.com/",
    "https://127.1/", "https://0x7f000001/", "https://example.com\\@127.0.0.1/",
])
def test_malformed_authorities_are_structured_failures(url):
    backend = Mock()
    context = BrowserContext(browser_running=True, _backend=backend)
    result = ToolRegistry(create_browser_tools(context)).execute("open_url", {"url": url})
    assert result.data["status"] == "invalid_url"
    backend.navigate.assert_not_called()


@pytest.mark.parametrize("url", [
    "http://localhost/", "http://localhost./", "http://anything.localhost/",
    "http://127.0.0.1/", "http://10.1.2.3/", "http://100.64.0.1/", "http://169.254.169.254/",
    "http://0.0.0.0/", "http://224.1.2.3/", "http://192.0.2.10/",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://[fe80::1]/", "http://[64:ff9b::7f00:1]/",
])
def test_non_public_url_is_blocked_before_navigation(url):
    backend = Mock()
    result = BrowserContext(browser_running=True, _backend=backend).open_url(url)
    assert result["status"] == "blocked_destination"
    backend.navigate.assert_not_called()


def test_dns_rebinding_check_and_connect_use_same_numeric_address(monkeypatch):
    resolve = Mock(side_effect=[
        [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.215.14", 443))],
        [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))],
    ])
    upstream = Mock()
    monkeypatch.setattr(browser_security.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(browser_security.socket, "socket", Mock(return_value=upstream))
    assert connect_public("example.com", 443) is upstream
    resolve.assert_called_once()
    upstream.connect.assert_called_once_with(("93.184.215.14", 443))
    with pytest.raises(BlockedBrowserDestination):
        connect_public("example.com", 443)
    upstream.connect.assert_called_once()


def test_mixed_public_and_private_dns_answers_fail_closed(monkeypatch):
    answers = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 443))
               for ip in ("93.184.215.14", "192.168.1.1")]
    monkeypatch.setattr(browser_security.socket, "getaddrinfo", Mock(return_value=answers))
    dial = Mock()
    monkeypatch.setattr(browser_security.socket, "socket", dial)
    with pytest.raises(BlockedBrowserDestination):
        connect_public("example.com", 443)
    dial.assert_not_called()


def test_url_output_strips_userinfo_all_query_values_and_fragment():
    url = safe_browser_url("https://alice:password@example.com/?q=opaque&%61ccess_token=secret&x=unknown#private")
    assert url == "https://example.com/"


def test_current_page_cannot_expose_query_values_even_if_reflected_in_title():
    backend = Mock()
    backend.is_open.return_value = True
    backend.current_url.return_value = "https://example.com/sorry/index?q=opaque%2Bcode&session=secret"
    backend.current_title.return_value = "Challenge for opaque+code"
    registry = ToolRegistry(create_browser_tools(BrowserContext(browser_running=True, _backend=backend)))
    result = registry.execute("get_current_page", {})
    assert result.data == {"status": "available", "url": "https://example.com/sorry/index", "title": None}


def test_browser_data_and_echoed_secrets_are_omitted_from_logs_and_history(caplog):
    backend = Mock()
    backend.is_open.return_value = True
    url = "https://example.com/?access_token=private-token"
    backend.navigate.return_value = (url, "private-page-title")
    browser = BrowserContext(browser_running=True, _backend=backend)
    state = ConversationState(browser_context=browser)
    registry = ToolRegistry(create_browser_tools(browser))
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall("open_url", {"url": url})),
        LLMResponse(text="private-token private-page-title"),
    ])
    with caplog.at_level(logging.INFO, logger="jarvis.audit"):
        Orchestrator(provider, registry, state=state).process("Navigate to " + url)
    assert "private-token" not in str(provider.calls[-1].tool_result["url"])
    for surface in (caplog.text, str(registry.audit_logger.records), str(state.recent_messages()), str(state.history)):
        assert "private-token" not in surface
        assert "private-page-title" not in surface
    assert registry.audit_logger.records[-1]["execution_result"] == "completed"
    assert "navigated" in str(state.recent_messages())


def test_search_query_is_not_retained_in_audit_or_conversation():
    backend = Mock()
    backend.is_open.return_value = True
    backend.navigate.return_value = ("https://duckduckgo.com/?q=private-search", "private-search")
    browser = BrowserContext(browser_running=True, _backend=backend)
    state = ConversationState(browser_context=browser)
    registry = ToolRegistry(create_browser_tools(browser))
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall("search_web", {"query": "private-search"})),
        LLMResponse(text="Searched for private-search"),
    ])
    Orchestrator(provider, registry, state=state).process("Search for private-search")
    assert "private-search" not in str(registry.audit_logger.records)
    assert "private-search" not in str(state.recent_messages())


def test_repeated_open_reuses_session_and_closed_browser_is_reset():
    backend = Mock()
    backend.open.return_value = "chromium"
    backend.is_open.side_effect = [True, False]
    context = BrowserContext(_backend=backend)
    assert context.open_browser()["browser"] == "chromium"
    assert context.open_browser()["browser"] == "chromium"
    backend.open.assert_called_once_with("brave")
    context.current_url = context.last_opened_url = "https://example.com"
    assert context.current_page()["status"] == "unavailable"
    backend.close.assert_called_once()
    assert not context.browser_running
    assert context.current_url is None and context.last_opened_url is None


def test_partial_launch_closes_every_allocated_resource(monkeypatch):
    from playwright import sync_api

    gateway, playwright, browser, context = Mock(), Mock(), Mock(), Mock()
    gateway.url = "socks5://127.0.0.1:12345"
    playwright.chromium.launch.return_value = browser
    browser.new_context.return_value = context
    context.new_page.side_effect = RuntimeError("private failure details")
    monkeypatch.setattr(browser_context, "BrowserGateway", lambda: gateway)
    monkeypatch.setattr(sync_api, "sync_playwright", lambda: SimpleNamespace(start=lambda: playwright))
    backend = PlaywrightBrowserBackend()
    result = BrowserContext(_backend=backend).open_browser()
    assert result == {"status": "failed", "error": "Browser could not be opened"}
    context.close.assert_called_once()
    browser.close.assert_called_once()
    playwright.stop.assert_called_once()
    gateway.close.assert_called_once()
    assert not backend.is_open()


def test_gateway_close_disconnects_an_idle_client():
    gateway = BrowserGateway()
    try:
        with socket.create_connection(("127.0.0.1", int(gateway.url.rsplit(":", 1)[1])), timeout=2) as client:
            client.sendall(bytes((5, 1, 0)))
            assert browser_security._read(client, 2) == bytes((5, 0))
            gateway.close()
            assert client.recv(1) == b""
            assert not gateway._thread.is_alive()
    finally:
        gateway.close()


@pytest.mark.parametrize("exits_with_error", [False, True])
def test_cli_finally_closes_browser(monkeypatch, exits_with_error):
    from app import main
    from app.config import Settings

    browser = Mock()
    settings = Settings(database_url="")
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "configure_logging", Mock())
    for name in ("Database", "create_llm_provider", "FilesystemBoundary", "Orchestrator", "ToolRegistry"):
        monkeypatch.setattr(main, name, Mock())
    monkeypatch.setattr(main, "FilesystemContext", Mock())
    monkeypatch.setattr(main, "BrowserContext", lambda: browser)
    monkeypatch.setattr(main, "create_filesystem_tools", lambda *_: [])
    monkeypatch.setattr(main, "create_browser_tools", lambda _: [])
    monkeypatch.setattr(main, "_run_cli_loop", Mock(
        return_value=0, side_effect=KeyboardInterrupt if exits_with_error else None,
    ))
    if exits_with_error:
        with pytest.raises(KeyboardInterrupt):
            main.main()
    else:
        assert main.main() == 0
    browser.close.assert_called_once()


@pytest.mark.parametrize("status", [404, 429, 500])
def test_http_status_is_reported_and_error_page_state_is_accurate(status):
    backend = PlaywrightBrowserBackend()
    backend._browser = Mock()
    backend._page = Mock()
    backend._page.is_closed.return_value = False
    backend._page.goto.return_value = SimpleNamespace(status=status)
    backend._navigation = Mock(failure=None)
    backend._page.url = "https://example.com/error"
    backend._page.title.return_value = "Error page"
    context = BrowserContext(browser_running=True, _backend=backend, last_opened_url="https://example.com")
    result = ToolRegistry(create_browser_tools(context)).execute("open_url", {"url": "https://example.com/error"})
    assert not result.success
    assert result.data["status"] == "http_error"
    assert result.data["http_status"] == status
    assert context.current_url == "https://example.com/error"
    assert context.last_opened_url == "https://example.com"


def test_navigation_fails_closed_without_authorization():
    backend = PlaywrightBrowserBackend()
    backend._page = Mock()
    with pytest.raises(NavigationDenied, match="authorization is unavailable"):
        backend.navigate("https://example.com/")
    backend._page.goto.assert_not_called()


def test_browser_fails_closed_if_playwright_cannot_report_interceptor_loss(monkeypatch):
    monkeypatch.setattr(browser_context, "version", lambda name: "1.58.0")
    gateway = Mock()
    monkeypatch.setattr(browser_context, "BrowserGateway", gateway)
    result = BrowserContext().open_browser()
    assert result == {"status": "failed", "error": "Browser authorization requires Playwright 1.59 or newer"}
    gateway.assert_not_called()


def test_navigation_origin_preserves_idn_identity_and_normalizes_ipv6():
    assert browser_origin("https://faß.de/") == ("https", "xn--fa-hia.de", 443)
    assert browser_origin("https://faß.de/") != browser_origin("https://fass.de/")
    assert browser_origin("https://[2606:4700:0000::1111]/") == browser_origin("https://[2606:4700::1111]/")


def test_navigation_grant_allows_https_upgrade_without_downgrade_or_port_change():
    context, page = Mock(), Mock()
    page.url = "https://example.com/upgrade"
    session = context.new_cdp_session.return_value
    session.send.return_value = {"frameTree": {"frame": {"id": "controlled"}}}
    guard = BrowserNavigation(context, page)
    guard.begin("http://example.com/", None)
    for request_id, url in enumerate((
        "http://example.com:80/",
        "https://example.com/upgrade",
        "http://example.com/downgrade",
        "https://example.com:8443/other-port",
    )):
        redirect = {"redirectedRequestId": str(request_id - 1)} if request_id else {}
        guard._authorize_document({
            "requestId": str(request_id), "frameId": "controlled", "request": {"url": url, "method": "GET"}, **redirect,
        })
    assert [call.args[0] for call in session.send.call_args_list[-4:]] == [
        "Fetch.continueRequest", "Fetch.continueRequest", "Fetch.failRequest", "Fetch.failRequest",
    ]


@pytest.mark.parametrize("first, second, same", [
    ("http://EXAMPLE.com:80/", "http://example.com/", True),
    ("https://example.com:443/", "https://example.com/", True),
    ("https://example.com:8443/", "https://example.com/", False),
    ("http://example.com/", "https://example.com/", False),
    ("https://example.com./", "https://example.com/", False),
    ("https://faß.de/", "https://xn--fa-hia.de/", True),
    ("https://8.8.8.8:443/", "https://8.8.8.8/", True),
    ("https://[2606:4700:0000::1111]/", "https://[2606:4700::1111]/", True),
])
def test_research_destination_key_uses_navigation_origin(first, second, same):
    assert (browser_origin(first) == browser_origin(second)) is same
    assert (Orchestrator._navigation_destination(first) == Orchestrator._navigation_destination(second)) is same


def test_interleaved_browser_sessions_keep_independent_turn_grants():
    def start(host):
        context, page = Mock(), Mock()
        session = context.new_cdp_session.return_value
        session.send.return_value = {"frameTree": {"frame": {"id": "controlled"}}}
        guard = BrowserNavigation(context, page)
        request = BrowserRequest()
        scope = Context()
        scope.run(active_browser_request.set, request)
        scope.run(guard.begin, f"https://{host}/", None)
        return request, guard, session, scope

    def decide(guard, url):
        guard._authorize_document({
            "requestId": "id", "frameId": "controlled", "request": {"url": url, "method": "GET"},
        })

    first, first_guard, first_session, first_scope = start("one.example")
    _, second_guard, second_session, second_scope = start("two.example")
    # Protocol callbacks can be dispatched in a different Python context.
    second_scope.run(decide, first_guard, "https://one.example/")
    first_scope.run(decide, second_guard, "https://two.example/")
    first_guard.end(succeeded=True)
    second_guard.end(succeeded=True)
    first.finished = True
    second_scope.run(decide, first_guard, "https://one.example/after-turn")
    first_scope.run(decide, second_guard, "https://two.example/active")
    assert first_session.send.call_args.args[0] == "Fetch.failRequest"
    assert second_session.send.call_args.args[0] == "Fetch.continueRequest"


@pytest.mark.parametrize("finished", [False, True])
def test_navigation_checks_deadline_and_revocation_before_initial_request(monkeypatch, finished):
    from app import research_budget

    monkeypatch.setattr(research_budget, "monotonic", lambda: 0)
    budget = ResearchBudget(ResearchLimits(seconds=1))
    context, page = Mock(), Mock()
    session = context.new_cdp_session.return_value
    session.send.return_value = {"frameTree": {"frame": {"id": "controlled"}}}
    guard = BrowserNavigation(context, page)
    guard.begin("https://example.com/", budget)
    budget.finished = finished
    if not finished:
        monkeypatch.setattr(research_budget, "monotonic", lambda: 1)
    guard._authorize_document({
        "requestId": "initial", "frameId": "controlled", "request": {"url": "https://example.com/", "method": "GET"},
    })
    session.send.assert_called_with("Fetch.failRequest", {"requestId": "initial", "errorReason": "BlockedByClient"})
    assert isinstance(guard.failure, ResearchLimitReached)
    assert str(guard.failure) == ("research request ended" if finished else "time limit reached")


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("JARVIS_TEST_REAL_BROWSER") != "1", reason="Requires an installed Chromium and desktop")
@pytest.mark.parametrize("browser_name", ["chromium", "brave"])
def test_real_chromium_proxy_navigation_and_cleanup(monkeypatch, browser_name):
    if browser_name == "brave" and not PlaywrightBrowserBackend._find_brave_executable():
        pytest.skip("Brave is not installed")
    hits, blocked = [], []

    class PageServer(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirect-target")
                self.end_headers()
                return
            self.send_response(404 if self.path == "/missing" else 200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f'<title>Fixture page</title><img src="http://127.0.0.1:{self.server.server_port}/private-image">'.encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), PageServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real_connect = browser_security.connect_public

    def connect_fixture(host, port):
        if host == "fixture.example" and port == server.server_port:
            # Isolated integration fixture. Production never maps a public name to loopback.
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        if host == "127.0.0.1":
            blocked.append((host, port))
            return real_connect(host, port)
        raise BlockedBrowserDestination("Only the fixture is available")

    monkeypatch.setattr(browser_security, "connect_public", connect_fixture)
    if browser_name == "chromium":
        monkeypatch.setattr(PlaywrightBrowserBackend, "_find_brave_executable", staticmethod(lambda: None))
    context = BrowserContext()
    try:
        assert context.open_browser() == {"status": "opened", "browser": browser_name}
        backend = context._backend
        page = backend._page
        assert context.open_browser()["status"] == "opened"
        assert backend._page is page
        base = f"http://fixture.example:{server.server_port}"
        assert context.open_url(base + "/ok")["status"] == "navigated"
        page.wait_for_load_state("load", timeout=5000)
        assert context.current_page()["title"] == "Fixture page"
        assert blocked
        assert "/private-image" not in hits
        assert context.open_url(base + "/missing")["http_status"] == 404
        assert context.open_url(base + "/redirect")["status"] == "failed"
        assert "/redirect-target" not in hits

        page.close()
        assert not context.browser_running
        assert context.current_page()["status"] == "unavailable"
        assert context.open_browser()["status"] == "opened"
        browser, gateway = backend._browser, backend._gateway
        context.close()
        assert not browser.is_connected()
        assert not gateway._thread.is_alive()
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        thread.join()
