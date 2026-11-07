"""Pre-network authorization for the controlled Chromium page."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from ipaddress import IPv6Address
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from app.browser_security import BlockedBrowserDestination, InvalidBrowserURL, validate_browser_url
from app.research_budget import ResearchBudget, ResearchLimitReached


class NavigationDenied(RuntimeError):
    """A navigation was blocked before Chromium sent its document request."""


@dataclass
class BrowserRequest:
    finished: bool = False


active_browser_request: ContextVar[BrowserRequest | None] = ContextVar("active_browser_request", default=None)


def browser_origin(url: str) -> tuple[str, str, int]:
    validate_browser_url(url)
    parsed = urlsplit(url)
    host = parsed.hostname
    if ":" in host:
        host = IPv6Address(host).compressed
    elif not host.isascii():
        # The standard-library IDNA codec can map a Unicode host to a different
        # ASCII domain (for example, "ß" to "ss"). Never grant that alias.
        try:
            host = httpx.URL(url).raw_host.decode("ascii")
        except httpx.InvalidURL as exc:
            raise InvalidBrowserURL("Invalid browser navigation hostname") from exc
    return parsed.scheme, host, parsed.port or (443 if parsed.scheme == "https" else 80)


def allows_origin(approved: tuple[str, str, int], target: tuple[str, str, int]) -> bool:
    scheme, host, port = approved
    return target == approved or (
        scheme == "http" and target == ("https", host, 443 if port == 80 else port)
    )


class BrowserNavigation:
    """One controlled tab. Each explicit navigation replaces its origin grant."""

    def __init__(self, context: Any, page: Any, on_unavailable: Callable[[], None] | None = None) -> None:
        self._page = page
        self._on_unavailable = on_unavailable
        self.available = True
        self._origin: tuple[str, str, int] | None = None
        self._budget: ResearchBudget | None = None
        self._request: BrowserRequest | None = None
        self._remaining = 0
        self._pending_path: bytes | None = None
        self._redirect_from: str | None = None
        self._explicit = False
        self._reported: set[str] = set()
        self.failure: NavigationDenied | ResearchLimitReached | None = None

        self._session = context.new_cdp_session(page)
        self._main_frame = self._session.send("Page.getFrameTree")["frameTree"]["frame"]["id"]
        self._session.on("close", self._interceptor_lost)
        self._session.on("Fetch.requestPaused", self._authorize_document)
        self._session.send("Fetch.enable", {"patterns": [{"resourceType": "Document", "requestStage": "Request"}]})
        # Playwright context routing sees the first request of a new tab, even
        # before its Page or Frame is available. No new tab has an explicit grant.
        context.route("**/*", self._route)
        context.on("page", self._close_new_page)

    def begin(self, url: str, budget: ResearchBudget | None) -> None:
        if not self.available or self._explicit:
            raise NavigationDenied("Browser navigation authorization is unavailable")
        try:
            origin = browser_origin(url)
            pending_path = httpx.URL(url).raw_path
        except (BlockedBrowserDestination, InvalidBrowserURL, httpx.InvalidURL) as exc:
            raise NavigationDenied("Browser navigation destination is invalid or prohibited") from exc
        self._origin = origin
        self._budget = budget
        self._request = active_browser_request.get()
        self._remaining = 12
        self._pending_path = pending_path
        self._redirect_from = None
        self._explicit = True
        self.failure = None
        self._reported.clear()

    def end(self, *, succeeded: bool) -> None:
        if not succeeded:
            self._origin = None
        self._explicit = False
        self._pending_path = None
        self._redirect_from = None

    def retire(self) -> None:
        self.available = False
        self._origin = None

    def stop_loading(self) -> None:
        self.end(succeeded=False)
        self._reply("Page.stopLoading", {})

    def _interceptor_lost(self, *_: Any) -> None:
        if not self.available:
            return
        self.retire()
        self.failure = NavigationDenied("Browser navigation interception was lost")
        self._report(str(self.failure))
        if self._on_unavailable is not None:
            self._on_unavailable()

    def _reserve(self) -> None:
        if self._request is not None and self._request.finished:
            raise NavigationDenied("Browser navigation request ended")
        if self._budget is not None:
            self._budget.reserve_browser(navigation=True)
        elif self._remaining:
            self._remaining -= 1
        else:
            raise NavigationDenied("Browser navigation limit reached")

    def _check_document(self, url: str, event: dict) -> None:
        if not self.available or self._origin is None:
            self._reserve()
            raise NavigationDenied("Browser navigation has no approved destination")
        try:
            origin = browser_origin(url)
            path = httpx.URL(url).raw_path if self._pending_path is not None else None
        except (BlockedBrowserDestination, InvalidBrowserURL, httpx.InvalidURL) as exc:
            self._reserve()
            raise NavigationDenied("Browser navigation destination is invalid or prohibited") from exc
        redirect_from = event.get("redirectedRequestId")
        initial = (
            self._pending_path is not None and path == self._pending_path and redirect_from is None
            and event["request"].get("method") == "GET" and allows_origin(self._origin, origin)
        )
        explicit_hop = initial or (redirect_from is not None and redirect_from == self._redirect_from)
        if initial:
            if self._request is not None and self._request.finished:
                raise NavigationDenied("Browser navigation request ended")
            self._pending_path = None
        if initial and self._budget is not None:
            self._budget.remaining()
        elif not initial:
            self._reserve()
        try:
            if self._explicit and not explicit_hop and browser_origin(self._page.url) != self._origin:
                raise NavigationDenied("Browser navigation has no committed origin grant")
        except (BlockedBrowserDestination, InvalidBrowserURL) as exc:
            raise NavigationDenied("Browser navigation has no committed origin grant") from exc
        # Preserve the existing HTTP upgrade for the same authority. Once
        # upgraded, the grant cannot be used to downgrade the document to HTTP.
        if not allows_origin(self._origin, origin):
            raise NavigationDenied("Browser navigation left the approved origin")
        self._origin = origin
        if explicit_hop:
            self._redirect_from = event["requestId"]

    def _authorize_document(self, event: dict) -> None:
        request_id = event["requestId"]
        redirect = "redirectedRequestId" in event
        try:
            if event["frameId"] == self._main_frame:
                self._check_document(event["request"]["url"], event)
        except (NavigationDenied, ResearchLimitReached) as exc:
            if self._explicit and (redirect or self.failure is None):
                self.failure = exc
            self._report(str(exc))
            self._reply("Fetch.failRequest", {"requestId": request_id, "errorReason": "BlockedByClient"})
        else:
            self._reply("Fetch.continueRequest", {"requestId": request_id})

    def _reply(self, command: str, arguments: dict) -> None:
        try:
            self._session.send(command, arguments)
        except Exception:
            self._interceptor_lost()

    def _route(self, route: Any) -> None:
        try:
            controlled = self.available and route.request.frame.page is self._page
        except Exception:
            # The first navigation request of a popup can have no frame yet.
            controlled = False
        if controlled:
            route.continue_()
        else:
            route.abort("blockedbyclient")

    def _close_new_page(self, page: Any) -> None:
        try:
            self._reserve()
        except (NavigationDenied, ResearchLimitReached) as exc:
            self._report(str(exc))
        finally:
            if not page.is_closed():
                page.close(run_before_unload=False)
        self._report("Browser popup was blocked")

    def _report(self, message: str) -> None:
        if message not in self._reported:
            self._reported.add(message)
            logging.getLogger("jarvis").warning("%s", message)
