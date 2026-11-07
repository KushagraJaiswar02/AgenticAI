"""Session-scoped browser state and the small browser backend boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any
from urllib.parse import parse_qs, urlsplit

from app.browser_navigation import BrowserNavigation, NavigationDenied
from app.browser_observation import (
    MAX_SCAN_OFFSET, READ_PAGE, research_links, sanitize_page_text, search_result_sources, select_page_sections,
)
from app.browser_security import (
    BlockedBrowserDestination, BrowserGateway, InvalidBrowserURL, safe_browser_title, safe_browser_url,
    validate_browser_url,
)
from app.research_budget import ResearchLimitReached, active_research, research_timeout


class BrowserHTTPError(RuntimeError):
    def __init__(self, status: int, url: str | None = None) -> None:
        super().__init__(f"Navigation returned HTTP {status}")
        self.status = status
        self.url = url


class BrowserNavigationTimeout(RuntimeError):
    """The page did not finish navigation within Playwright's deadline."""


@dataclass(frozen=True, repr=False)
class _NavigationFailure:
    requested_url: str
    response_url: str | None = None
    http_status: int | None = None


class PlaywrightBrowserBackend:
    """Launch one visible Chromium page through Playwright."""

    def __init__(self, on_closed=None) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._gateway: BrowserGateway | None = None
        self._navigation: BrowserNavigation | None = None
        self._browser_name: str | None = None
        self._on_closed = on_closed

    def open(self, browser: str) -> str:
        if browser != "brave":
            raise RuntimeError("Browser is not allowlisted")
        if self.is_open() and self._browser_name:
            return self._browser_name
        from playwright.sync_api import sync_playwright

        if tuple(map(int, version("playwright").split(".")[:2])) < (1, 59):
            raise NavigationDenied("Browser authorization requires Playwright 1.59 or newer")
        self.close()
        try:
            self._gateway = BrowserGateway()
            self._playwright = sync_playwright().start()
            executable = self._find_brave_executable()
            self._browser = self._playwright.chromium.launch(
                headless=False,
                executable_path=executable,
                ignore_default_args=["--disable-popup-blocking"],
                proxy={"server": self._gateway.url, "bypass": "<-loopback>"},
                **({"timeout": research_timeout(30) * 1000} if active_research.get() else {}),
                args=[
                    "--disable-quic",
                    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
                ],
            )
            self._context = self._browser.new_context(service_workers="block", accept_downloads=False)
            self._page = self._context.new_page()
            self._navigation = BrowserNavigation(self._context, self._page, self._disable_navigation)
            if self._on_closed:
                self._page.on("close", self._page_closed)
                self._browser.on("disconnected", self._browser_disconnected)
            self._browser_name = "brave" if executable else "chromium"
            return self._browser_name
        except BaseException:
            self.close()
            raise

    def is_open(self) -> bool:
        return bool(
            self._browser and self._browser.is_connected() and self._page and not self._page.is_closed()
            and self._navigation and self._navigation.available
        )

    def _disable_navigation(self) -> None:
        gateway, self._gateway = self._gateway, None
        if gateway is not None:
            gateway.close()
        if self._page is not None and not self._page.is_closed():
            self._page.close(run_before_unload=False)

    def _page_closed(self, page: Any) -> None:
        if page is self._page and self._on_closed:
            self._on_closed()

    def _browser_disconnected(self, browser: Any) -> None:
        if browser is self._browser and self._on_closed:
            self._on_closed()

    def close(self) -> None:
        context, browser, playwright, gateway = self._context, self._browser, self._playwright, self._gateway
        if self._navigation is not None:
            self._navigation.retire()
        self._navigation = None
        self._page = self._context = self._browser = self._playwright = self._gateway = self._browser_name = None
        for resource, method in ((context, "close"), (browser, "close"), (playwright, "stop"), (gateway, "close")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception:
                    # A disconnected resource must not prevent the others from closing.
                    continue

    def navigate(self, url: str) -> tuple[str, str | None]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        validate_browser_url(url)
        if self._page is None:
            raise RuntimeError("No browser page is currently available")
        guard = self._navigation
        if guard is None:
            raise NavigationDenied("Browser navigation authorization is unavailable")
        guard.begin(url, active_research.get())
        succeeded = False
        try:
            response = self._page.goto(
                url, wait_until="domcontentloaded",
                **({"timeout": research_timeout(30) * 1000} if active_research.get() else {}),
            )
            succeeded = True
        except Exception as exc:
            if guard.failure is not None:
                raise guard.failure from exc
            if isinstance(exc, PlaywrightTimeoutError):
                guard.stop_loading()
                raise BrowserNavigationTimeout() from exc
            raise
        except BaseException:
            self.close()
            raise
        finally:
            guard.end(succeeded=succeeded)
        if response is not None and response.status >= 400:
            raise BrowserHTTPError(response.status, getattr(response, "url", None))
        return self.current_url(), self.current_title()

    def current_url(self) -> str:
        if self._page is None:
            raise RuntimeError("No browser page is currently available")
        return str(self._page.url)

    def current_title(self) -> str | None:
        if self._page is None:
            raise RuntimeError("No browser page is currently available")
        return self._page.title()

    def read_page(
        self, max_chars: int, wait_ms: int, *, targeted: bool = False, discover_links: bool = False,
        scan_offset: int = 0,
        search_selector: str = "",
    ) -> dict[str, Any]:
        page = self._page
        if page is None:
            raise RuntimeError("No browser page is currently available")
        if wait_ms:
            page.wait_for_timeout(research_timeout(wait_ms / 1000) * 1000)
        snapshot = page.wait_for_function(
            READ_PAGE, arg={
                "maxChars": max_chars, "targeted": targeted, "discoverLinks": discover_links,
                "scanOffset": scan_offset, "maxScanOffset": MAX_SCAN_OFFSET,
                "searchSelector": search_selector,
            },
            timeout=research_timeout(2) * 1000,
        )
        try:
            return snapshot.json_value()
        finally:
            snapshot.dispose()

    @staticmethod
    def _find_brave_executable() -> str | None:
        from app.tools import OpenApplicationTool

        return OpenApplicationTool._find_executable("brave")


@dataclass
class BrowserContext:
    """Facts about the one browser page controlled during a CLI session."""

    browser_running: bool = False
    current_url: str | None = None
    current_title: str | None = None
    last_opened_url: str | None = None
    _backend: Any = field(default=None, repr=False, compare=False)
    _browser_name: str | None = field(default=None, init=False, repr=False)
    _last_navigation_failure: _NavigationFailure | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._backend is None:
            self._backend = PlaywrightBrowserBackend(on_closed=self._reset_state)

    def _reset_state(self) -> None:
        self.browser_running = False
        self.current_url = self.current_title = self.last_opened_url = self._browser_name = None
        self._last_navigation_failure = None

    def close(self) -> None:
        try:
            close = getattr(self._backend, "close", None)
            if close is not None:
                close()
        finally:
            self._reset_state()

    def has_running_browser(self) -> bool:
        is_open = getattr(self._backend, "is_open", None)
        if is_open is not None and (self.browser_running or isinstance(self._backend, PlaywrightBrowserBackend)):
            try:
                available = is_open()
            except Exception:
                available = False
            if not available:
                self.close()
        return self.browser_running

    def open_browser(self, browser: str = "brave") -> dict[str, Any]:
        if browser != "brave":
            return {"status": "failed", "error": "Browser is not allowlisted"}
        if self.has_running_browser():
            return {"status": "opened", "browser": self._browser_name or browser}
        try:
            self._browser_name = self._backend.open(browser) or browser
        except ResearchLimitReached:
            raise
        except NavigationDenied as exc:
            self._reset_state()
            return {"status": "failed", "error": str(exc)}
        except Exception:
            self._reset_state()
            return {"status": "failed", "error": "Browser could not be opened"}
        self.browser_running = True
        return {"status": "opened", "browser": self._browser_name}

    def open_url(self, url: str) -> dict[str, Any]:
        try:
            validate_browser_url(url)
        except InvalidBrowserURL:
            return {"status": "invalid_url", "error": "URL must be a valid absolute HTTP or HTTPS URL without credentials"}
        except BlockedBrowserDestination:
            return {"status": "blocked_destination", "error": "Browser destinations must use public unicast addresses"}
        if not self.has_running_browser():
            return {"status": "failed", "error": "No controlled browser is currently running"}
        try:
            current_url, current_title = self._backend.navigate(url)
        except ResearchLimitReached:
            raise
        except BrowserHTTPError as exc:
            page = self.current_page()
            self._last_navigation_failure = _NavigationFailure(url, exc.url, exc.status)
            response_url = safe_browser_url(exc.url) if exc.url else page.get("url")
            metadata = {"url": response_url} if response_url else {}
            if response_url and page.get("url") == response_url:
                metadata["title"] = page.get("title")
            return {
                "status": "http_error", "http_status": exc.status,
                "error": f"Navigation returned HTTP {exc.status}", "requested_url": safe_browser_url(url), **metadata,
            }
        except BrowserNavigationTimeout:
            self.current_page()
            self._last_navigation_failure = _NavigationFailure(url)
            return {
                "status": "timeout", "error": "Browser navigation timed out", "requested_url": safe_browser_url(url),
            }
        except NavigationDenied as exc:
            self.current_page()
            self._last_navigation_failure = _NavigationFailure(url)
            return {"status": "failed", "error": str(exc), "requested_url": safe_browser_url(url)}
        except Exception:
            self.current_page()
            return {"status": "failed", "error": "Navigation failed or the destination was blocked"}
        self.current_url = safe_browser_url(current_url)
        self.current_title = safe_browser_title(current_url, current_title)
        self.last_opened_url = self.current_url
        return {"status": "navigated", "url": self.current_url, "title": self.current_title}

    def discover_search_sources(self, engine: str, expected_url: str) -> list[dict[str, str]]:
        """Read result anchors only while the controlled page is on the requested search endpoint."""
        expected = urlsplit(expected_url)
        selector = 'a[data-testid="result-title-a"],a.result__a' if engine == "duckduckgo" else "a:has(h3)"
        try:
            snapshot = self._backend.read_page(1, 500, discover_links=True, search_selector=selector)
            actual = urlsplit(snapshot["url"])
            if (actual.scheme, actual.netloc, actual.path) != (expected.scheme, expected.netloc, expected.path):
                return []
            if parse_qs(actual.query).get("q") != parse_qs(expected.query).get("q"):
                return []
            return search_result_sources(engine, snapshot["url"], snapshot.get("links", []))
        except ResearchLimitReached:
            raise
        except Exception:
            return []

    def current_page(self) -> dict[str, Any]:
        if not self.has_running_browser():
            return {"status": "unavailable", "error": "No controlled browser/page is currently available"}
        try:
            current_url = self._backend.current_url()
            current_title = self._backend.current_title()
        except Exception:
            self.has_running_browser()
            self.current_url = self.current_title = None
            return {"status": "unavailable", "error": "Current page is unavailable"}
        self.current_url = safe_browser_url(current_url)
        self.current_title = safe_browser_title(current_url, current_title)
        return {"status": "available", "url": self.current_url, "title": self.current_title}

    def get_page_content(
        self, max_chars: int, wait_ms: int, *, topics: list[str] | None = None, discover_links: bool = False,
        scan_offset: int = 0,
    ) -> dict[str, Any]:
        empty = {"url": None, "title": None, "content": "", "character_count": 0,
                 "truncated": False, "untrusted": True}
        if not self.has_running_browser():
            return {**empty, "status": "unavailable", "error": "No controlled browser page is available"}
        try:
            if topics or discover_links or scan_offset:
                snapshot = self._backend.read_page(
                    max_chars, wait_ms, targeted=bool(topics), discover_links=discover_links,
                    **({"scan_offset": scan_offset} if scan_offset else {}),
                )
            else:
                snapshot = self._backend.read_page(max_chars, wait_ms)
            raw_url = snapshot["url"]
            url = safe_browser_url(raw_url)
            if snapshot.get("inaccessible") or url == "[REDACTED]":
                return {**empty, "status": "unavailable", "error": "This browser page cannot be read"}
            title = safe_browser_title(raw_url, snapshot.get("title"))
            content = sanitize_page_text(snapshot["content"]).strip()
            truncated = bool(snapshot["truncated"]) or len(content) > max_chars
            metadata = {}
            if topics:
                sections = snapshot.get("sections", [])
                content, matches, missing, clipped = select_page_sections(sections, topics, max_chars)
                scan_truncated = bool(snapshot.get("scanTruncated"))
                truncated = clipped or scan_truncated
                metadata.update(matches=matches, unmatched_topics=missing, scan_truncated=scan_truncated)
                if (next_offset := snapshot.get("nextScanOffset")) is not None:
                    metadata["next_scan_offset"] = next_offset
                if missing or truncated:
                    headings = (
                        sanitize_page_text(safe_browser_title(raw_url, " ".join(section["heading"])) or "")[:100]
                        for section in sections if section["heading"]
                    )
                    metadata["section_headings"] = list(dict.fromkeys(filter(None, headings)))[:24]
            if discover_links:
                pending_topics = list(metadata.get("unmatched_topics") or [])
                pending_topics.extend(item["topic"] for item in metadata.get("matches", []) if item["match"] == "reference")
                metadata["links"] = research_links(
                    raw_url, snapshot.get("links", []), pending_topics or topics or [],
                    related_topics=topics,
                )
            content = content[:max_chars].rstrip()
        except ResearchLimitReached:
            raise
        except Exception:
            if not self.has_running_browser():
                return {**empty, "status": "unavailable", "error": "The browser page closed during extraction"}
            return {**empty, "status": "failed", "error": "Page text could not be extracted"}
        return {
            "status": "available" if content else "empty",
            "url": url[:2048], "title": title[:256] if title else title,
            "content": content, "character_count": len(content), "truncated": truncated,
            "untrusted": True,
            **metadata,
        }
