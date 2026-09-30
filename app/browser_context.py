"""Session-scoped browser state and the small browser backend boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PlaywrightBrowserBackend:
    """Launch one visible Chromium page through Playwright."""

    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    def open(self, browser: str) -> None:
        if browser != "brave":
            raise RuntimeError(f"Browser '{browser}' is not allowlisted")
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        executable = self._find_brave_executable()
        launch_args = {"headless": False}
        if executable:
            launch_args["executable_path"] = executable
        self._browser = self._playwright.chromium.launch(**launch_args)
        self._page = self._browser.new_page()

    def navigate(self, url: str) -> tuple[str, str]:
        if self._page is None:
            raise RuntimeError("No browser page is currently available")
        self._page.goto(url, wait_until="domcontentloaded")
        return self.current_url(), self.current_title()

    def current_url(self) -> str:
        if self._page is None:
            raise RuntimeError("No browser page is currently available")
        return str(self._page.url)

    def current_title(self) -> str | None:
        if self._page is None:
            raise RuntimeError("No browser page is currently available")
        return self._page.title()

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

    def __post_init__(self) -> None:
        if self._backend is None:
            self._backend = PlaywrightBrowserBackend()

    def open_browser(self, browser: str = "brave") -> dict[str, Any]:
        try:
            self._backend.open(browser)
        except Exception as exc:
            self.browser_running = False
            return {"status": "failed", "error": f"Browser could not be opened: {exc}"}
        self.browser_running = True
        return {"status": "opened", "browser": browser}

    def open_url(self, url: str) -> dict[str, Any]:
        if not self.browser_running:
            return {"status": "failed", "error": "No controlled browser is currently running"}
        try:
            current_url, current_title = self._backend.navigate(url)
        except Exception as exc:
            return {"status": "failed", "error": f"Navigation failed: {exc}"}
        self.current_url = current_url
        self.current_title = current_title
        self.last_opened_url = current_url
        return {"status": "navigated", "url": current_url, "title": current_title}

    def current_page(self) -> dict[str, Any]:
        if not self.browser_running:
            return {"status": "unavailable", "error": "No controlled browser/page is currently available"}
        try:
            current_url = self._backend.current_url()
            current_title = self._backend.current_title()
        except Exception as exc:
            return {"status": "unavailable", "error": f"Current page is unavailable: {exc}"}
        self.current_url = current_url
        self.current_title = current_title
        return {"status": "available", "url": current_url, "title": current_title}
