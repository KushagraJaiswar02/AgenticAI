"""Optional one-request provider health diagnostic."""

from __future__ import annotations

import logging
import time

from app.config import load_settings
from app.provider_manager import create_provider_manager


def main() -> int:
    logger = logging.getLogger("jarvis.health")
    settings = load_settings()
    manager = create_provider_manager(settings, logger=logger)
    providers = getattr(manager, "_providers", [])
    failures = 0
    for name, provider in providers:
        started = time.perf_counter()
        try:
            provider.generate("Reply with the single word: ok")
            print(f"{name:<12} OK {int((time.perf_counter() - started) * 1000)}ms")
        except Exception as exc:
            failures += 1
            print(f"{name:<12} FAIL {exc.__class__.__name__}")
    return 1 if failures and not providers else 0


if __name__ == "__main__":
    raise SystemExit(main())
