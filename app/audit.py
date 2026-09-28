"""Structured, sanitized audit records for tool policy and execution."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class AuditLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("jarvis.audit")
        self.records: list[dict[str, Any]] = []

    def record(self, **record: Any) -> None:
        safe = {key: self._sanitize(value, key) for key, value in record.items()}
        safe.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.records.append(safe)
        self._logger.info("tool_audit %s", json.dumps(safe, sort_keys=True, default=str))

    @classmethod
    def _sanitize(cls, value: Any, key: str = "") -> Any:
        lowered = key.lower()
        if any(word in lowered for word in ("password", "secret", "token", "api_key", "authorization")):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {str(k): cls._sanitize(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._sanitize(item, key) for item in value]
        if isinstance(value, str) and len(value) > 4096:
            return value[:4096] + "...[TRUNCATED]"
        return value
