"""Exact-action confirmation tokens with expiry and single-use semantics."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.policy import RiskLevel


@dataclass(frozen=True)
class ConfirmationRequest:
    token: str
    action_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: RiskLevel
    prompt: str
    expires_at: float


class ConfirmationManager:
    def __init__(self, ttl_seconds: float = 120.0, clock: Callable[[], float] | None = None) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock or time.time
        self._pending: dict[str, tuple[ConfirmationRequest, str]] = {}

    def create_request(self, tool_name: str, arguments: dict[str, Any], action_id: str, risk_level: RiskLevel, *, strong: bool = False, prompt: str | None = None) -> ConfirmationRequest:
        normalized = self._fingerprint(tool_name, arguments, action_id, risk_level)
        target = json.dumps(arguments, sort_keys=True, default=str)
        verb = "Type CONFIRM DELETE" if strong else "Proceed? [y/N]"
        request = ConfirmationRequest(
            token=secrets.token_urlsafe(18),
            action_id=action_id,
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level,
            prompt=prompt or f"This will execute {tool_name} with {target}. {verb}",
            expires_at=self._clock() + self._ttl_seconds,
        )
        self._pending[request.token] = (request, normalized)
        return request

    def approve(self, request: ConfirmationRequest, response: str, *, tool_name: str, arguments: dict[str, Any], action_id: str, risk_level: RiskLevel) -> bool:
        pending = self._pending.pop(request.token, None)
        if pending is None:
            return False
        if self._clock() >= request.expires_at:
            return False
        expected = self._fingerprint(tool_name, arguments, action_id, risk_level)
        if pending[1] != expected:
            return False
        normalized = response.strip().upper()
        return normalized == ("CONFIRM DELETE" if risk_level is RiskLevel.HIGH else "Y") or (risk_level is not RiskLevel.HIGH and normalized in {"YES", "PROCEED"})

    @staticmethod
    def _fingerprint(tool_name: str, arguments: dict[str, Any], action_id: str, risk_level: RiskLevel) -> str:
        payload = json.dumps({"tool": tool_name, "arguments": arguments, "action_id": action_id, "risk": risk_level.value}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
