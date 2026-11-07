"""Safe, structured diagnostics for provider failures."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from requests import exceptions as requests_errors

from app.errors import InvalidProviderResponseError, LLMProviderError


RECOVERABLE = frozenset({"daily_quota", "rate_limit", "server_error", "timeout", "network", "model_unavailable"})
REASONS = {
    "daily_quota": "daily quota exhausted",
    "rate_limit": "rate limited",
    "authentication": "authentication failed",
    "permission": "permission denied",
    "server_error": "provider temporarily unavailable",
    "timeout": "timeout",
    "network": "network failure",
    "model_unavailable": "model or endpoint unavailable",
    "invalid_request": "invalid request",
    "invalid_response": "invalid response",
    "unknown": "provider request failed",
}


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    model: str | None
    category: str
    status: int | None
    retry_after_seconds: int | None

    @property
    def reason(self) -> str:
        return REASONS[self.category]

    def describe(self) -> str:
        details = [self.reason]
        if self.status is not None:
            details.append(f"HTTP {self.status}")
        if self.retry_after_seconds is not None:
            details.append(f"retry after {self.retry_after_seconds} seconds")
        identity = f"{self.provider} ({self.model})" if self.model else self.provider
        return f"{identity}: {', '.join(details)}"


class ProviderUnavailableError(LLMProviderError):
    def __init__(self, failures: list[ProviderFailure], *, recovering: bool = False) -> None:
        self.failures = tuple(failures)
        if recovering:
            message = "Browser observations were collected, but no eligible fallback returned a summary"
        elif len(failures) == 1 and failures[0].category == "invalid_request":
            message = "The provider rejected the request"
        else:
            message = "All configured providers are currently unavailable for this request"
        super().__init__(f"{message}. {'; '.join(failure.describe() for failure in failures)}".rstrip())


def classify_failure(provider: str, model: str | None, error: Exception) -> ProviderFailure:
    chain = []
    cause = error
    while cause is not None and cause not in chain:
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    upstream = next((item for item in chain if _status(item) is not None), error)
    status = _status(upstream)
    category = "unknown"
    response = getattr(upstream, "response", None)
    headers = getattr(response, "headers", None) or {}
    retry_after = _retry_seconds(headers.get("retry-after"))
    if status == 429:
        category, retry_after = _quota_details(upstream, retry_after)
    elif status == 401:
        category = "authentication"
    elif status == 403:
        category = "permission"
    elif status == 404:
        category = "model_unavailable"
    elif status in {408, 504}:
        category = "timeout"
    elif status is not None and status >= 500:
        category = "server_error"
    elif status is not None and status >= 400:
        category = "invalid_request"
    elif isinstance(error, InvalidProviderResponseError):
        category = "invalid_response"
    elif any(isinstance(item, (TimeoutError, httpx.TimeoutException, requests_errors.Timeout)) for item in chain):
        category = "timeout"
    elif any(isinstance(item, (
        ConnectionError, httpx.NetworkError, httpx.RemoteProtocolError, requests_errors.ConnectionError,
    )) for item in chain):
        category = "network"
    model = getattr(error, "model", None) or model
    safe_model = model if isinstance(model, str) and re.fullmatch(r"[\w./:@+-]{1,160}", model) else None
    return ProviderFailure(provider, safe_model, category, status, retry_after)


def _status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) or getattr(error, "status_code", None) or getattr(error, "code", None)
    return int(status) if isinstance(status, int) and 400 <= status <= 599 else None


def _quota_details(error: Exception, retry_after: int | None) -> tuple[str, int | None]:
    body = getattr(error, "details", None) or getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    daily = False
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str):
            daily = bool(re.search(r"per[ _-]?day|daily", message, re.I))
        details = body.get("details")
        for detail in details if isinstance(details, list) else ():
            if not isinstance(detail, dict):
                continue
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo" and retry_after is None:
                delay = detail.get("retryDelay")
                if isinstance(delay, str) and re.fullmatch(r"\d+(?:\.\d+)?s", delay):
                    retry_after = _retry_seconds(delay[:-1])
            if detail.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
                violations = detail.get("violations")
                for violation in violations if isinstance(violations, list) else ():
                    if isinstance(violation, dict):
                        quota = f"{violation.get('quotaId', '')} {violation.get('description', '')}"
                        daily = daily or bool(re.search(r"per[ _-]?day|daily", quota, re.I))
        if retry_after is None and isinstance(message, str):
            delay = re.search(
                r"\bretry in (?:(\d{1,6})h)?(?:(\d{1,6})m)?(\d{1,6}(?:\.\d{1,12})?)s\b", message, re.I,
            )
            if delay:
                hours, minutes, seconds = delay.groups()
                retry_after = int(hours or 0) * 3600 + int(minutes or 0) * 60 + (_retry_seconds(seconds) or 0)
    return ("daily_quota" if daily else "rate_limit"), retry_after


def _retry_seconds(value: str | None) -> int | None:
    if not isinstance(value, str) or len(value) > 100:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, math.ceil(seconds)) if math.isfinite(seconds) else None
