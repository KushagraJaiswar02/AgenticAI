"""Deterministic tool risk classification and policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.errors import ApplicationError


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    CONFIRM = "CONFIRM"
    HIGH = "HIGH"
    FORBIDDEN = "FORBIDDEN"


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    DENY = "DENY"


class PolicyError(ApplicationError):
    """Base class for safety-policy failures."""


class PolicyDeniedError(PolicyError):
    """Raised when a tool is forbidden or otherwise denied."""


class ConfirmationRequiredError(PolicyError):
    """Raised when execution needs an exact, user-approved confirmation."""

    def __init__(self, request: Any) -> None:
        self.request = request
        super().__init__(request.prompt)


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    risk_level: RiskLevel
    reason: str
    confirmation_strength: str | None = None


class PolicyEngine:
    """Evaluate tool metadata locally; no model or external state is consulted."""

    def evaluate(self, tool: Any, arguments: dict[str, Any] | None = None) -> PolicyDecision:
        del arguments  # Argument validation belongs to the tool schema boundary.
        risk = getattr(tool, "risk_level", None)
        try:
            risk = risk if isinstance(risk, RiskLevel) else RiskLevel(risk)
        except (TypeError, ValueError) as exc:
            raise PolicyDeniedError(f"Tool '{getattr(tool, 'name', '<unknown>')}' has invalid risk metadata") from exc

        if risk is RiskLevel.FORBIDDEN:
            return PolicyDecision(PolicyAction.DENY, risk, "This capability is forbidden by JARVIS policy")
        if risk is RiskLevel.SAFE:
            return PolicyDecision(PolicyAction.ALLOW, risk, "Read-only tool")
        if risk is RiskLevel.LOW:
            return PolicyDecision(PolicyAction.ALLOW, risk, "Low-impact tool")
        if risk is RiskLevel.HIGH:
            return PolicyDecision(
                PolicyAction.REQUIRE_CONFIRMATION,
                risk,
                "High-impact action requires strong confirmation",
                confirmation_strength="strong",
            )
        return PolicyDecision(
            PolicyAction.REQUIRE_CONFIRMATION,
            risk,
            "Consequential action requires explicit confirmation",
            confirmation_strength="normal",
        )
