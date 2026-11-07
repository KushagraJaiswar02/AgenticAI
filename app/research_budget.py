"""Request-local resource limits for autonomous browser research."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic


@dataclass(frozen=True)
class ResearchLimits:
    pages: int = 8
    iterations: int = 24
    navigations: int = 12
    evidence_chars: int = 16000
    seconds: float = 90.0
    provider_calls: int = 8

    def __post_init__(self) -> None:
        if not all(isfinite(value) and value > 0 for value in (
            self.pages, self.iterations, self.navigations, self.evidence_chars, self.seconds, self.provider_calls,
        )):
            raise ValueError("Research limits must be positive and finite")


class ResearchLimitReached(RuntimeError):
    pass


@dataclass
class ResearchBudget:
    limits: ResearchLimits
    deadline: float = field(init=False)
    provider_calls: int = 0
    iterations: int = 0
    navigations: int = 0
    pages: int = 0
    finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.deadline = monotonic() + self.limits.seconds

    def remaining(self) -> float:
        if self.finished:
            raise ResearchLimitReached("research request ended")
        seconds = self.deadline - monotonic()
        if seconds <= 0:
            raise ResearchLimitReached("time limit reached")
        return seconds

    def reserve_provider(self) -> None:
        self.remaining()
        if self.provider_calls >= self.limits.provider_calls:
            raise ResearchLimitReached("provider-call limit reached")
        self.provider_calls += 1

    def reserve_browser(self, *, navigation: bool = False) -> None:
        self.remaining()
        if self.iterations >= self.limits.iterations:
            raise ResearchLimitReached("research iteration limit reached")
        if navigation:
            if self.navigations >= self.limits.navigations:
                raise ResearchLimitReached("navigation attempt limit reached")
            if self.pages >= self.limits.pages:
                raise ResearchLimitReached("page limit reached")
            self.navigations += 1
        self.iterations += 1


active_research: ContextVar[ResearchBudget | None] = ContextVar("active_research", default=None)


def research_timeout(default: float) -> float:
    budget = active_research.get()
    return min(default, budget.remaining()) if budget else default
