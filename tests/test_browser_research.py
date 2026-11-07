import json
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from app.browser_observation import search_result_sources
from app.browser_research import run_research
from app.errors import LLMProviderError
from app.fake_provider import FakeLLMProvider
from app.llm import LLMResponse
from app.provider_manager import ProviderManager
from app.research_budget import ResearchBudget, ResearchLimitReached, ResearchLimits, active_research
from app.tools import ToolExecutionResult


@pytest.mark.parametrize(("limits", "expected", "navigations"), [
    ({"pages": 1}, "page limit reached", 1),
    ({"navigations": 1}, "navigation attempt limit reached", 1),
    ({"iterations": 2}, "research iteration limit reached", 1),
    ({"evidence_chars": len("Calibrated.")}, "evidence limit reached", 1),
    ({"provider_calls": 1}, "couldn't complete synthesis", 2),
    ({"seconds": 1}, "time limit reached", 1),
])
def test_limits_preserve_partial_evidence(monkeypatch, limits, expected, navigations):
    now = 0.0
    monkeypatch.setattr("app.research_budget.monotonic", lambda: now)
    budget = ResearchBudget(replace(ResearchLimits(), **limits))
    urls = []

    def execute(name, args):
        nonlocal now
        if name == "open_url":
            urls.append(args["url"])
            return ToolExecutionResult(True, data={"url": args["url"]})
        assert name == "get_page_content"
        if "seconds" in limits:
            now = 2.0
        return ToolExecutionResult(True, data={
            "url": urls[-1], "content": "Calibrated."[:args["max_chars"]],
            "matches": [{"topic": "calibration", "match": "heading"}],
            "links": [{"url": "https://example.com/retention", "text": "Retention"}],
        })

    def generate(prompt, instruction):
        budget.reserve_provider()
        if prompt.startswith("Research"):
            return json.dumps({"objectives": [{"topic": "calibration"}, {"topic": "retention"}]})
        raise LLMProviderError("no provider available for synthesis")

    output = run_research(
        "Research calibration and retention at https://example.com/start", budget,
        execute=execute, generate=generate, destination=lambda url: tuple(urlsplit(url)[:3]) if url else None,
    )
    assert expected in output
    assert "> Calibrated." in output and "retention (unresolved)" in output
    assert len(urls) == navigations and len(set(urls)) == len(urls)


def test_provider_limit_counts_actual_manager_fallback_attempts():
    class Unavailable(FakeLLMProvider):
        def generate(self, *args, **kwargs):
            raise LLMProviderError("unavailable")

    backup = FakeLLMProvider([LLMResponse(text="unused")])
    manager = ProviderManager([("first", Unavailable([])), ("second", backup)])
    budget = ResearchBudget(ResearchLimits(provider_calls=1))
    token = active_research.set(budget)
    try:
        with pytest.raises(ResearchLimitReached, match="provider-call limit"):
            manager.generate("Research this", tools=[])
    finally:
        active_research.reset(token)
    assert budget.provider_calls == 1 and not backup.calls


def test_search_discovery_rejects_private_query_and_non_search_redirects():
    sources = search_result_sources("duckduckgo", "https://duckduckgo.com/?q=topic", [
        ["https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle", "Article"],
        ["https://duckduckgo.com/l/?uddg=http%3A%2F%2F127.0.0.1%2F", "Private"],
        ["https://example.org/article?token=secret", "Sensitive"],
        ["https://duckduckgo.com/settings", "Search settings"],
    ])
    assert sources == [{"url": "https://example.com/article", "text": "Article"}]


def test_failed_target_is_not_replayed_after_landing_page_discovery():
    calls = []
    budget = ResearchBudget(ResearchLimits())

    def execute(name, arguments):
        if name == "get_page_content":
            return ToolExecutionResult(True, data={
                "url": "https://example.com/", "links": [
                    {"url": "https://example.com/missing", "text": "Calibration"},
                    {"url": "https://other.example/article", "text": "Calibration"},
                    {"url": "https://example.com/calibration", "text": "Calibration"},
                ],
                **({
                    "url": calls[-1], "content": "Calibration uses a reference beam.",
                    "matches": [{"topic": "calibration", "match": "heading"}],
                } if calls[-1].endswith("/calibration") else {}),
            })
        url = arguments["url"]
        calls.append(url)
        return ToolExecutionResult(not url.endswith("/missing"), data={
            "url": url, "http_status": 404 if url.endswith("/missing") else 200,
        })

    result = run_research(
        "Research calibration at https://example.com/missing", budget, execute=execute,
        generate=lambda prompt, _: json.dumps({"objectives": [{"topic": "calibration"}]})
        if prompt.startswith("Research") else "Evidence summary",
        destination=lambda url: tuple(urlsplit(url)[:3]) if url else None,
    )
    assert result == "Evidence summary"
    assert calls == ["https://example.com/missing", "https://example.com/", "https://example.com/calibration"]


def test_observed_index_survives_failed_scoped_search_for_missing_identifier():
    base = "https://docs.example.com"
    docs = {
        "/rules": ("Rules", [{"url": base + "/hooks", "text": "Hooks"}]),
        "/useEffect": ("useEffect", []),
        "/hooks": (None, [{"url": base + "/useState", "text": "useState"}]),
        "/useState": ("useState", []),
    }
    calls = []
    current = ""

    def execute(name, arguments):
        nonlocal current
        if name == "search_web":
            calls.append(arguments["query"])
            if arguments["query"].startswith("site:"):
                return ToolExecutionResult(False)
            return ToolExecutionResult(True, data={"sources": [
                {"url": base + "/rules", "text": "Rules"},
                {"url": base + "/useEffect", "text": "useEffect"},
            ]})
        if name == "open_url":
            current = arguments["url"]
            calls.append(urlsplit(current).path)
            return ToolExecutionResult(True, data={"url": current})
        topic, links = docs[urlsplit(current).path]
        return ToolExecutionResult(True, data={
            "url": current, "links": links, "content": f"{topic} definition." if topic else "",
            "matches": [{"topic": topic, "match": "heading"}] if topic else [],
        })

    output = run_research(
        "Research Rules, useEffect, and useState.", ResearchBudget(ResearchLimits()), execute=execute,
        generate=lambda message, _: json.dumps({
            "search_query": "Hook documentation", "objectives": [{"topic": topic} for topic in ("Rules", "useEffect", "useState")],
        }) if message.startswith("Research") else "Evidence summary",
        destination=lambda url: tuple(urlsplit(url)[:3]) if url else None,
    )
    assert output == "Evidence summary"
    assert calls == [
        "Hook documentation", "/rules", "/useEffect", "site:docs.example.com useState", "/hooks", "/useState",
    ]
