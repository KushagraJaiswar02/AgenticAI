import json

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

from app.config import Settings, load_settings
from app.errors import InvalidProviderResponseError, LLMProviderError
from app.llm import ToolDefinition
from app.openrouter_provider import OpenRouterProvider
from app.orchestrator import Orchestrator
from app.provider_manager import ProviderManager
from app.tools import Tool, ToolExecutionResult, ToolRegistry


MODEL = "google/gemma-4-26b-a4b-it:free"
ROUTING = {
    "order": ["google-ai-studio"],
    "only": ["google-ai-studio"],
    "allow_fallbacks": False,
    "require_parameters": True,
}
TOOLS = [ToolDefinition("lookup_smoke_code", "Look up a code", {"type": "object"})]


def completion(*, tool=False, model=MODEL, provider="Google AI Studio"):
    message = {"role": "assistant", "content": None if tool else "example-code"}
    if tool:
        message["tool_calls"] = [{
            "type": "function",
            "id": "call-1",
            "function": {"name": "lookup_smoke_code", "arguments": '{"lookup_id":"smoke-42"}'},
        }]
    return httpx.Response(200, json={
        "id": "generation-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "provider": provider,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
    })


def endpoint(provider_name, tag):
    return {"provider_name": provider_name, "tag": tag, "supported_parameters": ["tools"]}


@pytest.fixture
def build_adapter():
    clients = []

    def build(*responses, endpoints=None, route_initial=None):
        requests = []
        if endpoints is None:
            endpoints = [endpoint("Google AI Studio", "google-ai-studio")]
        queued = iter(responses)

        def handle(request):
            if request.method == "GET":
                assert request.url.path == f"/api/v1/models/{MODEL}/endpoints"
                return httpx.Response(200, json={"data": {"id": MODEL, "endpoints": endpoints}})
            requests.append(json.loads(request.content))
            if route_initial is not None and len(requests) == 1:
                return route_initial(requests[-1])
            return next(queued)

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = OpenAI(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            http_client=http_client, max_retries=0,
        )
        clients.append(client)
        provider = OpenRouterProvider(Settings(openrouter_api_key="test-key", openrouter_model=MODEL), client=client)
        return provider, requests

    yield build
    for client in clients:
        client.close()


def test_gemma_model_loads_from_openrouter_configuration(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", MODEL)
    assert load_settings(dotenv_path="missing-test.env").openrouter_model == MODEL


def test_gemma_initial_uses_default_routing_then_pins_native_continuation(build_adapter):
    provider, requests = build_adapter(completion(tool=True), completion())
    manager = ProviderManager([("openrouter", provider)])
    call = manager.generate("Look up smoke-42", tools=TOOLS).tool_call
    assert call.name == "lookup_smoke_code"
    assert call.arguments == {"lookup_id": "smoke-42"}
    assert call._turn.backend_provider == "google ai studio"
    assert call._turn.backend_endpoint == "google-ai-studio"

    final = manager.generate("Look up smoke-42", tool_call=call, tool_result={"code": "example-code"})
    assert final.text == "example-code"
    assert [request["model"] for request in requests] == [MODEL, MODEL]
    assert "provider" not in requests[0]
    assert requests[1]["provider"] == ROUTING
    assert requests[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call-1", "content": '{"code": "example-code"}',
    }
    assert requests[1]["tools"] == requests[0]["tools"]


@pytest.mark.parametrize(("model", "backend", "message"), [
    ("different/model:free", "Google AI Studio", "changed models"),
    (MODEL, "Unexpected Provider", "changed hosting providers"),
    (MODEL, None, "did not report the Gemma hosting provider"),
])
def test_gemma_continuation_rejects_identity_changes(build_adapter, model, backend, message):
    provider, _ = build_adapter(completion(tool=True), completion(model=model, provider=backend))
    call = provider.generate("Look up smoke-42", tools=TOOLS).tool_call
    with pytest.raises(InvalidProviderResponseError, match=message):
        provider.generate("Look up smoke-42", tool_call=call, tool_result={"code": "example-code"})


def test_exhausted_default_routing_surfaces_initial_429(build_adapter):
    provider, requests = build_adapter(httpx.Response(429, json={"error": {"message": "Rate limited"}}))
    with pytest.raises(LLMProviderError, match="OpenRouter request failed") as failure:
        provider.generate("Look up smoke-42", tools=TOOLS)
    assert failure.value.__cause__.status_code == 429
    assert len(requests) == 1
    assert "provider" not in requests[0]


def test_default_routing_can_recover_an_upstream_429_before_returning_success(build_adapter):
    upstream_attempts = []

    def simulate_openrouter(request):
        assert "provider" not in request
        upstream_attempts.extend([("First Host", 429), ("Second Host", 200)])
        return completion(tool=True, provider="Second Host")

    provider, requests = build_adapter(
        completion(provider="Second Host"),
        endpoints=[endpoint("First Host", "first"), endpoint("Second Host", "second/region")],
        route_initial=simulate_openrouter,
    )
    call = provider.generate("Look up smoke-42", tools=TOOLS).tool_call
    provider.generate("Look up smoke-42", tool_call=call, tool_result={"code": "example-code"})
    assert upstream_attempts == [("First Host", 429), ("Second Host", 200)]
    assert len(requests) == 2
    assert requests[-1]["provider"] == {**ROUTING, "order": ["second/region"], "only": ["second/region"]}


@pytest.mark.parametrize(("provider_name", "endpoints"), [
    (None, None),
    ("Unknown Host", None),
    ("Host", [endpoint("Host", "host/a"), endpoint("Host", "host/b")]),
])
def test_unverified_initial_identity_cannot_return_an_executable_call(build_adapter, provider_name, endpoints):
    provider, requests = build_adapter(completion(tool=True, provider=provider_name), endpoints=endpoints)
    with pytest.raises(InvalidProviderResponseError, match="hosting provider|hosting endpoint"):
        provider.generate("Look up smoke-42", tools=TOOLS)
    assert len(requests) == 1


def test_interleaved_gemma_turns_keep_independent_provider_pins(build_adapter):
    provider, requests = build_adapter(
        completion(tool=True, provider="Host A"), completion(tool=True, provider="Host B"),
        completion(provider="Host A"), completion(provider="Host B"),
        endpoints=[endpoint("Host A", "host-a"), endpoint("Host B", "host-b")],
    )
    manager = ProviderManager([("openrouter", provider)])
    first = manager.generate("First lookup", tools=TOOLS).tool_call
    second = manager.generate("Second lookup", tools=TOOLS).tool_call
    for prompt, call in (("First lookup", first), ("Second lookup", second)):
        assert manager.generate(prompt, tool_call=call, tool_result={"code": "example-code"}).text == "example-code"
    assert [request["provider"]["only"] for request in requests[2:]] == [["host-a"], ["host-b"]]


def test_unavailable_pinned_continuation_never_goes_to_backup(build_adapter):
    provider, requests = build_adapter(
        completion(tool=True), httpx.Response(503, json={"error": {"message": "Endpoint unavailable"}}),
    )

    class Backup:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1

    backup = Backup()
    manager = ProviderManager([("openrouter", provider), ("backup", backup)])
    call = manager.generate("Look up smoke-42", tools=TOOLS).tool_call
    with pytest.raises(LLMProviderError, match="currently unavailable"):
        manager.generate("Look up smoke-42", tool_call=call, tool_result={"code": "example-code"})
    assert backup.calls == 0
    assert "provider" not in requests[0]
    assert requests[1]["provider"] == ROUTING


def test_replayed_gemma_call_uses_cached_result_on_the_same_pin(build_adapter):
    provider, requests = build_adapter(completion(tool=True), completion(tool=True), completion())
    calls = []

    class Input(BaseModel):
        lookup_id: str

    class Lookup(Tool):
        name = "lookup_smoke_code"
        description = "Look up a code"
        input_model = Input

        def execute(self, arguments):
            calls.append(arguments.lookup_id)
            return ToolExecutionResult(True, {"code": "example-code"})

    manager = ProviderManager([("openrouter", provider)])
    assert Orchestrator(manager, ToolRegistry([Lookup()])).process("Look up smoke-42") == "example-code"
    assert calls == ["smoke-42"]
    assert [r["provider"] for r in requests[1:]] == [ROUTING, ROUTING]
    assert requests[1]["messages"][-1] == requests[2]["messages"][-1]
