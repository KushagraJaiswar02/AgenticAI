from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import LLMProviderError
from app.grok_provider import GrokProvider
from app.llm import ToolDefinition
from app.openrouter_provider import OpenRouterProvider


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        if self.error:
            raise self.error
        return self.response


class FakeChat:
    def __init__(self, response=None, error=None):
        self.completions = FakeCompletions(response, error)


def response(text="ok", tool=False):
    message = SimpleNamespace(
        content="" if tool else text,
        tool_calls=[SimpleNamespace(id="call-1", function=SimpleNamespace(name="weather", arguments='{"location":"Indore"}'))] if tool else [],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize(("provider_type", "settings", "model"), [
    (OpenRouterProvider, Settings(openrouter_api_key="test", openrouter_model="router-model"), "router-model"),
    (GrokProvider, Settings(grok_api_key="test", grok_model="grok-model"), "grok-model"),
])
def test_cloud_adapter_normalizes_text_and_tool_calls(provider_type, settings, model):
    client = SimpleNamespace(chat=FakeChat(response()))
    provider = provider_type(settings, client=client)
    assert provider.generate("hello").text == "ok"
    assert client.chat.completions.payload["model"] == model
    client.chat.completions.response = response(tool=True)
    result = provider.generate("weather", tools=[ToolDefinition("weather", "Weather", {"type": "object"})])
    assert result.tool_call.arguments == {"location": "Indore"}


@pytest.mark.parametrize(("provider_type", "settings"), [
    (OpenRouterProvider, Settings(openrouter_model="model")),
    (GrokProvider, Settings(grok_model="model")),
])
def test_cloud_adapter_requires_credential(provider_type, settings):
    with pytest.raises(LLMProviderError, match="API key is not configured"):
        provider_type(settings)


def test_grok_request_error_is_normalized():
    provider = GrokProvider(Settings(grok_api_key="test", grok_model="grok-model"), client=SimpleNamespace(chat=FakeChat(error=TimeoutError("timeout"))))
    with pytest.raises(LLMProviderError, match="Grok request failed"):
        provider.generate("hello")
