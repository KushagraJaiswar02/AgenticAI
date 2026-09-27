from types import SimpleNamespace

import pytest

from app.cerebras_provider import CerebrasProvider
from app.config import Settings
from app.cohere_provider import CohereProvider
from app.errors import LLMProviderError
from app.groq_provider import GroqProvider
from app.llm import ToolDefinition
from app.mistral_provider import MistralProvider
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


def chat_response(text="ok", tool=False):
    message = SimpleNamespace(
        content="" if tool else text,
        tool_calls=(
            [SimpleNamespace(id="call-1", function=SimpleNamespace(
                name="weather", arguments='{"location": "Indore"}'
            ))] if tool else []
        ),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def base_settings() -> Settings:
    return Settings(
        groq_api_key="g", groq_model="groq-model",
        cerebras_api_key="c", cerebras_model="cerebras-model",
        openrouter_api_key="o", openrouter_model="openrouter/model",
        cohere_api_key="h", cohere_model="command-model",
        mistral_api_key="m", mistral_model="mistral-model",
    )


@pytest.mark.parametrize(
    ("provider_type", "client_attr", "key"),
    [
        (GroqProvider, "groq_api_key", "groq-model"),
        (CerebrasProvider, "cerebras_api_key", "cerebras-model"),
        (OpenRouterProvider, "openrouter_api_key", "openrouter/model"),
        (MistralProvider, "mistral_api_key", "mistral-model"),
    ],
)
def test_chat_provider_normalizes_text_and_tools(provider_type, client_attr, key):
    client = SimpleNamespace(chat=FakeChat(chat_response()))
    provider = provider_type(base_settings(), client=client)

    assert provider.generate("hello").text == "ok"
    assert provider.generate(
        "weather",
        tools=[ToolDefinition("weather", "weather", {"type": "object"})],
    ).text == "ok"
    assert client.chat.completions.payload["model"] == key


def test_groq_normalizes_tool_call_id():
    client = SimpleNamespace(chat=FakeChat(chat_response(tool=True)))
    result = GroqProvider(base_settings(), client=client).generate("weather")
    assert result.tool_call.call_id == "call-1"
    assert result.tool_call.arguments == {"location": "Indore"}


def test_cloud_provider_errors_are_normalized():
    client = SimpleNamespace(chat=FakeChat(error=TimeoutError("timeout")))
    with pytest.raises(LLMProviderError, match="Groq request failed"):
        GroqProvider(base_settings(), client=client).generate("hello")


def test_cohere_extracts_only_text_content():
    response = SimpleNamespace(message=SimpleNamespace(content=[
        SimpleNamespace(type="thinking", text="internal"),
        SimpleNamespace(type="text", text="public answer"),
    ]))
    result = CohereProvider(base_settings(), client=SimpleNamespace(chat=lambda **_: response)).generate("hello")
    assert result.text == "public answer"
