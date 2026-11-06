from app.config import Settings
from app.model_registry import ModelRegistry
from app.model_router import ModelRouter
from app.provider_manager import create_provider_manager
from app.llm import LLMResponse


def test_default_task_routing():
    router = ModelRouter(Settings())
    assert router.local_model("hello there").model == "llama3.2:3b"
    assert router.local_model("debug this Python function").model == "qwen3:4b"
    assert router.local_model("what is the weather?", has_tools=True).model == "qwen3:4b"
    assert router.local_model("describe this", has_image=True).model == "gemma3:4b"


def test_task_mappings_are_configurable():
    settings = Settings(ollama_general_model="general-x", ollama_coding_model="code-x", ollama_tool_model="tool-x", ollama_vision_model="vision-x")
    router = ModelRouter(settings)
    assert router.local_model("hello").model == "general-x"
    assert router.local_model("write code").model == "code-x"
    assert router.local_model("weather", has_tools=True).model == "tool-x"
    assert router.local_model("image", has_image=True).model == "vision-x"


def test_registry_exposes_capabilities_and_availability():
    settings = Settings(gemini_api_key="configured", openrouter_api_key="configured", openrouter_model="router", grok_api_key="configured")
    registry = ModelRegistry(settings, ollama_models={"llama3.2:3b"})
    assert registry.find("ollama", "llama3.2:3b").available
    assert not registry.find("ollama", "qwen3:4b").available
    assert "image" in registry.find("ollama", "gemma3:4b").input_types
    assert registry.find("grok", "grok-3-mini").available


def test_offline_mode_constructs_only_local_provider():
    manager = create_provider_manager(Settings(offline_mode=True), constructors={"ollama": lambda _: type("Local", (), {"generate": lambda self, prompt, **kwargs: LLMResponse(text="local")})()})
    assert [name for name, _ in manager._providers] == ["ollama"]
    assert manager.generate("hello").text == "local"


def test_explicit_cloud_provider_is_first_and_fallback_can_be_disabled():
    names = ("gemini", "openrouter", "grok", "ollama")
    constructors = {name: (lambda provider_name: lambda _: type("P", (), {"generate": lambda self, prompt, **kwargs: LLMResponse(text=provider_name)})())(name) for name in names}
    settings = Settings(
        llm_provider="grok", llm_provider_order=names, grok_api_key="key",
        gemini_api_key="key", openrouter_api_key="key", openrouter_model="router",
    )
    manager = create_provider_manager(settings, constructors=constructors)
    assert [name for name, _ in manager._providers] == ["grok", "gemini", "openrouter", "ollama"]
    manager = create_provider_manager(Settings(
        llm_provider="grok", llm_provider_order=names, grok_api_key="key", fallback_enabled=False,
    ), constructors=constructors)
    assert [name for name, _ in manager._providers] == ["grok"]
