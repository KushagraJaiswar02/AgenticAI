"""Runtime configuration for JARVIS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.research_budget import ResearchLimits


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


SUPPORTED_PROVIDERS = ("gemini", "openrouter", "groq", "ollama")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    llm_provider_order: tuple[str, ...] = ("gemini", "openrouter", "groq", "ollama")
    llm_request_timeout: float = 30.0
    offline_mode: bool = False
    fallback_enabled: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout: float = 30.0
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout: float = 30.0
    openrouter_api_key: str | None = None
    openrouter_model: str = ""
    openrouter_timeout: float = 30.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_general_model: str = "llama3.2:3b"
    ollama_coding_model: str = "qwen3:4b"
    ollama_tool_model: str = "qwen3:4b"
    ollama_vision_model: str = "gemma3:4b"
    ollama_keep_alive: str = "5m"
    ollama_context_window: int = 4096
    ollama_timeout: float = 120.0
    # Compatibility with the previous simple/complex settings API.
    ollama_simple_model: str = "llama3.2:3b"
    ollama_complex_model: str = "qwen3:4b"
    ollama_model: str | None = None
    ollama_simple_timeout: float = 20.0
    ollama_complex_timeout: float = 120.0
    database_url: str = "sqlite:///jarvis.db"
    filesystem_allowed_roots: tuple[str, ...] = ()
    jarvis_workspace: str = ""
    project_root: str = ""
    research_limits: ResearchLimits = ResearchLimits()
    youtube_data_api_key: str | None = field(default=None, repr=False)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(*, dotenv_path: str | None = None) -> Settings:
    """Load settings from environment variables without logging secret values."""
    load_dotenv(dotenv_path=dotenv_path)
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    order_value = os.getenv("LLM_PROVIDER_ORDER", "gemini,openrouter,groq,ollama")
    retired = {"openai", "grok", "cerebras", "mistral", "cohere"}
    provider_order = tuple(
        item.strip().lower() for item in order_value.split(",")
        if item.strip() and item.strip().lower() not in retired
    )
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(f"LLM_PROVIDER must be one of: {', '.join(SUPPORTED_PROVIDERS)}")
    if not provider_order or any(item not in SUPPORTED_PROVIDERS for item in provider_order):
        raise ConfigurationError("LLM_PROVIDER_ORDER contains no providers or an unsupported provider")
    try:
        request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
        gemini_timeout = float(os.getenv("GEMINI_TIMEOUT", "30"))
        openrouter_timeout = float(os.getenv("OPENROUTER_TIMEOUT", "30"))
        groq_timeout = float(os.getenv("GROQ_TIMEOUT", "30"))
        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        context_window = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "4096"))
        research_limits = ResearchLimits(
            pages=int(os.getenv("JARVIS_RESEARCH_MAX_PAGES", "8")),
            iterations=int(os.getenv("JARVIS_RESEARCH_MAX_ITERATIONS", "24")),
            navigations=int(os.getenv("JARVIS_RESEARCH_MAX_NAVIGATIONS", "12")),
            evidence_chars=int(os.getenv("JARVIS_RESEARCH_MAX_EVIDENCE_CHARS", "16000")),
            seconds=float(os.getenv("JARVIS_RESEARCH_MAX_SECONDS", "90")),
            provider_calls=int(os.getenv("JARVIS_RESEARCH_MAX_PROVIDER_CALLS", "8")),
        )
    except ValueError as exc:
        raise ConfigurationError("Invalid provider timeout, context window, or research limit") from exc
    if min(request_timeout, gemini_timeout, openrouter_timeout, groq_timeout, ollama_timeout) <= 0 or context_window <= 0:
        raise ConfigurationError("Timeouts and context window must be positive")

    project_root = str(Path(os.getenv("JARVIS_PROJECT_ROOT", "").strip() or os.getcwd()).expanduser().resolve())
    workspace_root = str(Path(os.getenv("JARVIS_WORKSPACE", "").strip() or Path(project_root).parent).expanduser().resolve())
    allowed_value = os.getenv("JARVIS_ALLOWED_ROOTS", "").strip() or workspace_root
    general = os.getenv("OLLAMA_GENERAL_MODEL", os.getenv("OLLAMA_SIMPLE_MODEL", "llama3.2:3b")).strip()
    coding = os.getenv("OLLAMA_CODING_MODEL", os.getenv("OLLAMA_COMPLEX_MODEL", "qwen3:4b")).strip()
    tool = os.getenv("OLLAMA_TOOL_MODEL", coding).strip()
    vision = os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b").strip()
    return Settings(
        llm_provider=provider,
        llm_provider_order=provider_order,
        llm_request_timeout=request_timeout,
        offline_mode=_bool("JARVIS_OFFLINE_MODE", False),
        fallback_enabled=_bool("LLM_FALLBACK_ENABLED", True),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        gemini_timeout=gemini_timeout,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        groq_timeout=groq_timeout,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "").strip(),
        openrouter_timeout=openrouter_timeout,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_general_model=general,
        ollama_coding_model=coding,
        ollama_tool_model=tool,
        ollama_vision_model=vision,
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m").strip(),
        ollama_context_window=context_window,
        ollama_timeout=ollama_timeout,
        ollama_simple_model=general,
        ollama_complex_model=coding,
        ollama_simple_timeout=float(os.getenv("OLLAMA_SIMPLE_TIMEOUT", "20")),
        ollama_complex_timeout=float(os.getenv("OLLAMA_COMPLEX_TIMEOUT", str(ollama_timeout))),
        filesystem_allowed_roots=tuple(item.strip() for item in allowed_value.split(os.pathsep) if item.strip()),
        jarvis_workspace=workspace_root,
        project_root=project_root,
        research_limits=research_limits,
        youtube_data_api_key=os.getenv("YOUTUBE_DATA_API_KEY"),
    )
