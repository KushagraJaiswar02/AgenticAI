# JARVIS — Development Setup

## 1. Initial Environment

Target:
- Windows desktop
- Python 3.12
- Git
- virtual environment

Recommended setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

## 2. Dependency Philosophy

The V0 stack is locked, but implementation remains incremental. Do not install or implement a later V0 stage before its preceding stage is working.

Core dependencies:

- SQLite
- SQLAlchemy
- Pydantic
- python-dotenv
- httpx
- official OpenAI Python SDK
- pytest

V0 integration dependencies:

- OpenAI Responses API through the official OpenAI Python SDK
- ChromaDB
- Playwright with Chromium
- pywinauto

The exact package versions are UNSPECIFIED. V0.1 uses `gpt-5.6-luna` as the documented/default model, configurable through `OPENAI_MODEL`. The required environment variables are `OPENAI_API_KEY` and `OPENAI_MODEL`. Direct httpx calls are not used for the OpenAI API. No wake-word dependency, unrestricted shell library, remote vector database, or local foundation-model download is required.

## 3. Secrets

Store secrets in `.env`.

Never commit `.env`.

Maintain `.env.example` containing variable names only.

## 4. Development Rules

- Use type hints.
- Keep modules focused.
- Validate external input.
- Return structured tool results.
- Log failures with useful context.
- Never log secrets.
- Add tests for core orchestration and tools.
- Avoid unnecessary global state.

## 5. Test Layers

Tests are separated by responsibility:

- **Unit tests** cover configuration, normalization, tool schemas, registry behavior, weather parsing, and `FakeLLMProvider`.
- **Provider tests** mock the Gemini and OpenAI SDKs and verify adapter normalization, native tool calls, errors, and Gemini thought-signature context preservation.
- **Orchestrator integration tests** use the real `Orchestrator` and `ToolRegistry` with `FakeLLMProvider`. They verify deterministic tool selection, execution, result handoff, failures, and sequential interactions without Gemini, OpenAI, DNS, Wi-Fi, or external weather services.
- **Local fallback tests** verify that only recoverable `LLMProviderError` failures invoke the deterministic `LocalIntentRouter`; non-provider failures and unrecognized intents do not use it.
- **Live integration tests**, when present, must be marked `integration` and are never part of the default offline suite. They may fail because of network, quota, rate-limit, availability, or provider-outage conditions.

`FakeLLMProvider` exists only for deterministic application tests. It returns
predetermined provider-neutral responses and records calls; it does not
simulate intelligence or natural-language understanding.

## 6. Suggested Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
python -m pytest -q
python -m pytest -m integration
```

The first test command is the normal offline suite. The second explicitly runs
tests marked `integration`.

## 7. Incremental V0 Stages

V0 is one release with these internal stages:

1. **V0.1 — Core text pipeline:** Python 3.12 runtime, configuration, logging, provider-neutral LLM interface with OpenAI adapter, text input/output, and basic orchestrator.
2. **V0.2 — Tool system and permissions:** Pydantic tool schemas, registry, structured results, permission manager, target-specific confirmation, and constrained system/utility tools.
3. **V0.3 — Persistence and memory:** SQLAlchemy + SQLite migrations and repositories, conversation/task/tool persistence, explicit structured memory, and ChromaDB through replaceable memory and embedding interfaces.
4. **V0.4 — Filesystem and browser control:** safe-path filesystem tools, allowlisted subprocess/pywinauto automation, and Playwright with Chromium.
5. **V0.5 — Bounded multi-step execution:** orchestrator-owned execution state, step limits, structured failure handling, and progressive tool-result reasoning.
6. **V0.6 — Push-to-talk voice:** OpenAI audio API STT and TTS behind replaceable interfaces. No wake word.

These stages are implementation stages of V0, not separate products.
