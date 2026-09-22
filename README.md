# JARVIS

JARVIS is a local Python terminal assistant with a provider-neutral LLM interface. Gemini is the active default provider, and OpenAI remains available as a fallback.

## Current milestone

V0.2 adds a small, explicit tool layer with native LLM function/tool calling. The first tool is a weather lookup that is executed only when the LLM natively requests it through the provider abstraction. The orchestrator remains provider-agnostic and does not branch on provider-specific logic.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. `GEMINI_MODEL` defaults to `gemini-3.8-flash` and may be changed there. Set `LLM_PROVIDER=openai` to use the OpenAI fallback with `OPENAI_API_KEY` and `OPENAI_MODEL`.

## Run

```powershell
python -m app.main
```

Type a message at the `User:` prompt. Enter `exit` or `quit` to stop.

## Tests

```powershell
python -m pytest -q
```

The normal suite is deterministic and offline: it uses `FakeLLMProvider` for
orchestrator integration tests and mocks the weather HTTP layer. The fake
provider returns predetermined `LLMResponse` objects; it does not simulate
intelligence or natural-language understanding.

If the configured cloud provider raises a recoverable `LLMProviderError`, the
orchestrator may use the deterministic `LocalIntentRouter` for a small set of
obvious weather requests. The router emits the same provider-neutral
`ToolCall` used by cloud providers; the `ToolRegistry` remains the only
component allowed to execute tools. It is a fallback, not a replacement for
the LLM.

Provider adapter tests mock the Gemini and OpenAI SDKs. Any future live Gemini
tests must be marked `integration` and run explicitly:

```powershell
python -m pytest -m integration
```
