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

Provider failover follows `LLM_PROVIDER_ORDER` and skips providers whose
implementation or required configuration is unavailable. Supported providers
are Gemini, OpenAI, Groq, Cerebras, OpenRouter, Mistral, Cohere, and Ollama.
The default order is
`gemini,openai,groq,cerebras,openrouter,mistral,cohere,ollama`;
`LLM_PROVIDER` is the preferred starting provider, not an exclusive selection.
Each cloud provider has its own `*_MODEL` and `*_TIMEOUT` setting. Cloud requests use
their configured 30-second default. Ollama uses
`OLLAMA_SIMPLE_TIMEOUT` (20 seconds by default) and
`OLLAMA_COMPLEX_TIMEOUT` (120 seconds by default).

Ollama is optional and must be installed separately:

```powershell
ollama pull llama3.2:3b
ollama pull qwen3:4b
ollama list
```

The local endpoint defaults to `http://localhost:11434`. Configure it with
`OLLAMA_BASE_URL`, `OLLAMA_SIMPLE_MODEL`, and `OLLAMA_COMPLEX_MODEL`.
`llama3.2:3b` handles clearly simple or ambiguous requests; `qwen3:4b`
handles clearly complex analysis, debugging, planning, and multi-step work.
The simple model is selected for speed rather than by forcing Qwen into
`/no_think`; `/no_think` is not considered a performance solution.
Only the complex Qwen path receives the optional thinking request field.

`open_application` is intentionally allowlisted. The current supported
application is `brave`; arbitrary executable paths, shell commands, and shell
input are rejected.

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
obvious requests. The router emits the same provider-neutral `ToolCall` used
by cloud providers; the `ToolRegistry` remains the only component allowed to
execute tools. It is a fallback, not a replacement for the LLM.

| Tool | External Network | Local Fallback |
| --- | --- | --- |
| weather | Yes | Yes |
| time | No | Yes |
| date | No | Yes |
| calculator | No | Yes |
| system_info | No | Yes |
| open_application | No | Yes |

Local tools are deliberately small and deterministic. The local router does
not provide general natural-language understanding. Its current pipeline is:

```text
normalization
    ↓
rule-based intent recognition
    ↓
entity extraction
    ↓
confidence decision
    ↓
provider-neutral ToolCall
```

The local router is a permanent JARVIS subsystem. Its implementation may
evolve from deterministic rules into a hybrid semantic/local-model system, but
the router-to-`ToolCall` boundary remains stable.

Provider adapter tests mock the Gemini and OpenAI SDKs. Any future live Gemini
tests must be marked `integration` and run explicitly:

```powershell
python -m pytest -m integration
```

Optional provider health diagnostics make one request per configured provider
and are never part of startup:

```powershell
python -m app.provider_health
```

Providers without both an API key and model are skipped. Tool calling is
provider/model dependent; adapters normalize native tool calls and never fake
unsupported tool behavior.
