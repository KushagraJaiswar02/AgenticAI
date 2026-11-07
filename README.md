# JARVIS

JARVIS is a local Python terminal assistant with a provider-neutral LLM interface. Gemini is the default cloud provider, with OpenRouter and GroqCloud selectable as alternatives and Ollama as the local provider.

## Current milestone

V0.2 adds a small, explicit tool layer with native LLM function/tool calling. The first tool is a weather lookup that is executed only when the LLM natively requests it through the provider abstraction. The orchestrator remains provider-agnostic and does not branch on provider-specific logic.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env` to use the default cloud provider. OpenRouter uses `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.

For GroqCloud, create a key in the [Groq Console](https://console.groq.com/keys)
and set `GROQ_API_KEY` in `.env`. The adapter uses GroqCloud's OpenAI-compatible
endpoint and defaults to `openai/gpt-oss-120b`. Set `GROQ_MODEL=qwen/qwen3.8-27b`
to try GroqCloud's preview Qwen model. Set `LLM_PROVIDER=groq` to try GroqCloud
first. If your existing `LLM_PROVIDER_ORDER` lists `grok`, replace that entry
with `groq` to include GroqCloud in the fallback chain.

Provider failover follows `LLM_PROVIDER_ORDER`, beginning with `LLM_PROVIDER`.
The default order is `gemini,openrouter,groq,ollama`; unavailable or unconfigured
providers are skipped. Set `LLM_FALLBACK_ENABLED=false` to disable fallback. Set
`JARVIS_OFFLINE_MODE=true` to restrict requests to Ollama, regardless of cloud
provider configuration.

Ollama is optional and must be installed separately:

```powershell
ollama list
```

Ollama defaults to `http://localhost:11434`. The deterministic router selects
`OLLAMA_GENERAL_MODEL` (`llama3.2:3b`) for conversation, `OLLAMA_CODING_MODEL`
and `OLLAMA_TOOL_MODEL` (`qwen3:4b`) for technical and tool work, and
`OLLAMA_VISION_MODEL` (`gemma3:4b`) when image data is supplied through the
provider interface. Uncertain requests use the general model.
The registry is in `app/model_registry.py`; task classification is in
`app/model_router.py`. Ollama requests are serialized, retain a selected model
through tool continuation, unload the previous model before switching, and use
configurable `OLLAMA_KEEP_ALIVE` (5m) and `OLLAMA_CONTEXT_WINDOW` (4096).
JARVIS does not download missing models. Missing text models may fall back once
to the configured general model; missing vision models return an error.

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

Provider adapter tests mock the Gemini and OpenAI-compatible SDK clients. Any future live Gemini
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
