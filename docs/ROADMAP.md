# JARVIS — Roadmap

## V0 Implementation Stages

V0 is one release implemented through the following incremental internal stages. These are not separate products.

## V0.1 — Core text pipeline

- Python 3.12 Windows runtime
- configuration and logging
- provider-neutral LLM interface with OpenAI adapter
- basic orchestrator
- text input and response

## V0.2 — Tool system and permissions

- Pydantic tool schemas and validation
- tool registry
- structured tool results
- calculator and system information
- constrained application launching
- permission manager
- target-specific confirmation
- deterministic local fallback intents
- allowlisted Brave application launching
- native tool-call continuation without duplicate execution

## V0.2/V0.3 stabilization

- Gemini → OpenAI → Ollama provider failover
- Groq, Cerebras, OpenRouter, Mistral, and Cohere adapters
- `llama3.2:3b` for simple local requests
- `qwen3:4b` for complex local reasoning
- separate simple/complex Ollama timeouts
- tool-versus-knowledge system guidance

## V0.3 — Persistence and memory

- SQLite through SQLAlchemy
- migrations and repositories
- conversation and task persistence
- explicit structured memory
- local ChromaDB semantic retrieval
- replaceable embedding interface

## V0.4 — Filesystem and browser control

- safe-path filesystem tools
- allowlisted subprocess execution
- supported pywinauto GUI/application automation
- Playwright with Chromium
- URL navigation and search

## V0.5 — Bounded multi-step execution

- orchestrator-owned execution state
- bounded tool sequences
- tool-result feedback to the LLM
- structured failure and recovery behavior

## V0.6 — Push-to-talk voice

- OpenAI audio API STT
- OpenAI audio API TTS
- replaceable STT/TTS interfaces
- push-to-talk only
- no wake-word detection

## Later versions

- screen understanding
- richer Windows control
- task recovery beyond V0 bounds
- scheduled/background tasks
- personality and deeper personalization
- long-running autonomous tasks
- additional local LLMs
- additional integrations

## Rule

Do not skip directly to later versions because a feature sounds impressive. Each V0 stage should produce a working, testable system.
