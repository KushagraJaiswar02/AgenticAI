# JARVIS — Project Definition

## 1. Purpose

JARVIS is a personal desktop AI assistant designed to interact with the user through text and voice, reason about requests using an LLM, retrieve relevant memory, and execute controlled actions on the user's computer and connected services.

The project is initially a local Python application using a cloud LLM as its reasoning engine. Local models may be introduced later behind the same LLM abstraction.

## 2. Core Principle

The LLM is not JARVIS by itself.

JARVIS is the system composed of:
- input/output interfaces
- orchestrator
- LLM
- memory
- tools
- task/state management
- permission and safety controls

The LLM reasons and proposes actions. The application validates and executes those actions.

## 3. v0 Goal

v0 proves the complete core loop:

User → input → orchestrator → LLM → tool/memory → result → response.

The target V0 release progressively includes:
- text interaction
- voice interaction
- conversation context
- opening applications
- basic browser actions
- basic file operations
- explicit long-term memory
- semantic memory retrieval
- multi-step tool execution
- confirmation for destructive actions

V0 is one release implemented through incremental internal stages. The first implementation stage is a text-only vertical slice. Voice, browser control, semantic memory, filesystem automation, and multi-step execution are added progressively within the same V0 release; they are not separate products.

### V0.2 tool milestone

V0.2 adds a small, explicit tool layer to the existing provider-neutral orchestration pattern. The first concrete tool is a weather lookup. The weather tool is registered with a tool registry, validated with Pydantic, and executed only after the LLM makes a native function/tool call that the orchestrator accepts. The LLM does not directly execute tools; JARVIS controls execution and feeds the structured tool result back through the provider's native function-response flow.

## Locked V0 Technology Stack

- Runtime: Python 3.12 on Windows desktop.
- Architecture: modular monolith.
- Core libraries: SQLite, SQLAlchemy, Pydantic, python-dotenv, httpx, and pytest.
- LLM: OpenAI Responses API through the official OpenAI Python SDK, behind a provider-neutral LLM interface.
- Default V0.1 model: `gpt-5.6-luna`, configurable through `OPENAI_MODEL`.
- Required LLM configuration: `OPENAI_API_KEY` and `OPENAI_MODEL`.
- Semantic memory: local ChromaDB behind a replaceable embeddings interface and memory-store boundary.
- Voice: OpenAI audio API initially for STT and TTS, behind STT/TTS interfaces; push-to-talk only.
- Browser: Playwright with Chromium initially.
- Windows/system automation: Python `subprocess` only for explicitly allowlisted commands, with pywinauto for supported GUI/application automation where required.
- Validation: Pydantic schemas.

Unrestricted shell execution and unrestricted filesystem access are not part of V0.
The OpenAI SDK is confined to the OpenAI adapter/provider implementation. V0.1 does not use direct httpx calls for the OpenAI API, and API keys are never hardcoded or logged.

## 4. v0 Scope

### Included
- Python modular monolith
- cloud LLM API
- SQLite
- vector-memory abstraction/local vector capability
- STT and TTS
- tool registry
- browser automation
- basic Windows/system control
- filesystem tools
- memory manager
- permission/confirmation layer
- logging
- configuration through environment variables
- tests

### Explicitly excluded from v0
- training a custom foundation model
- always-on autonomous operation
- complex multi-agent architecture
- continuous screen-vision loop
- personality cloning
- fully autonomous long-running tasks
- unnecessary local model downloads

## 5. Initial User Experience

Example:

User: "Open Brave."

JARVIS determines the appropriate tool, validates it, executes it, receives the result, and responds.

User: "Remember that EventX is my event-management project."

JARVIS stores an explicit long-term memory.

User: "What is my event-management project?"

JARVIS retrieves the relevant memory and answers.

User: "Delete that file."

If the action is destructive, JARVIS resolves the context and requests confirmation before execution.

## 6. Design Principles

1. Modular boundaries.
2. Least privilege for tools.
3. LLM output is untrusted input.
4. Destructive operations require confirmation.
5. Structured facts belong in structured storage.
6. Semantic information belongs in semantic retrieval.
7. Provider-specific code must remain behind interfaces.
8. Prefer simple local infrastructure during v0.
9. Build working vertical slices instead of speculative infrastructure.
10. Every major architectural decision is documented.

## 7. Future Direction

Future versions may add:
- wake-word detection
- local LLMs
- richer screen understanding
- deeper browser/OS control
- project-aware coding assistance
- scheduled/background tasks
- advanced planning
- personalized personality and communication style
- additional integrations
