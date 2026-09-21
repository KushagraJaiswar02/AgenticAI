# JARVIS — Architecture

## 1. Architectural Style

JARVIS v0 is a modular monolith.

The application runs locally and consists of independent modules with clear interfaces.

The locked runtime is Python 3.12 on Windows. SQLite is accessed through SQLAlchemy. Pydantic defines and validates application, tool, and provider-boundary schemas. Configuration is loaded with python-dotenv, HTTP integrations use httpx, and tests use pytest.

V0 is one release delivered through incremental internal stages. The first stage is a text-only vertical slice. Voice, browser control, semantic memory, filesystem automation, and bounded multi-step execution are added progressively without changing the modular-monolith boundary.

```text
User
 |
 v
Input Layer
 |-- Text
 |-- STT
 |
 v
Orchestrator
 |
 +--> Context Builder
 |       +--> Conversation State
 |       +--> Memory Retrieval
 |
 +--> LLM Interface
 |
 +--> Tool Registry
 |       +--> System Tools
 |       +--> File Tools
 |       +--> Browser Tools
 |       +--> Memory Tools
 |
 +--> Permission Manager
 |
 +--> Task/Execution State
 |
 v
Response Layer
 |-- Text
 |-- TTS
```

## 2. Main Components

### Input Layer
Converts user input into normalized text.

v0 supports:
- typed input
- speech-to-text

### Orchestrator
Central controller.

Responsibilities:
- receive requests
- maintain task/conversation state
- build LLM context
- expose available tools
- validate tool calls
- request confirmations
- execute tools
- feed results back to the LLM
- produce final responses

The orchestrator must not contain implementation details for individual tools.

### LLM Interface

A provider-neutral interface.

Conceptually:

```text
generate(prompt) -> normalized LLM response
```

The active V0 default is Gemini via the official Google Gen AI SDK. OpenAI remains available as a fallback through its own adapter. Provider logic stays inside each adapter; the orchestrator depends only on the LLM interface and never branches on provider-specific implementation details.

The active default model is `gemini-3.8-flash`, configured through `GEMINI_MODEL`. The fallback model is `gpt-5.6-luna`, configured through `OPENAI_MODEL`. Both APIs are accessed through provider adapters and environment configuration. No direct API calls are embedded in application business logic.

### Tool System

Tools are explicit capabilities.

Each tool should define:
- unique name
- description
- input schema
- execution function
- risk level
- confirmation requirement

The LLM may request tools, but the orchestrator validates the request before execution.

Tool schemas and arguments are validated with Pydantic. Windows/system tools use Python `subprocess` only for explicitly allowlisted commands and pywinauto for supported GUI/application automation where required. Unrestricted shell execution and unrestricted filesystem access are prohibited.

### Memory System

Two broad classes:

1. Structured memory
   - exact facts
   - preferences
   - entities
   - state
   - configuration

2. Semantic memory
   - conversation fragments
   - documents
   - notes
   - unstructured knowledge

SQLite is the authoritative structured store. Semantic retrieval is implemented behind a vector-store interface.

The initial local vector store is ChromaDB. The memory manager communicates with a replaceable semantic-memory/embedding interface and must not depend directly on ChromaDB APIs. Embeddings are also replaceable behind an interface.

### Permission Manager

Controls potentially dangerous operations.

Example risk levels:
- SAFE
- LOW
- MODERATE
- DESTRUCTIVE

Destructive actions require explicit user confirmation unless a future policy explicitly allows otherwise.

### Response Layer

Converts internal results into user-facing responses.

Supports:
- text
- TTS

The initial TTS implementation uses the OpenAI audio API behind a TTS interface. STT likewise uses the OpenAI audio API behind an STT interface. Voice is push-to-talk only; no wake-word component is included.

## 3.1 Dependency Boundaries

- The input layer depends on input interfaces and delivers normalized text to the orchestrator.
- The orchestrator depends on interfaces for LLM, tools, memory, persistence, permissions, and response handling; it does not contain provider or tool implementation details.
- The LLM adapter is the only component that depends on the official OpenAI Python SDK and Responses API. It reads `OPENAI_API_KEY` and `OPENAI_MODEL` through configuration and implements the provider-neutral LLM interface.
- The tool registry exposes validated Pydantic tool schemas and metadata to the orchestrator; tools never call the LLM directly.
- The permission manager runs outside the LLM and must approve or reject every operation requiring authorization before execution.
- System tools depend on allowlisted subprocess execution and, where required, pywinauto; they cannot invoke arbitrary shell commands.
- Filesystem tools depend on configured safe boundaries and cannot access unrestricted paths.
- Browser tools depend on Playwright and Chromium initially; browser content is treated as untrusted data.
- The persistence layer depends on SQLAlchemy and SQLite and owns authoritative application state.
- The memory manager depends on persistence plus replaceable semantic-memory and embedding interfaces; it does not import ChromaDB directly.
- The initial ChromaDB adapter implements the semantic-memory boundary locally.
- Voice adapters depend on the OpenAI audio API but expose only STT/TTS interfaces to the rest of the application.
- Response handling depends on text output and the TTS interface, not on a concrete TTS provider.

## 3. Execution Flow

```text
1. User provides request.
2. Input layer normalizes it.
3. Orchestrator loads relevant state.
4. Context builder retrieves relevant memory.
5. LLM receives request + context + available tools.
6. LLM returns either a response or tool request.
7. Orchestrator validates the request.
8. Permission manager evaluates risk.
9. Tool executes if permitted.
10. Tool result returns to orchestrator.
11. Result is supplied back to LLM when further reasoning is required.
12. LLM generates final response.
13. Response layer outputs text and/or speech.
14. Relevant state/memory is persisted.
```

## 4. Multi-Step Tasks

JARVIS may execute a sequence of tools.

Example:

```text
"Open Brave and search YouTube for React tutorials."

Plan:
1. open_application(Brave)
2. open/search browser
3. search for React tutorials
4. report result
```

The orchestrator owns execution state. The LLM proposes reasoning and actions; it does not directly execute them.

## 5. Failure Handling

Every tool must return structured success/failure information.

Failures must not be silently swallowed.

Example:

```json
{
  "success": false,
  "error": "Application not found",
  "retryable": false
}
```

The LLM may explain or recover from the failure when appropriate.

## 6. Provider Independence

LLM, STT, TTS, vector store, and browser implementations should be replaceable through interfaces.

Avoid coupling core business logic to a single provider.

## 7. v0 Boundary

Do not introduce:
- microservices
- distributed queues
- multiple autonomous agents
- complex event buses

unless an actual requirement emerges.
