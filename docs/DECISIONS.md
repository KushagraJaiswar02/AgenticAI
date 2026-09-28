# JARVIS — Architecture Decision Record

## ADR-001 — Python

### Decision
Use Python for the JARVIS application core.

### Reason
Python provides strong ecosystem support for LLM APIs, automation, voice, browser control, AI/ML, and rapid iteration.

---

## ADR-002 — Modular Monolith

### Decision
JARVIS v0 is a modular monolith.

### Reason
The project is a single-user desktop assistant. Microservices would add operational complexity without solving a current problem.

---

## ADR-003 — Cloud LLM First

### Decision
Use a cloud LLM as the initial reasoning engine.

### Reason
Local model hosting would add hardware, memory, performance, and model-management constraints. The LLM provider remains behind an abstraction so local models can be added later.

---

## ADR-004 — SQLite as Primary Database

### Decision
Use SQLite for structured persistent state.

### Reason
JARVIS is initially a local desktop application. SQLite is lightweight, reliable, portable, and sufficient for v0.

---

## ADR-005 — Vector Store Behind an Interface

### Decision
Do not hard-code the architecture around a specific vector database.

### Reason
The required retrieval scale is unknown. A local vector implementation is preferable initially; a dedicated database can be introduced if justified.

---

## ADR-006 — LLM Does Not Directly Execute Tools

### Decision
The LLM may request tools, but the orchestrator validates and executes them.

### Reason
LLM output is probabilistic and must be treated as untrusted input. This boundary is essential for security and predictable behavior.

---

## ADR-007 — Confirmation for Destructive Actions

### Decision
Destructive actions require explicit confirmation by default.

### Examples
- deleting files
- destructive shell commands
- irreversible system changes

### Reason
An assistant with computer access must fail safely.

---

## ADR-008 — Structured vs Semantic Memory

### Decision
Use structured storage for exact facts and vector retrieval for semantic/unstructured information.

### Reason
Vector search is not a substitute for authoritative structured data.

---

## ADR-009 — No Custom Model Training in v0

### Decision
Do not train a custom foundation model for v0.

### Reason
The primary engineering problem is building the agent system around a capable model, not training a new base model.

---

## ADR-010 — No Multi-Agent Architecture in v0

### Decision
Use one orchestrator and explicit tools.

### Reason
Multiple agents would increase complexity before there is evidence that they are required.

---

## ADR-011 — Text Before Voice

### Decision
Text interaction must work before voice becomes a core dependency.

### Reason
This makes debugging deterministic and separates reasoning/tool problems from audio problems.

---

## ADR-012 — Least Privilege

### Decision
Tools receive only the permissions necessary for their operation.

### Reason
Computer control introduces real-world risk. Capabilities must be constrained independently from model reasoning.

---

## ADR-013 — Python 3.12

### Decision
Use Python 3.12 on Windows desktop for V0.

### Reason
The project requires a stable, explicitly supported runtime for the modular monolith and its automation, LLM, voice, browser, validation, and persistence integrations.

---

## ADR-014 — SQLite + SQLAlchemy

### Decision
Use SQLite as the authoritative database and SQLAlchemy as the persistence boundary.

### Reason
The application is local and single-user in V0. SQLAlchemy provides a focused persistence boundary while SQLite remains lightweight and portable.

---

## ADR-015 — Pydantic Validation

### Decision
Use Pydantic for application, provider-boundary, tool, and tool-result schemas and validation.

### Reason
LLM output and external input are untrusted and must be validated before they influence execution.

---

## ADR-016 — OpenAI Responses API and Official SDK

### Decision
Use OpenAI as the initial provider. For V0.1 LLM reasoning, use the official OpenAI Python SDK and the OpenAI Responses API. Use `gpt-5.6-luna` as the documented/default initial model, configurable through `OPENAI_MODEL`.

The API key is supplied through `OPENAI_API_KEY` and is never hardcoded or logged. V0.1 does not use direct httpx calls for the OpenAI API.

JARVIS exposes its own provider-neutral LLM interface. The orchestrator depends only on that interface. The OpenAI SDK and Responses API exist only inside the OpenAI adapter/provider implementation.

### Reason
This locks a concrete, supported V0.1 integration while preserving provider-neutral LLM, STT, and TTS interfaces so providers can be replaced later without changing orchestration or core business logic.

---

## ADR-017 — ChromaDB Local Vector Store

### Decision
Use ChromaDB locally as the initial semantic vector store.

### Reason
V0 needs local semantic retrieval without a remote vector service. The memory manager must use a semantic-memory interface, and embeddings must use a replaceable interface; no core module may couple directly to ChromaDB.

---

## ADR-018 — Playwright + Chromium

### Decision
Use Playwright with Chromium initially for browser automation.

### Reason
This provides the required V0 URL navigation, search, and browser actions behind a browser-tool boundary without expanding to multiple browser engines.

---

## ADR-019 — Allowlisted Subprocess Execution

### Decision
Use Python `subprocess` only for explicitly allowlisted commands. Use pywinauto for supported GUI/application automation where required. Do not expose unrestricted shell execution or unrestricted filesystem access.

### Reason
Computer control is a high-risk capability. Explicit allowlists and safe filesystem boundaries enforce least privilege outside the LLM.

---

## ADR-020 — Incremental V0 Implementation

### Decision
V0 is one release delivered through internal stages: V0.1 core text pipeline, V0.2 tools and permissions, V0.3 persistence and memory, V0.4 filesystem and browser control, V0.5 bounded multi-step execution, and V0.6 push-to-talk voice.

### Reason
Text-first vertical slices make orchestration and security deterministic while allowing the complete documented V0 capability set to be added progressively. These stages do not create separate products or change the modular-monolith architecture.

---

## ADR-021 — Two-Model Ollama Fallback

### Decision

When Ollama is selected, use `llama3.2:3b` for simple or ambiguous requests and
`qwen3:4b` for clearly complex requests. Keep both model names and their
timeouts configurable through environment settings.

### Reason

Local testing showed that Qwen's inference latency remains high for trivial
requests even when `/no_think` changes the exposed reasoning output. A smaller
model is therefore the fast path; Qwen remains available for deeper reasoning.

---

## ADR-022 — Safe Application Launching

### Decision

Application launching is exposed only through an allowlisted tool. The initial
allowlist contains `brave` and maps it to known Brave executable locations.
The tool uses `subprocess.Popen(..., shell=False)` without accepting arbitrary
commands, arguments, or executable paths.

### Reason

Desktop execution is a privileged capability. An explicit mapping preserves
extensibility while preventing shell injection and arbitrary process launch.

---

## ADR-023 — Explicit Multi-Provider Failover

### Decision

Support Gemini, OpenAI, Groq, Cerebras, OpenRouter, Mistral, Cohere, and
Ollama behind the existing provider-neutral interface. Provider priority is
configured only through `LLM_PROVIDER_ORDER`; providers missing a required API
key or model are skipped before a request. Each provider owns its SDK adapter,
timeout, response normalization, and capability differences.

### Reason

Independent adapters provide resilience without coupling the orchestrator to
vendor APIs. A single attempt is made per provider before moving to the next,
avoiding multiplied latency and preserving the existing tool-continuation
stickiness rule.

---

## ADR-024 — Centralized Deterministic Tool Policy

### Decision

All tool calls pass through `PolicyEngine` and the `ToolRegistry` execution boundary. Risk metadata is application-owned; confirmation tokens are exact-action, expiring, and single-use. Forbidden capabilities have no approval path.

### Reason

LLM output is untrusted input. A single local policy boundary prevents provider and local-router paths from bypassing safety checks while keeping individual tools focused on validation and execution.
