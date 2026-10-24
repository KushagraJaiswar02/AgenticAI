# JARVIS — Roadmap

## V0 Implementation Stages

V0 is one release implemented through the following incremental internal stages. These are not separate products.

## V0.4 safety milestone

- deterministic risk metadata and policy engine
- exact, expiring confirmation tokens
- canonical filesystem allowlist boundary
- bounded filesystem tools
- structured sanitized tool audit records
- arbitrary shell execution remains forbidden

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
- reliable browser interaction: click, type, scroll, wait for page state, and handle navigation
- extract and inspect actual page content instead of relying on screenshots or page titles alone
- handle dynamic content, redirects, timeouts, stale selectors, and recoverable navigation failures
- distinguish web research from interactive browser control; choose the appropriate method per task
- support goal-oriented tasks such as reading official documentation, finding and playing a requested video or playlist, and researching/comparing products and current listings
- verify outcomes before reporting success; never claim a page was read, a video played, or a price verified without evidence
- browser evaluation suite covering documentation lookup, media navigation, product comparisons, price discovery, and common failure cases

## V0.5 — Bounded multi-step execution

- orchestrator-owned execution state
- bounded tool sequences
- tool-result feedback to the LLM
- structured failure and recovery behavior
- explicit task plan and step ledger: original goal, completed steps, current step, pending steps, observations, and blockers
- pass verified outputs between steps; do not silently treat assumptions as facts
- stop, retry safely, re-plan, or ask the user when a step fails or the next action is unclear
- verify task completion against the user's original request

## V0.6 — Push-to-talk voice

- OpenAI audio API STT
- OpenAI audio API TTS
- replaceable STT/TTS interfaces
- push-to-talk only
- no wake-word detection

## Later versions

- **Context-aware task recovery:** if Jarvis gets stuck, loses the thread, or detects that its plan no longer matches observed results, it should recover from the saved task state rather than guess or restart blindly
- **Progress-aware clarification:** explain what has been completed, identify the exact blocker, and ask a focused question such as, “I've completed X. I couldn't complete Y. Your original request also included Z—should I try another route?”
- **Hallucination and drift detection:** compare proposed actions and completion claims with the original request, execution history, and tool results; if evidence is missing or contradictory, pause and disclose uncertainty
- resume an interrupted task from its last verified checkpoint when safe
- screen understanding
- richer Windows control
- scheduled/background tasks
- personality and deeper personalization
- long-running autonomous tasks
- additional local LLMs
- additional integrations

## Rule

Do not skip directly to later versions because a feature sounds impressive. Each V0 stage should produce a working, testable system. Task recovery must be designed around explicit state and evidence, not just a prompt telling the model to “try harder.”
