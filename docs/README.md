# JARVIS

Personal desktop AI assistant.

## Current Target

JARVIS v0 is a local Python modular monolith with:
- cloud LLM reasoning
- SQLite structured storage
- semantic memory abstraction
- controlled computer tools
- text and voice interfaces
- explicit permission boundaries

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

## Documentation

Read in this order:

1. PROJECT.md
2. ARCHITECTURE.md
3. DEVELOPMENT.md
4. DATABASE.md
5. TOOLS.md
6. MEMORY.md
7. SECURITY.md
8. DECISIONS.md
9. ROADMAP.md

## Core Rule

The LLM proposes reasoning and actions.

The JARVIS application validates, authorizes, executes, and records those actions.
