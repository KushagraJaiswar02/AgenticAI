# JARVIS

JARVIS is a local Python terminal assistant with a provider-neutral LLM interface. Gemini is the active default provider, and OpenAI remains available as a fallback.

## Current milestone

V0.2 adds a small, explicit tool layer. The first tool is a weather lookup that is executed only when the LLM requests it through the normal provider abstraction. The orchestrator remains provider-agnostic and does not branch on provider-specific logic.

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
