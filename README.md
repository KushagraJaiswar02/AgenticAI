# JARVIS

JARVIS is a Python terminal assistant for conversation, research, and controlled desktop tasks. It connects to Gemini, OpenRouter, Groq, or a local Ollama instance and gives the model access to a set of validated tools. The application, rather than the model, decides whether a tool may run and when an action needs confirmation.

The project targets **Windows and Python 3.12**. It is under active development; voice interaction and semantic long-term memory are project goals, not features of the current terminal entry point.

## What it can do

- Chat with a cloud or local model, with configurable provider fallback.
- Get the time, date, weather, and basic system information; evaluate arithmetic; and open the allowlisted Brave application.
- Work with files inside configured filesystem boundaries, subject to validation and confirmation for consequential actions.
- Open and inspect web pages and perform bounded browser research using Playwright.
- Search and inspect YouTube videos through the YouTube Data API, and summarize or compare transcripts supplied by the user.
- Fall back to a small deterministic intent router for certain simple tool requests when a provider fails recoverably. This is not a general-purpose language model.

Available tool calls depend on the selected provider and model. See [tool documentation](docs/TOOLS.md) for details and examples.

## Quick start

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` for the default provider. Check that `GEMINI_MODEL` names a model available to your account, or select another provider as described below. Keep `.env` private; do not commit API keys.

Start the assistant:

```powershell
python -m app.main
```

Enter a message at the prompt. Type `exit` or `quit` to stop. Try `What time is it?`, `Open Brave`, or `Search YouTube for Python tutorials` (the last requires a YouTube API key).

For browser tools, install Playwright's Chromium browser once after installing the dependencies:

```powershell
python -m playwright install chromium
```

## Configuration

Settings are read from environment variables and `.env`. Start with [.env.example](.env.example).

### Model providers

`LLM_PROVIDER` selects the preferred provider: `gemini`, `openrouter`, `groq`, or `ollama`. Cloud providers need their corresponding key and model:

- Gemini: `GEMINI_API_KEY` and `GEMINI_MODEL`.
- OpenRouter: `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.
- Groq: `GROQ_API_KEY` and `GROQ_MODEL`.

`LLM_PROVIDER_ORDER` controls the fallback sequence; the example uses `gemini,openrouter,groq,ollama`. Set `LLM_FALLBACK_ENABLED=false` to disable provider fallback. Provider availability, model access, native tool support, and any usage charges depend on the service and model selected.

For local inference, install and start [Ollama](https://ollama.com/), then download the models you intend to use. For the default text and tool mappings:

```powershell
ollama pull llama3.2:3b
ollama pull qwen3:4b
ollama list
```

Set `LLM_PROVIDER=ollama`. The default server is `http://localhost:11434`; configure it with `OLLAMA_BASE_URL`. The general, coding, tool, and vision model mappings can be changed using the `OLLAMA_*_MODEL` variables in `.env.example`. The default vision mapping is `gemma3:4b`; install it separately if using image input through the provider interface. JARVIS does not download models for you.

Set `JARVIS_OFFLINE_MODE=true` to restrict model requests to Ollama and disable YouTube API access. Other tools, such as weather and browser tools, can still access the network.

### Filesystem access

Set `JARVIS_WORKSPACE` to an existing directory for the default file context, and `JARVIS_ALLOWED_ROOTS` to the directories tools may access. Separate multiple roots with the platform path separator (`;` on Windows). The workspace must be inside an allowed root.

```dotenv
JARVIS_WORKSPACE=C:\Projects\MyProject
JARVIS_ALLOWED_ROOTS=C:\Projects\MyProject
```

If these settings are omitted, the workspace defaults to the **parent of the project root**, and that workspace becomes the allowed root. Set both explicitly if you want a narrower boundary. The project root defaults to the current working directory and can be set with `JARVIS_PROJECT_ROOT`.

### YouTube

Enable YouTube Data API v3 in a Google Cloud project and set `YOUTUBE_DATA_API_KEY` in `.env`. This is separate from model-provider credentials. Search and inspection use API quota; automatic transcript retrieval is not implemented. You can supply transcript text or a workspace file for summaries and comparisons. See the [YouTube tool guide](docs/TOOLS.md#youtube-research-tools) for syntax and limits.

## Safety and privacy

Tool arguments are validated and evaluated by a policy engine before execution. Actions that require approval prompt for confirmation tied to the specific operation; high-impact deletion requires a stronger confirmation phrase. Filesystem access is restricted to configured roots, arbitrary shell execution is not exposed, and application launching is allowlisted. Browser and other external content is treated as untrusted data. See [SECURITY.md](docs/SECURITY.md) for the design and limitations.

Cloud model requests and research synthesis may send user-provided content to the configured provider. Use local Ollama if that better suits your data requirements. A local SQLite database is initialized as `jarvis.db` in the working directory.

## Development and tests

```powershell
python -m pytest -q
```

The normal test suite uses fake providers and mocked external services. Tests marked `integration`, when present, can be selected explicitly with `python -m pytest -m integration`. Optional provider diagnostics make network requests and are not part of startup:

```powershell
python -m app.provider_health
```

The application code lives in [`app/`](app/), tests in [`tests/`](tests/), and design notes in [`docs/`](docs/). Useful starting points are the [architecture](docs/ARCHITECTURE.md), [development guide](docs/DEVELOPMENT.md), and [roadmap](docs/ROADMAP.md). Some design documents describe planned capabilities as well as implemented ones.
