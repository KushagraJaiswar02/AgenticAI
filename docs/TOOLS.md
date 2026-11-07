# JARVIS — Tool System

## 1. Purpose

Tools are controlled capabilities exposed to the orchestrator and, indirectly, to the LLM.

## 2. Tool Contract

Every tool should define:

```text
name
description
input schema
risk level
confirmation policy
execute()
```

## 3. Initial Tool Groups

### System
- open application
- close application
- system information
- execute approved command

### Files
- list files
- read file
- create file
- move/copy file
- delete file with confirmation

### Browser
- `open_browser`
- `open_url`
- `get_current_page`
- `search_web` (default engine: DuckDuckGo)
- `get_page_content` (read-only visible text from the current page)

`open_url` initializes the controlled browser through the policy-checked
`open_browser` tool when needed. It reuses an existing browser session.

`get_page_content` accepts `max_chars` from 1 to 12,000 (default 4,000) and
`wait_ms` from 0 to 3,000 (default 500) for dynamic rendering. It returns
sanitized page metadata and untrusted text. The provider can summarize the
observation, and the turn ends without executing further tool calls.

Back, forward, and reload navigation are not implemented.

### Memory
- remember
- recall
- remove memory

### Utility
- calculator
- time
- web lookup

## 4. Tool Safety

The LLM cannot bypass tool validation.

Tool arguments must be validated before execution.

High-risk tools must require confirmation.

Never expose arbitrary unrestricted shell execution without a permission boundary.

## 5. Tool Results

Use structured results:

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

or:

```json
{
  "success": false,
  "data": null,
  "error": "reason",
  "retryable": false
}
```

## YouTube research tools

Enable YouTube Data API v3 for a Google Cloud project and set `YOUTUBE_DATA_API_KEY` in the environment or local `.env`. Use a key restricted to the YouTube Data API. The key is separate from model-provider credentials. Offline mode disables YouTube API access.

- `youtube_search(query, limit=10)` returns up to 25 validated video results, with their canonical links and available search metadata. The CLI accepts `Search YouTube for React hooks tutorials` or `Find React hooks tutorials on YouTube`.
- `youtube_inspect(video, transcript_language=None)` returns metadata and a separate transcript status. `video` accepts an ID or a watch, short share, Shorts, embed, or live URL. The CLI accepts `Inspect YouTube https://youtu.be/M7lc1UVf-VE`.

Both tools use the official API through the existing registry. Results are rendered locally with YouTube attribution. The default search quota is 100 calls per project per day. Pagination and retry attempts consume quota. JARVIS limits searches to three pages and caps a turn at six API attempts within a 15-second request budget. Inspection uses the video's metadata endpoint, which Google lists at one general quota unit per call. [YouTube quota reference](https://developers.google.com/youtube/data/determine_quota_cost)

For a transcript you are permitted to share, use `Summarize https://youtu.be/M7lc1UVf-VE Language: en Transcript: pasted text here` or `Summarize https://youtu.be/M7lc1UVf-VE Transcript file: "captions.vtt"`. Transcript files use the existing allowed workspace and may be plain text, VTT, or SRT. Use a file for multiline captions in the single-line CLI.

Compare up to five videos with explicit file bindings. The language in parentheses is optional. Bind each video ID or URL to its own quoted workspace file.

```text
Compare the examples in these videos. Transcripts: M7lc1UVf-VE (en) = "first.vtt" | bHQqvYy5KYo (es) = "second.srt"
```

For mixed research, put the web URL in the question before `Transcripts:`. Add `Include YouTube metadata` there to display API metadata separately. Every usable video must receive its own citation. Failed or conflicting bindings are identified by source number and video ID when known. The existing research evidence limit is shared across videos, with up to 12,000 characters per transcript. Mixed research reserves at least half the evidence budget for web excerpts.

User-supplied evidence goes to the configured synthesis provider, with tools disabled. Citations use collected evidence IDs and verified cue timestamps. Search results and API metadata stay local. Transcript status and provenance are separate from API metadata, and the caption-availability flag does not grant access to transcript text. OAuth and automatic transcript retrieval are outside Phase 1. [YouTube caption download](https://developers.google.com/youtube/v3/docs/captions/download)
