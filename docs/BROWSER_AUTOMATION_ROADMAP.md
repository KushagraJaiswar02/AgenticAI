# Browser automation roadmap

- **Initial provider failover — VERIFIED.** The reported CLI run exhausted Gemini quota, received an OpenRouter 429, and continued with Groq under the original request ID.
- **Active-conversation provider failover — NOT VERIFIED.** Transferring a tool-enabled conversation to a new provider remains outside the verified path. Text-only synthesis recovery is tracked separately in the [provider reliability milestone](ROADMAP.md#provider-reliability-milestone--browser-synthesis-recovery).
- **Browser startup — VERIFIED.** The reported CLI run started the browser automatically.
- **URL navigation — BASICALLY VERIFIED.** Direct navigation and redirects work. The historical 404 destination was redacted. Controlled Chromium tests cover structured HTTP failures, timeouts, and recovery through a same-site landing page and an observed link. General destination selection and recovery on public sites still need verification.
- **Page extraction and summarization — VERIFIED.** Previous live Chromium tests delivered extracted text to Gemini and produced source-grounded answers. See the [browser milestone](ROADMAP.md#browser-milestone-1--intelligent-navigation-and-targeted-extraction).

## Research intelligence

- **Structured topic extraction — VERIFIED.** Targeted excerpts preserve headings, paragraphs and lists. Real Chromium coverage includes late sections of a controlled technical manual and public React and Wikipedia pages.
- **Contextual internal-link discovery — VERIFIED.** Visible links are ranked by label, destination and nearby text. Live React research followed observed reference links. A guessed same-site path stayed blocked and received one opportunity to select an observed alternative.
- **Bounded scan continuation — VERIFIED.** In Chromium, a later scan window recovered three independent sections beyond the first 96,000 scanned characters at the existing default output limit. The model must use the returned offset. Node and turn budgets still apply.
- **Source coverage and prompt-injection safeguards — IMPLEMENTED.** The orchestrator tracks unresolved topics across reads and reports an evidence-free miss directly. Providers receive trusted research instructions separately from untrusted observations. Scope denials and 404 recovery remain in place.
- **Compact terminal synthesis — VERIFIED.** After a scripted Chromium traversal of the public React documentation, live Groq summarized the observed pages using the compact evidence payload. Provider and backend identity checks still apply.
- **Broader research coverage — NOT STARTED.** Cross-site link discovery and independent verification of every generated claim are outside this milestone.

## Autonomous research orchestration

- **VERIFIED.** Requests beginning with “Research” use a request-local objective and evidence ledger. The provider produces a text plan and a final answer. The orchestrator drives navigation through the existing registry, browser, extraction and audit paths. Other tool turns retain their native continuation and provider ownership.
- **VERIFIED.** Seed URLs come from the user or visible result anchors on a verified search endpoint. Search selects one source origin. Following links stays within the observed page origin, and a scoped search cannot introduce a new origin. Failed destinations are not retried. Browser policy and confirmation requirements still apply.
- **IMPLEMENTED.** Coverage distinguishes supporting excerpts, partial matches and unresolved objectives. Research can follow a reference entry, resume a scan, and keep collected excerpts if browsing or synthesis cannot finish. An eligible provider can synthesize the existing evidence without replaying browser actions.

| Limit | Environment variable | Default |
| --- | --- | --- |
| Successful navigations, including search pages | `JARVIS_RESEARCH_MAX_PAGES` | 8 |
| Browser tool steps and intercepted navigation events | `JARVIS_RESEARCH_MAX_ITERATIONS` | 24 |
| Navigation attempts, redirect hops, and observed popup attempts | `JARVIS_RESEARCH_MAX_NAVIGATIONS` | 12 |
| Stored excerpt characters | `JARVIS_RESEARCH_MAX_EVIDENCE_CHARS` | 16,000 |
| Deadline in seconds | `JARVIS_RESEARCH_MAX_SECONDS` | 90 |
| Provider attempts, including failover | `JARVIS_RESEARCH_MAX_PROVIDER_CALLS` | 8 |

The deadline is checked between operations and passed to supported Playwright and provider requests. It does not forcibly interrupt an operating-system or driver stall.

### Browser navigation authorization

An explicit `open_url` or `search_web` grants the controlled tab the destination's HTTP origin. The same tab can navigate within that origin. A same-host HTTP-to-HTTPS upgrade replaces the grant. Opening a different origin requires another explicit tool navigation. Chromium's DevTools Fetch interceptor pauses main-frame document requests before sending them, including each redirect hop. A separate Playwright context route blocks requests from new tabs, and the browser closes those tabs. Chromium's native popup blocker stays enabled.

Autonomous research charges its first attempt at the tool boundary. Redirects, page-initiated navigation, and popups observed by the browser context spend the same request's navigation and iteration budget. Each explicit navigation in an ordinary tool turn allows up to 12 follow-up navigation events. Both kinds of turn expire their grants on return, exception, or cancellation. The current page can still be read after the turn. Another explicit tool navigation is needed to open a new document.

A pending explicit navigation approves the requested document and its redirect chain. A failed navigation revokes its grant. If Chromium loses its interception session, JARVIS shuts down the gateway and closes the controlled tab. The next explicit navigation can launch a fresh session. Browser startup requires Playwright 1.59 or newer for interception-session loss notification. The SOCKS gateway still enforces public-address and DNS restrictions. Public iframe and subresource requests remain outside the document-origin grant.

Earlier autonomous research verification covered 432 passing tests with Chromium enabled and two skipped opt-in public search smoke tests. A separate unscripted live React run read the official `useState`, `useEffect`, and Rules of Hooks pages in 16.64 seconds. The Hulk run read dedicated Marvel profiles and reported edition and excerpt limitations in 12.75 seconds. A three-page Chromium fixture completed with live planning and synthesis in 6.85 seconds, with one visit per page and no navigation from the embedded instructions. Provider outage recovery after the first extraction was tested with an injected outage and real Chromium. Live transfer of a tool-enabled transcript remains outside this milestone.

**Next milestone.** Verify individual claims against their excerpts and improve evidence selection for broad objectives and ambiguous sources.
