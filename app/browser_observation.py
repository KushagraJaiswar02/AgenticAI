"""Bounded visible-text extraction for the controlled Chromium page."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, unquote, urlsplit

from app.browser_security import (
    BlockedBrowserDestination, InvalidBrowserURL, safe_browser_title, safe_browser_url, validate_browser_url,
)


DEFAULT_CONTENT_CHARS = 4000
MAX_CONTENT_CHARS = 12000
DEFAULT_RENDER_WAIT_MS = 500
MAX_RENDER_WAIT_MS = 3000
MAX_RESEARCH_TOPICS = 6
MAX_SCAN_OFFSET = 288000


READ_PAGE = r"""({maxChars, targeted, discoverLinks, scanOffset, maxScanOffset, searchSelector}) => {
    const url = location.href;
    if (url !== 'about:blank' && !['http:', 'https:'].includes(location.protocol)) {
        return {url, inaccessible: true};
    }
    const rawTitle = document.title;
    const title = rawTitle.length <= 512 ? rawTitle : null;
    const root = document.documentElement;
    if (!root) return {url, title, content: '', truncated: false};

    const excluded = [
        'head', 'script', 'style', 'noscript', 'template', 'footer',
        'form', 'input', 'textarea', 'select', 'option',
        'button', 'output', 'iframe', 'svg', 'canvas', '[contenteditable]',
        '[hidden]', '[inert]', '[aria-hidden="true"]', '[data-private]', '[data-sensitive]',
        '[role="contentinfo"]', '[role="form"]', '[role="textbox"]'
    ].join(',');
    const navigation = 'nav,aside,body > header,[role="navigation"],[role="banner"],[role="complementary"]';
    const parts = [];
    const sections = [];
    const links = new Map();
    const headingSelector = 'h1,h2,h3,h4,h5,h6,[role="heading"][aria-level]';
    const blockSelector = headingSelector + ',p,li,dt,dd,pre,blockquote,td,th';
    let heading = null;
    let section = {level: 0, heading: [], blocks: []};
    let block = null, blockNode = null;
    if (targeted) sections.push(section);
    let count = 0, visited = 0, scanned = 0, truncated = false, scanTruncated = false;
    let skip = scanOffset || 0, nextScanOffset = null;
    let node = root;
    while (node) {
        if (++visited > 12000) {
            truncated = scanTruncated = true;
            break;
        }
        let descend = true;
        if (node.parentElement?.matches('details:not([open])') && node.localName !== 'summary') {
            descend = false;
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            const style = getComputedStyle(node);
            descend = !node.matches(excluded) && (discoverLinks || !node.matches(navigation))
                && style.display !== 'none'
                && style.visibility !== 'hidden' && style.visibility !== 'collapse'
                && style.opacity !== '0' && style.contentVisibility !== 'hidden'
                && (style.display === 'contents' || node.getClientRects().length > 0);
        } else if (node.nodeType === Node.TEXT_NODE) {
            const navigationOnly = Boolean(node.parentElement?.closest(navigation));
            const start = navigationOnly ? 0 : Math.min(skip, node.length);
            if (!navigationOnly) skip -= start;
            const size = Math.min(node.length - start, navigationOnly ? 160 : 96000 - scanned);
            const text = node.substringData(start, size).replace(/\s+/gu, ' ').trim();
            if (!navigationOnly) scanned += size;
            if (text) {
                const anchor = node.parentElement?.closest('a[href]');
                if (targeted && !navigationOnly) {
                    const currentHeading = node.parentElement?.closest(headingSelector);
                    if (currentHeading && currentHeading !== heading) {
                        const level = Number(currentHeading.getAttribute('aria-level') || currentHeading.localName[1]);
                        section = {level: Math.max(1, Math.min(6, level || 2)), heading: [], blocks: []};
                        sections.push(section);
                        heading = currentHeading;
                        blockNode = block = null;
                    }
                    if (currentHeading) {
                        section.heading.push(text);
                    } else {
                        const owner = node.parentElement?.closest(blockSelector) || node.parentElement;
                        if (owner !== blockNode || !block) {
                            block = {
                                kind: owner?.closest('li') ? 'list' : (owner?.closest('pre') ? 'code' : 'text'),
                                parts: [], prose: false, linked: false,
                            };
                            section.blocks.push(block);
                            blockNode = owner;
                        }
                        block.parts.push(text);
                        if (anchor) block.linked = true;
                        else if (/[\p{L}\p{N}]/u.test(text)) block.prose = true;
                    }
                }
                if (discoverLinks) {
                    if (anchor && anchor.href.length <= 2048 && (!searchSelector || anchor.matches(searchSelector))) {
                        const known = links.get(anchor.href);
                        if ((!known && links.size < 240) || (known?.navigationOnly && !navigationOnly)) {
                            links.set(anchor.href, {
                                anchor, text: text.slice(0, 160),
                                block: navigationOnly ? null : block,
                                heading: navigationOnly ? null : section.heading,
                                navigationOnly,
                                blockPosition: block?.parts.length || 0,
                            });
                        } else if (known?.anchor === anchor && known.text.length < 160) {
                            known.text = (known.text + ' ' + text).slice(0, 160);
                        }
                    }
                }
                if (!navigationOnly) {
                    const room = maxChars - count - (parts.length ? 1 : 0);
                    const characters = Array.from(text);
                    if (room > 0) {
                        parts.push(characters.slice(0, room).join(''));
                        count += Math.min(room, characters.length) + (parts.length > 1 ? 1 : 0);
                    }
                    if (characters.length > room) {
                        truncated = true;
                        if (!targeted && !discoverLinks) break;
                    }
                }
            }
            if (!navigationOnly && start + size < node.length) {
                truncated = scanTruncated = true;
                const offset = (scanOffset || 0) + scanned;
                if (targeted && offset <= maxScanOffset) nextScanOffset = offset;
                break;
            }
        }

        if (descend && node.firstChild) {
            node = node.firstChild;
        } else {
            while (node !== root && !node.nextSibling) node = node.parentNode;
            node = node === root ? null : node.nextSibling;
        }
    }
    const candidates = Array.from(links, ([href, link]) => {
        const heading = (link.heading || []).filter(part => part.length <= 180).slice(0, 4).join(' ');
        const parts = link.block?.parts || [];
        const before = parts.slice(Math.max(0, link.blockPosition - 4), link.blockPosition)
            .filter(part => part.length <= 180).join(' ');
        const after = parts.slice(link.blockPosition, link.blockPosition + 2)
            .filter(part => part.length <= 180).join(' ');
        const nearby = [before, after].filter(Boolean).join(' ');
        return [href, link.text, [heading, nearby].filter(Boolean).join(' ')];
    });
    return {url, title, content: parts.join(' '), truncated, scanTruncated, nextScanOffset, sections, links: candidates};
}"""


def sanitize_page_text(content: str) -> str:
    """Apply the same browser URL policy to URLs quoted in visible page text."""
    return re.sub(r"""https?://[^\s<>"']+""", lambda match: safe_browser_url(match[0]), content, flags=re.I)


def browser_summary_input(prompt: str, observation: dict) -> str:
    """Serialize each excerpt once, without link and scan discovery metadata."""
    discovery = {"links", "section_headings", "next_source", "next_scan_offset", "recovery"}
    evidence = {key: value for key, value in observation.items() if key not in discovery}
    if "prior_observations" in evidence:
        evidence["prior_observations"] = [
            {key: value for key, value in item.items() if key not in discovery}
            for item in evidence["prior_observations"]
        ]
    return json.dumps({"user_request": prompt, "untrusted_page_observation": evidence}, ensure_ascii=False)


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[^\W_]{3,}", text.casefold())) - {"the", "and", "for", "with", "from"}
    return {
        word[:-3] + "y" if len(word) > 4 and word.endswith("ies")
        else word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is"))
        else word
        for word in words
    }


def select_page_sections(
    sections: list[dict], topics: list[str], max_chars: int,
) -> tuple[str, list[dict], list[str], bool]:
    """Prefer heading matches and their subsections, then nearby body text."""
    headings = [" ".join(section["heading"]) for section in sections]
    blocks = [
        [_section_block(item) for item in section["blocks"]]
        if "blocks" in section else [("text", " ".join(section["parts"]), True)]
        for section in sections
    ]
    heading_terms = [_terms(heading) for heading in headings]
    block_terms = [[_terms(text) if prose else set() for _, text, prose in group] for group in blocks]
    section_ends = []
    for index, section in enumerate(sections):
        end = index + 1
        while end < len(sections) and sections[end]["level"] > section["level"]:
            end += 1
        section_ends.append(end)
    excerpts: list[str] = []
    selected: list[tuple[dict, int]] = []
    missing = []
    for topic in topics:
        terms = _terms(topic)
        phrase = topic.strip("'\"").casefold()
        focal_blocks = [
            max(
                range(len(group)),
                key=lambda i: (
                    len(terms & block_terms[index][i]),
                    phrase in group[i][1].casefold(),
                    group[i][0] != "code",
                ),
                default=None,
            )
            for index, group in enumerate(blocks)
        ]
        scores = []
        for index, heading in enumerate(heading_terms):
            overlap = len(terms & heading)
            focal = focal_blocks[index]
            body_overlap = len(terms & block_terms[index][focal]) if focal is not None else 0
            strong_heading = bool(
                overlap and (overlap == len(terms) or overlap >= 2)
                and any(prose for group in blocks[index:section_ends[index]] for _, _, prose in group)
            )
            body_exact = focal is not None and phrase in blocks[index][focal][1].casefold()
            scores.append((strong_heading, phrase in headings[index].casefold(), body_overlap, body_exact, overlap))
        index = max(range(len(scores)), key=scores.__getitem__, default=None)
        if index is None or not (scores[index][0] or scores[index][2] >= max(1, (2 * len(terms) + 2) // 3)):
            missing.append(topic)
            continue
        heading_match = scores[index][0]
        match_kind = "heading" if heading_match else "text"
        if heading_match:
            excerpt = "\n\n".join(
                _format_section(headings[i], blocks[i]) for i in range(index, section_ends[index])
            )
        else:
            focal = focal_blocks[index]
            kind, text, prose = blocks[index][focal]
            source_blocks = sections[index].get("blocks")
            if kind == "list" and len(text) < 240 and source_blocks and source_blocks[focal].get("linked"):
                match_kind = "reference"
            match = re.search(re.escape(phrase), text, re.I)
            if not match:
                match = next((word for word in re.finditer(r"[^\W_]{3,}", text) if _terms(word.group()) & terms), None)
            start = max(0, match.start() - 100) if match else 0
            surrounding = [(kind, text[start:], prose), *blocks[index][focal + 1:focal + 2]]
            excerpt = _format_section(headings[index], surrounding)
        excerpt = sanitize_page_text(excerpt)
        if excerpt not in excerpts:
            excerpts.append(excerpt)
        selected.append(({
            "topic": topic,
            "heading": sanitize_page_text(headings[index])[:160] or None,
            "match": match_kind,
        }, excerpts.index(excerpt)))
    budget = max(0, max_chars - 2 * (len(excerpts) - 1))
    pieces = [""] * len(excerpts)
    for remaining, index in enumerate(sorted(range(len(excerpts)), key=lambda i: len(excerpts[i])), 1):
        pieces[index] = excerpts[index][:budget // (len(excerpts) - remaining + 1)].rstrip()
        budget -= len(pieces[index])
    matches = []
    for match, index in selected:
        if not pieces[index]:
            missing.append(match["topic"])
            continue
        if len(pieces[index]) < len(excerpts[index]):
            match["truncated"] = True
        matches.append(match)
    return "\n\n".join(filter(None, pieces)), matches, missing, pieces != excerpts


def _format_section(heading: str, blocks: list[tuple[str, str, bool]]) -> str:
    lines = [heading] if heading else []
    lines.extend(f"- {text}" if kind == "list" else text for kind, text, _ in blocks if text)
    return "\n".join(lines)


def _section_block(item: dict) -> tuple[str, str, bool]:
    text = " ".join(item["parts"])
    pointer = item.get("linked") and len(text) < 180 and re.search(
        r"^(?:for\b[^.]{0,80},\s*)?(?:see|consult|refer to|visit|read more|learn more)\b", text, re.I,
    )
    return item["kind"], text, bool(item.get("prose", True) and not pointer)


def research_link_score(link: dict[str, str], topics: list[str]) -> tuple[int, int]:
    terms = _terms(" ".join(topics))
    path = unquote(urlsplit(link["url"]).path)
    path_words = _terms(re.sub(r"[_/-]", " ", path))
    path_words.update(re.sub(r"[-_]", "", part.casefold()) for part in path.split("/"))
    return len(terms & (_terms(link["text"]) | path_words)), len(terms & _terms(link.get("context", "")))


def research_links(
    raw_url: str, candidates: list[list[str]], topics: list[str], *, related_topics: list[str] | None = None,
) -> list[dict[str, str]]:
    """Expose bounded same-origin links without credentials, queries, or fragments."""
    page = urlsplit(raw_url)
    origin = (page.scheme, page.netloc)
    terms = _terms(" ".join(topics))
    ranked: dict[str, tuple[tuple[int, int, int, int], dict[str, str]]] = {}
    for href, text, *context in candidates:
        try:
            validate_browser_url(href)
            parsed = urlsplit(href)
        except (BlockedBrowserDestination, InvalidBrowserURL):
            continue
        if (parsed.scheme, parsed.netloc) != origin or parsed.query or parsed.path == page.path:
            continue
        url = safe_browser_url(href)
        label = sanitize_page_text(safe_browser_title(raw_url, safe_browser_title(href, text)) or "")
        if url == "[REDACTED]" or not label:
            continue
        nearby = safe_browser_title(raw_url, safe_browser_title(href, context[0])) if context and context[0] else None
        nearby = sanitize_page_text(nearby)[:240] if nearby else ""
        link = {"url": url, "text": label}
        if nearby and nearby != label:
            link["context"] = nearby
        direct, contextual = research_link_score(link, topics)
        related_direct, related_context = research_link_score(link, related_topics or [])
        score = (direct, contextual, related_context, related_direct)
        if any(score) or not terms:
            ranked.setdefault(url, (score, link))
    return [link for _, link in sorted(ranked.values(), key=lambda item: item[0], reverse=True)[:12]]


def search_result_sources(
    engine: str, search_url: str, candidates: list[list[str]],
) -> list[dict[str, str]]:
    """Accept only observed public result destinations from a verified search page."""
    origin = urlsplit(search_url).netloc
    sources: dict[str, dict[str, str]] = {}
    for href, label, *_ in candidates:
        try:
            validate_browser_url(href)
            parsed = urlsplit(href)
            if engine == "duckduckgo" and parsed.netloc in {origin, "duckduckgo.com"} and parsed.path == "/l/":
                href = parse_qs(parsed.query).get("uddg", [""])[0]
            elif engine == "google" and parsed.netloc == origin and parsed.path == "/url":
                params = parse_qs(parsed.query)
                href = (params.get("q") or params.get("url") or [""])[0]
            validate_browser_url(href)
            target = urlsplit(href)
        except (BlockedBrowserDestination, InvalidBrowserURL):
            continue
        if target.netloc == origin or target.query:
            continue
        label = sanitize_page_text(safe_browser_title(search_url, label) or "")[:160]
        url = safe_browser_url(href)
        sources.setdefault(url, {"url": url, "text": label})
    return list(sources.values())[:12]
