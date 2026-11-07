"""Identify a YouTube video without resolving or fetching a supplied URL."""

import re
from urllib.parse import parse_qsl, urlsplit


VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
VIDEO_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"})
YOUTUBE_TOOLS = frozenset({"youtube_search", "youtube_inspect"})


def video_id(value: str) -> str:
    if VIDEO_ID.fullmatch(value):
        return value
    if (not 0 < len(value) <= 2048 or re.search(r"[\s\x00-\x1f\x7f\\]", value)
            or re.search(r"%(?![0-9a-fA-F]{2})", value)):
        raise ValueError("Invalid YouTube video ID or URL")
    try:
        url = urlsplit(value if "://" in value else "https://" + value)
        default_port = {"http": 80, "https": 443}.get(url.scheme)
        if (default_port is None or url.hostname not in VIDEO_HOSTS or "@" in url.netloc
                or url.netloc.endswith(":") or url.port not in (None, default_port)):
            raise ValueError
        ids = [item for key, item in parse_qsl(url.query, keep_blank_values=True, max_num_fields=40) if key == "v"]
        parts = url.path.split("/")
        if url.hostname == "youtu.be":
            candidate = parts[1] if len(parts) == 2 and not ids else ""
        elif url.path == "/watch":
            candidate = ids[0] if len(ids) == 1 else ""
        else:
            candidate = parts[2] if len(parts) == 3 and parts[1] in {"shorts", "embed", "live"} and not ids else ""
        if VIDEO_ID.fullmatch(candidate):
            return candidate
    except ValueError:
        pass
    raise ValueError("Invalid YouTube video ID or URL")


def video_identity(value: str) -> tuple[str, str]:
    return "youtube", video_id(value)


def video_url(value: str) -> str:
    return "https://www.youtube.com/watch?v=" + video_id(value)
