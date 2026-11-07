"""Bounded read-only access to the official YouTube Data API."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic, sleep

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.research_budget import active_research
from app.youtube_identity import video_id, video_identity, video_url


MAX_RESULTS = 25
MAX_RESPONSE_BYTES = 512 * 1024
MAX_API_CALLS = 6


class YouTubeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _APIModel(BaseModel):
    model_config = ConfigDict(strict=True)


class _Snippet(_APIModel):
    title: str | None = None
    description: str | None = None
    channelTitle: str | None = None
    channelId: str | None = None
    publishedAt: str | None = None


class _SearchID(_APIModel):
    videoId: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$", min_length=11, max_length=11)


class _SearchItem(_APIModel):
    id: _SearchID
    snippet: _Snippet = Field(default_factory=_Snippet)


class _SearchPage(_APIModel):
    items: list[_SearchItem] = Field(max_length=50)
    nextPageToken: str | None = Field(default=None, max_length=1024)


class _Details(_APIModel):
    duration: str | None = None
    caption: str | None = None


class _Statistics(_APIModel):
    viewCount: str | None = None
    likeCount: str | None = None
    commentCount: str | None = None


class _Status(_APIModel):
    privacyStatus: str | None = None
    uploadStatus: str | None = None
    embeddable: bool | None = None


class _VideoItem(_APIModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$", min_length=11, max_length=11)
    snippet: _Snippet = Field(default_factory=_Snippet)
    contentDetails: _Details = Field(default_factory=_Details)
    statistics: _Statistics = Field(default_factory=_Statistics)
    status: _Status = Field(default_factory=_Status)


class _VideoPage(_APIModel):
    items: list[_VideoItem] = Field(max_length=50)


def _snippet(snippet: _Snippet) -> dict:
    return {
        "title": snippet.title, "description": snippet.description, "channel_name": snippet.channelTitle,
        "channel_id": snippet.channelId, "published_at": snippet.publishedAt,
    }


def _count(value: str | None) -> int | None:
    return int(value) if value and len(value) <= 20 and value.isascii() and value.isdigit() else None


@dataclass
class _RequestBudget:
    deadline: float = field(default_factory=lambda: monotonic() + 15)
    calls: int = 0

    def remaining(self) -> float:
        research = active_research.get()
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise YouTubeError("timeout", "YouTube request deadline reached")
        return min(remaining, research.remaining()) if research else remaining

    def reserve(self) -> float:
        remaining = self.remaining()
        if self.calls >= MAX_API_CALLS:
            raise YouTubeError("request_limit", "YouTube request limit reached")
        self.calls += 1
        return remaining


_active_budget: ContextVar[_RequestBudget | None] = ContextVar("youtube_request_budget", default=None)


@contextmanager
def youtube_request_budget():
    token = _active_budget.set(_RequestBudget())
    try:
        yield
    finally:
        _active_budget.reset(token)


class YouTubeAPI:
    def __init__(self, api_key: str | None, *, transport: httpx.BaseTransport | None = None):
        self._api_key = api_key.strip() if api_key else ""
        self._transport = transport

    def _client(self) -> httpx.Client:
        if not self._api_key:
            raise YouTubeError("missing_api_key", "Set YOUTUBE_DATA_API_KEY to enable YouTube metadata")
        return httpx.Client(
            base_url="https://www.googleapis.com/youtube/v3/",
            headers={"x-goog-api-key": self._api_key, "accept-encoding": "identity"},
            transport=self._transport, follow_redirects=False, trust_env=False,
        )

    @staticmethod
    def _error(status: int, payload: object) -> YouTubeError:
        error = payload.get("error") if isinstance(payload, dict) else None
        reasons = set()
        if isinstance(error, dict):
            for field_name in ("errors", "details"):
                entries = error.get(field_name)
                if isinstance(entries, list):
                    reasons.update(item["reason"] for item in entries if isinstance(item, dict)
                                   and isinstance(item.get("reason"), str))
        if reasons & {"keyInvalid", "API_KEY_INVALID"} or status == 401:
            return YouTubeError("invalid_api_key", "YouTube rejected the API key")
        if reasons & {"quotaExceeded", "dailyLimitExceeded", "QUOTA_EXCEEDED"}:
            return YouTubeError("quota_exhausted", "YouTube API quota exhausted")
        if status == 429 or reasons & {"rateLimitExceeded", "userRateLimitExceeded", "RATE_LIMIT_EXCEEDED"}:
            return YouTubeError("rate_limited", "YouTube API rate limit reached")
        if status == 403:
            return YouTubeError("permission_denied", "YouTube API access denied")
        if status == 404:
            return YouTubeError("inaccessible_video", "YouTube resource is unavailable to this API caller")
        if status >= 500:
            return YouTubeError("server_error", "YouTube API is temporarily unavailable")
        return YouTubeError("request_rejected", "YouTube API rejected the request")

    def _get(self, client: httpx.Client, endpoint: str, params: dict, schema, budget: _RequestBudget):
        for attempt in range(2):
            remaining = budget.reserve()
            try:
                with client.stream("GET", endpoint, params=params,
                                   timeout=httpx.Timeout(min(5, remaining), connect=min(3, remaining))) as response:
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise YouTubeError("invalid_response", "YouTube API returned an unsupported encoding")
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        budget.remaining()
                        raw.extend(chunk)
                        if len(raw) > MAX_RESPONSE_BYTES:
                            raise YouTubeError("invalid_response", "YouTube API response exceeded the size limit")
                    budget.remaining()
                    if response.is_success:
                        try:
                            return schema.model_validate_json(bytes(raw))
                        except ValidationError:
                            raise YouTubeError("invalid_response", "YouTube API returned invalid data") from None
                    try:
                        payload = json.loads(raw)
                    except (ValueError, UnicodeError):
                        payload = None
                    failure = self._error(response.status_code, payload)
            except httpx.TimeoutException:
                failure = YouTubeError("timeout", "YouTube API request timed out")
            except httpx.HTTPError:
                failure = YouTubeError("network_error", "Could not connect to the YouTube API")
            if attempt or failure.code not in {"timeout", "network_error", "server_error"} or budget.remaining() <= 0.15:
                raise failure
            sleep(0.1)

    def search(self, query: str, limit: int = 10) -> dict:
        if not 1 <= limit <= MAX_RESULTS or not 0 < len(query.strip()) <= 300:
            raise ValueError("Invalid YouTube search bounds")
        budget = _active_budget.get() or _RequestBudget()
        results, seen, tokens = [], set(), set()
        token = None
        position = 0
        with self._client() as client:
            for _ in range(3):
                params = {"part": "snippet", "type": "video", "q": query, "maxResults": limit - len(results)}
                if token:
                    params["pageToken"] = token
                page = self._get(client, "search", params, _SearchPage, budget)
                for item in page.items:
                    position += 1
                    identifier = item.id.videoId
                    identity = video_identity(identifier)
                    if identity not in seen and len(results) < limit:
                        seen.add(identity)
                        results.append({"video_id": identifier, "canonical_url": video_url(identifier),
                                        "position": position, **_snippet(item.snippet)})
                token = page.nextPageToken
                if not token or token in tokens or len(results) == limit:
                    break
                tokens.add(token)
        return {"videos": results, "has_more": bool(token)}

    def metadata(self, values: list[str]) -> dict[str, dict]:
        if not 1 <= len(values) <= 50:
            raise ValueError("Invalid YouTube metadata batch size")
        identifiers = dict.fromkeys(video_id(value) for value in values)
        with self._client() as client:
            page = self._get(client, "videos", {
                "part": "snippet,contentDetails,statistics,status", "id": ",".join(identifiers),
            }, _VideoPage, _active_budget.get() or _RequestBudget())
        return {
            item.id: {
                "video_id": item.id, "canonical_url": video_url(item.id), **_snippet(item.snippet),
                "duration": item.contentDetails.duration,
                "captions_available": {"true": True, "false": False}.get(item.contentDetails.caption),
                "view_count": _count(item.statistics.viewCount), "like_count": _count(item.statistics.likeCount),
                "comment_count": _count(item.statistics.commentCount), "privacy_status": item.status.privacyStatus,
                "upload_status": item.status.uploadStatus, "embeddable": item.status.embeddable,
            }
            for item in page.items if item.id in identifiers
        }
