import httpx
import pytest

from app.youtube_api import MAX_RESPONSE_BYTES, YouTubeAPI, YouTubeError, youtube_request_budget


FIRST = "AaBbCcDdE-1"
SECOND = "BbCcDdEeF-2"


def make_api(handler, key="example-test-key"):
    return YouTubeAPI(key, transport=httpx.MockTransport(handler))


def test_search_deduplicates_across_pages_without_losing_position():
    calls = []

    def serve(request):
        calls.append(request)
        assert request.url.host == "www.googleapis.com" and request.url.path == "/youtube/v3/search"
        assert request.url.params["part"] == "snippet" and request.url.params["type"] == "video"
        assert request.url.params["q"] == "React hooks"
        assert request.headers["x-goog-api-key"] == "example-test-key"
        assert "example-test-key" not in str(request.url)
        if len(calls) == 1:
            assert request.url.params["maxResults"] == "2"
            return httpx.Response(200, json={"items": [{"id": {"videoId": FIRST}, "snippet": {
                "title": "Hooks", "description": "Snippet", "channelTitle": "Channel",
                "channelId": "channel-1", "publishedAt": "2020-01-01T00:00:00Z",
            }}], "nextPageToken": "next"})
        assert request.url.params["pageToken"] == "next"
        return httpx.Response(200, json={"items": [{"id": {"videoId": FIRST}}, {"id": {"videoId": SECOND}}]})

    result = make_api(serve).search("React hooks", 2)
    assert [(row["video_id"], row["position"]) for row in result["videos"]] == [(FIRST, 1), (SECOND, 3)]
    assert result["videos"][0]["canonical_url"].endswith(FIRST)
    assert result["videos"][0]["channel_id"] == "channel-1"
    assert result["videos"][1]["description"] is None
    assert result["has_more"] is False


def test_empty_search_and_partial_metadata_batch():
    def serve(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"items": []})
        assert request.url.params["id"] == f"{FIRST},{SECOND}"
        assert request.url.params["part"] == "snippet,contentDetails,statistics,status"
        return httpx.Response(200, json={"items": [{
            "id": FIRST, "contentDetails": {"duration": "PT1M10S", "caption": "true"},
            "statistics": {"viewCount": "0", "likeCount": "2"},
            "status": {"privacyStatus": "unlisted"},
        }]})

    api = make_api(serve)
    assert api.search("none") == {"videos": [], "has_more": False}
    records = api.metadata([FIRST, SECOND, FIRST])
    assert list(records) == [FIRST]
    assert records[FIRST]["duration"] == "PT1M10S"
    assert records[FIRST]["captions_available"] is True
    assert records[FIRST]["view_count"] == 0 and records[FIRST]["like_count"] == 2
    assert records[FIRST]["privacy_status"] == "unlisted"
    assert records[FIRST]["comment_count"] is None and records[FIRST]["title"] is None


@pytest.mark.parametrize("status, reason, expected, attempts", [
    (400, "API_KEY_INVALID", "invalid_api_key", 1), (403, "quotaExceeded", "quota_exhausted", 1),
    (403, "forbidden", "permission_denied", 1), (429, "rateLimitExceeded", "rate_limited", 1),
    (503, "backendError", "server_error", 2), (404, "videoNotFound", "inaccessible_video", 1),
    (302, "", "request_rejected", 1),
])
def test_error_classification_and_bounded_retry(status, reason, expected, attempts, monkeypatch):
    monkeypatch.setattr("app.youtube_api.sleep", lambda _: None)
    calls = []

    def serve(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "https://foreign.test/"},
                              json={"error": {"details": [{"reason": reason}], "message": "example-test-key"}})

    with pytest.raises(YouTubeError) as exc:
        make_api(serve).search("test")
    assert exc.value.code == expected
    assert "example-test-key" not in str(exc.value)
    assert len(calls) == attempts


def test_timeout_retries_once_and_missing_key_sends_no_request(monkeypatch):
    monkeypatch.setattr("app.youtube_api.sleep", lambda _: None)
    calls = []

    def serve(request):
        calls.append(request)
        raise httpx.ReadTimeout("sensitive diagnostic", request=request)

    with pytest.raises(YouTubeError) as missing:
        make_api(serve, key=None).search("test")
    assert missing.value.code == "missing_api_key" and not calls
    with pytest.raises(YouTubeError) as timeout:
        make_api(serve).search("test")
    assert timeout.value.code == "timeout" and len(calls) == 2


@pytest.mark.parametrize("raw", [b"not json", b"{}", b'{"items":[{"id":{"videoId":"invalid"}}]}',
                               b" " * (MAX_RESPONSE_BYTES + 1)],
                         ids=["malformed-json", "missing-items", "invalid-video-id", "oversized"])
def test_rejects_invalid_or_oversized_success(raw):
    with pytest.raises(YouTubeError) as exc:
        make_api(lambda _: httpx.Response(200, content=raw)).search("test")
    assert exc.value.code == "invalid_response"


def test_pagination_and_total_attempt_limit(monkeypatch):
    monkeypatch.setattr("app.youtube_api.sleep", lambda _: None)
    calls = []

    def serve(request):
        calls.append(request)
        if len(calls) % 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"items": [], "nextPageToken": str(len(calls))})

    assert make_api(serve).search("test") == {"videos": [], "has_more": True}
    assert len(calls) == 6


def test_response_deadline_is_checked_while_streaming_small_chunks(monkeypatch):
    from app import youtube_api

    now = [0]
    monkeypatch.setattr(youtube_api, "monotonic", lambda: now[0])

    class SlowBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"{"
            now[0] = 16
            yield b"}"
            pytest.fail("An expired response should not continue reading")

    with pytest.raises(YouTubeError) as exc:
        make_api(lambda _: httpx.Response(200, stream=SlowBody())).search("test")
    assert exc.value.code == "timeout"


def test_metadata_calls_share_a_turn_budget_and_reset_between_turns():
    calls = []

    def serve(request):
        calls.append(request)
        return httpx.Response(200, json={"items": []})

    api = make_api(serve)
    with youtube_request_budget():
        for _ in range(6):
            assert api.metadata([FIRST]) == {}
        with pytest.raises(YouTubeError) as exc:
            api.metadata([FIRST])
        assert exc.value.code == "request_limit"
    assert api.metadata([FIRST]) == {}
    assert len(calls) == 7
