"""Read-only YouTube tools and the local presentation of their API results."""

from pathlib import Path
import re
from dataclasses import dataclass
from unicodedata import category

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools import Tool, ToolExecutionResult
from app.youtube_api import MAX_RESULTS, YouTubeAPI, YouTubeError
from app.youtube_evidence import TranscriptIssue, YouTubeEvidence
from app.youtube_identity import video_id, video_url


class YouTubeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=10, strict=True, ge=1, le=MAX_RESULTS)


class YouTubeInspectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    video: str = Field(min_length=1, max_length=2048)
    transcript_language: str | None = Field(default=None, pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$", max_length=40)

    @field_validator("video")
    @classmethod
    def validate_video(cls, value: str) -> str:
        return video_id(value)


def _api_failure(error: YouTubeError) -> ToolExecutionResult:
    return ToolExecutionResult(False, {"status": error.code, "error": str(error)}, error=str(error))


class YouTubeSearchTool(Tool):
    name = "youtube_search"
    description = "Search official YouTube video results by query."
    input_model = YouTubeSearchInput

    def __init__(self, api: YouTubeAPI):
        self.api = api

    def execute(self, arguments: YouTubeSearchInput) -> ToolExecutionResult:
        try:
            return ToolExecutionResult(True, {"status": "available", **self.api.search(arguments.query, arguments.limit)})
        except YouTubeError as error:
            return _api_failure(error)


class YouTubeInspectTool(Tool):
    name = "youtube_inspect"
    description = "Inspect official YouTube metadata for one video ID or supported URL."
    input_model = YouTubeInspectInput

    def __init__(self, api: YouTubeAPI):
        self.api = api

    def execute(self, arguments: YouTubeInspectInput) -> ToolExecutionResult:
        identifier = arguments.video
        try:
            metadata = self.api.metadata([identifier]).get(identifier)
            if metadata is None:
                raise YouTubeError("inaccessible_video", "This video is unavailable to the YouTube API caller")
        except YouTubeError as error:
            failure = _api_failure(error)
            status = ("restricted" if error.code == "permission_denied" else
                      "unsupported" if error.code in {"missing_api_key", "inaccessible_video"} else "retrieval_failed")
            failure.data["transcript"] = {"status": status, "source": None, "language": arguments.transcript_language}
            return failure
        evidence = YouTubeEvidence(
            evidence_id=f"youtube-metadata:{identifier}", source_kind="youtube_metadata", video_id=identifier,
            canonical_url=video_url(identifier), retrieval_method="youtube_data_api", retrieval_status="available",
            metadata=metadata,
        )
        return ToolExecutionResult(True, {
            "status": "available", "evidence": evidence.model_dump(),
            "transcript": {"status": "unavailable" if metadata["captions_available"] is False else "unsupported",
                           "source": None, "language": arguments.transcript_language},
        })


def create_youtube_tools(api_key: str | None) -> list[Tool]:
    api = YouTubeAPI(api_key)
    return [YouTubeSearchTool(api), YouTubeInspectTool(api)]


def _display(value: object, limit: int = 400) -> str:
    return " ".join("".join(c for c in str(value)[:limit] if category(c)[0] != "C" or c in "\n\t").split())


def render_youtube_result(result: ToolExecutionResult, *, show_transcript: bool = True) -> str:
    if not result.success:
        return result.error or "YouTube request failed."
    data = result.data or {}
    if "videos" in data:
        videos = data["videos"]
        if not videos:
            return "YouTube returned no matching videos."
        lines = ["YouTube search results"]
        for item in videos:
            title = _display(item["title"] or "Title unavailable")
            channel = _display(item["channel_name"] or "Channel unavailable")
            date = f", {_display(item['published_at'], 40)}" if item["published_at"] else ""
            lines.append(f"{item['position']}. {title} ({channel}{date}) — {item['canonical_url']}")
            if item["description"]:
                lines.append("   " + _display(item["description"]))
        if data["has_more"]:
            lines.append("YouTube indicated that more results were available.")
        return "\n".join(lines)
    evidence = data["evidence"]
    metadata = evidence["metadata"]
    lines = ["YouTube video metadata", evidence["canonical_url"]]
    for key, label in (
        ("title", "Title"), ("channel_name", "Channel"), ("channel_id", "Channel ID"),
        ("published_at", "Published"), ("duration", "Duration"), ("captions_available", "Captions indicated"),
        ("view_count", "Views"), ("like_count", "Likes"), ("comment_count", "Comments"),
        ("privacy_status", "Privacy"), ("upload_status", "Upload status"), ("description", "Description"),
    ):
        value = metadata.get(key)
        if value is not None:
            lines.append(f"{label}: {_display(value)}")
    if show_transcript:
        lines.append(f"Transcript: {data['transcript']['status']}. You can supply transcript text or a file for analysis.")
    return "\n".join(lines)


_SEARCH = re.compile(r"^(?:(?:can|could|would) you )?(?:please )?(?:search (?:on )?youtube(?: for)?|youtube search)\s+(.+)$", re.I)
_FIND = re.compile(r"^(?:find|search for)\s+(.+?)\s+on youtube[.!?]?$", re.I)
_INSPECT = re.compile(r"^(?:(?:can|could|would) you )?(?:please )?(?:inspect|summari[sz]e|research|compare|explain|tell)\b", re.I)
_REFERENCE = re.compile(r"(?:https?://[^\s<>\[\]()]+|\b(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/[^\s<>\[\]()]+)", re.I)
_BODY = re.compile(r"(?:^|\s)transcript(?P<file> file)?[ \t]*:[ \t]*", re.I)
_LANGUAGE = re.compile(r"(?:^|\s)language[ \t]*:[ \t]*([A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*)\s*$", re.I)
_TRANSCRIPTS = re.compile(r"(?:^|\s)transcripts[ \t]*:", re.I)
_MULTI_INTENT = re.compile(r"^(?:please\s+)?(?:which|what|how|identify|contrast|analy[sz]e)\b", re.I)
_BINDING = re.compile(r'\s*(?P<video>\S+?)(?:\s+\((?P<language>[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*)\))?\s*=\s*"(?P<file>[^"\r\n]+)"\s*')
_METADATA = re.compile(r"\binclude youtube metadata\b[.!]?", re.I)
MAX_TRANSCRIPTS = 5


@dataclass(frozen=True)
class TranscriptInput:
    number: int
    video_id: str
    language: str | None = None
    file: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class YouTubeRequest:
    prompt: str
    query: str | None = None
    videos: tuple[str, ...] = ()
    transcripts: tuple[TranscriptInput, ...] = ()
    transcript_issues: tuple[TranscriptIssue, ...] = ()
    explicit_bindings: bool = False
    include_metadata: bool = False
    language: str | None = None
    web_prompt: str | None = None


def parse_youtube_request(raw: str) -> YouTubeRequest | None:
    marker = _BODY.search(raw)
    multiple = _TRANSCRIPTS.search(raw)
    if multiple and (not marker or multiple.start() < marker.start()):
        head = raw[:multiple.start()].strip()
        if _INSPECT.match(head) or _MULTI_INTENT.match(head):
            return _parse_transcript_bindings(head, raw[multiple.end():])
    head = raw[:marker.start()] if marker else raw
    language = _LANGUAGE.search(head)
    if language:
        head = head[:language.start()]
    head = " ".join(head.split())
    search = _SEARCH.fullmatch(head) or _FIND.fullmatch(head)
    if search and not marker:
        return YouTubeRequest(head, query=search[1])
    if not _INSPECT.match(head):
        return None
    explicit = re.match(
        r"^(?:(?:can|could|would) you )?(?:please )?(?:inspect|summari[sz]e|research|compare|explain) youtube\b\s*(\S*)",
        head, re.I,
    )
    references = [item.rstrip(".,!?;") for item in _REFERENCE.findall(head)]
    if explicit and explicit[1] and explicit[1] not in references:
        references.insert(0, explicit[1])
    identifiers, web = [], []
    for reference in references:
        try:
            identifiers.append(video_id(reference))
        except ValueError:
            if re.search(r"youtube\.com|youtu\.be", reference, re.I) or (explicit and reference == explicit[1]):
                return YouTubeRequest("")
            web.append(reference)
    if not identifiers:
        return YouTubeRequest("") if explicit or marker else None
    for reference in references:
        if reference not in web:
            head = head.replace(reference, "[YouTube video]")
    body = raw[marker.end():].strip() if marker else None
    if len(set(identifiers)) > 5 or (marker and len(set(identifiers)) != 1):
        return YouTubeRequest("")
    is_file = bool(marker and marker["file"])
    return YouTubeRequest(
        head, videos=tuple(dict.fromkeys(identifiers)),
        transcripts=(TranscriptInput(1, identifiers[0], language[1] if language else None,
                                     file=body.strip('"') if is_file else None,
                                     text=None if is_file else body),) if marker else (),
        language=language[1] if language else None,
        web_prompt=head if web else None,
    )


def _parse_transcript_bindings(head: str, bindings: str) -> YouTubeRequest:
    if len(bindings) > 16_384:
        return YouTubeRequest("", transcript_issues=(TranscriptIssue(number=0, code="bindings_too_large"),))
    clauses = []
    offset = 0
    quoted = False
    for index, char in enumerate(bindings):
        if char == '"':
            quoted = not quoted
        elif char == "|" and not quoted:
            clauses.append(bindings[offset:index])
            offset = index + 1
    clauses.append(bindings[offset:])
    if len(clauses) > MAX_TRANSCRIPTS:
        return YouTubeRequest("", transcript_issues=(TranscriptIssue(number=MAX_TRANSCRIPTS + 1, code="too_many_transcripts"),))

    inputs, issues = [], []
    for number, clause in enumerate(clauses, 1):
        match = _BINDING.fullmatch(clause)
        if not match:
            issues.append(TranscriptIssue(number=number, code="invalid_binding"))
            continue
        try:
            identifier = video_id(match["video"])
        except ValueError:
            issues.append(TranscriptIssue(number=number, code="invalid_video_id"))
        else:
            inputs.append(TranscriptInput(number, identifier, match["language"], file=match["file"]))

    duplicate_ids = {item.video_id for item in inputs if sum(other.video_id == item.video_id for other in inputs) > 1}
    references = {}
    web = False
    for value in _REFERENCE.findall(head):
        reference = value.rstrip(".,!?;")
        try:
            references[reference] = video_id(reference)
        except ValueError:
            if re.search(r"youtube\.com|youtu\.be", reference, re.I):
                return YouTubeRequest("", transcript_issues=(TranscriptIssue(number=0, code="invalid_video_reference"),))
            web = True
    mentioned = set(references.values())
    valid = []
    for item in inputs:
        code = ("duplicate_video_id" if item.video_id in duplicate_ids else
                "unexpected_video_id" if mentioned and item.video_id not in mentioned else None)
        if code:
            issues.append(TranscriptIssue(number=item.number, video_id=item.video_id, code=code))
        else:
            valid.append(item)
    for identifier in dict.fromkeys(references.values()):
        if not any(item.video_id == identifier for item in inputs):
            issues.append(TranscriptIssue(number=0, video_id=identifier, code="missing_transcript"))
    for reference, identifier in references.items():
        number = next((item.number for item in valid if item.video_id == identifier), None)
        head = head.replace(reference, f"[Video {number}]" if number else "[Video without usable binding]")
    include_metadata = bool(_METADATA.search(head))
    prompt = " ".join(_METADATA.sub("", head).split()) or "Compare the supplied video transcripts."
    return YouTubeRequest(
        prompt, videos=tuple(dict.fromkeys(item.video_id for item in valid)), transcripts=tuple(valid),
        transcript_issues=tuple(sorted(issues, key=lambda item: item.number)), include_metadata=include_metadata,
        explicit_bindings=True,
        web_prompt=prompt if web else None,
    )


def transcript_format(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")
