import json

import pytest

from app.youtube_evidence import ingest_transcript, render_evidence_answer


VIDEO = "AaBbCcDdE-1"
VTT = "WEBVTT\n\n00:01.250 --> 00:03.000\nFirst cue.\n\n00:04.000 --> 00:05.000\nSecond cue."
SRT = "1\n00:00:01,250 --> 00:00:03,000\nFirst cue.\n\n2\n00:00:04,000 --> 00:00:05,000\nSecond cue."


@pytest.mark.parametrize("data", [VTT, SRT])
def test_timed_transcript_preserves_order_language_and_timestamps(data):
    evidence = ingest_transcript(VIDEO, data, language="en-GB")
    assert evidence.retrieval_status == "available"
    assert evidence.retrieval_method == "user_supplied" and evidence.language == "en-GB"
    assert evidence.video_id == VIDEO
    assert [(s.text, s.start_ms, s.end_ms) for s in evidence.segments] == [
        ("First cue.", 1250, 3000), ("Second cue.", 4000, 5000),
    ]


@pytest.mark.parametrize("value, format, status", [
    ("Plain transcript", "txt", "available"), ("  ", None, "unavailable"),
    ("Caption", "xml", "unsupported"), ("WEBVTT\n\n00:10.000 --> 00:05.000\nBackwards", "vtt", "retrieval_failed"),
    ("1\n99:78:99,000 --> 99:79:99,000\nBad", "srt", "retrieval_failed"),
])
def test_ingestion_statuses(value, format, status):
    evidence = ingest_transcript(VIDEO, value, format=format)
    assert evidence.retrieval_status == status
    if format == "txt":
        assert evidence.segments[0].start_ms is None
        assert evidence.segments[0].end_ms is None


def test_truncated_cue_cannot_be_cited_at_its_original_time():
    evidence = ingest_transcript(VIDEO, VTT, max_chars=4)
    assert evidence.truncated and evidence.segments[0].text == "Firs"
    assert evidence.segments[0].start_ms is None
    raw = json.dumps({"answer": "Short cue", "citations": [{"evidence_id": evidence.evidence_id, "start_ms": 1250}]})
    with pytest.raises(ValueError, match="Timestamp"):
        render_evidence_answer(raw, evidence, [])


def test_mixed_citations_use_verified_sources_and_time():
    evidence = ingest_transcript(VIDEO, VTT)
    web = [{"evidence_id": "web:1", "url": "https://example.com/guide"}]
    output = render_evidence_answer(json.dumps({
        "answer": "The supplied transcript and the web guide cover the same topic.",
        "citations": [{"evidence_id": evidence.evidence_id, "start_ms": 1250, "end_ms": 3000},
                      {"evidence_id": "web:1"}],
    }), evidence, web)
    assert f"watch?v={VIDEO}&t=1s" in output
    assert "1.25s" in output and "https://example.com/guide" in output
    assert "user-supplied transcript" in output


@pytest.mark.parametrize("citation", [
    {"evidence_id": "unknown"},
    {"evidence_id": f"youtube:{VIDEO}", "start_ms": 1251},
    {"evidence_id": f"youtube:{VIDEO}", "start_ms": 1250, "end_ms": 9999},
    {"evidence_id": f"youtube:{VIDEO}", "end_ms": 3000},
    {"evidence_id": "web:1", "start_ms": 1250},
])
def test_rejects_unknown_sources_and_uncollected_timestamps(citation):
    evidence = ingest_transcript(VIDEO, VTT)
    with pytest.raises(ValueError):
        render_evidence_answer(json.dumps({"answer": "Summary", "citations": [citation]}), evidence,
                               [{"evidence_id": "web:1", "url": "https://example.com/"}])


@pytest.mark.parametrize("answer", [
    "Invented [link](https://foreign.test/)", "At 12:35 the topic changes.", "Unsafe \x1b[2J terminal escape",
])
def test_rejects_unvalidated_references_in_free_text(answer):
    evidence = ingest_transcript(VIDEO, VTT)
    with pytest.raises(ValueError):
        render_evidence_answer(json.dumps({
            "answer": answer, "citations": [{"evidence_id": evidence.evidence_id}],
        }), evidence, [])


@pytest.mark.parametrize("references, error", [
    ([{"evidence_id": f"youtube:{VIDEO}", "start_ms": 8000}], "Timestamp"),
    ([{"evidence_id": "unknown"}], "Unknown"),
    ([{"evidence_id": f"youtube:{VIDEO}", "start_ms": 1250}], "Each usable"),
    ([{"evidence_id": f"youtube:{VIDEO}", "start_ms": 1250},
      {"evidence_id": "youtube:BbCcDdEeF-2"}], "needs a timestamped citation"),
])
def test_multi_video_citations_are_isolated_and_require_all_usable_sources(references, error):
    first = ingest_transcript(VIDEO, VTT)
    second = ingest_transcript("BbCcDdEeF-2", "WEBVTT\n\n00:08.000 --> 00:09.000\nOther video")
    with pytest.raises(ValueError, match=error):
        render_evidence_answer(json.dumps({"answer": "Comparison", "citations": references}), (first, second), [])


def test_citation_label_preserves_milliseconds_for_long_videos():
    evidence = ingest_transcript(VIDEO, "WEBVTT\n\n01:00:01.257 --> 01:00:02.000\nLong recording cue.")
    answer = {"answer": "The cue appears late.", "citations": [
        {"evidence_id": f"youtube:{VIDEO}", "start_ms": 3601257},
    ]}
    result = render_evidence_answer(json.dumps(answer), evidence, [])
    assert "3601.257s" in result and "&t=3601s" in result
