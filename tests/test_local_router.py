from app.llm import ToolCall
from app.local_router import IntentMatch, LocalIntentRouter


def test_local_router_extracts_weather_location_variants() -> None:
    router = LocalIntentRouter()

    for prompt in (
        "what's the weather in Ujjain?",
        "tell me Ujjain's weather",
        "how is the weather in Ujjain?",
        "how's Ujjain weather?",
        "what's Ujjain's weather like?",
        "is it raining in Ujjain?",
        "will it rain in Ujjain?",
        "how hot is it in Ujjain?",
        "how cold is it in Ujjain?",
        "what's the temperature in Ujjain?",
        "should I take an umbrella in Ujjain?",
        "what's it like outside in Ujjain?",
    ):
        assert router.route(prompt) == ToolCall(
            name="weather",
            arguments={"location": "Ujjain"},
        )


def test_local_router_does_not_invent_a_location() -> None:
    router = LocalIntentRouter()

    assert router.route("what's the weather?") is None
    assert router.route("how's the weather today?") is None


def test_local_router_extracts_local_tool_intents() -> None:
    router = LocalIntentRouter()
    assert router.route("what time is it?") == ToolCall(name="time", arguments={})
    assert router.route("what's today's date?") == ToolCall(name="date", arguments={})
    assert router.route("calculate (25 + 15) * 3") == ToolCall(
        name="calculator",
        arguments={"expression": "(25 + 15) * 3"},
    )
    assert router.route("what OS am I running?") == ToolCall(
        name="system_info",
        arguments={},
    )
    assert router.route("what is 25 times 17") == ToolCall(
        name="calculator",
        arguments={"expression": "25 * 17"},
    )
    assert router.route("what's 144 divided by 12") == ToolCall(
        name="calculator",
        arguments={"expression": "144 / 12"},
    )
    assert router.route("how much is 25 plus 10") == ToolCall(
        name="calculator",
        arguments={"expression": "25 + 10"},
    )
    assert router.route("what is 20 percent of 500") == ToolCall(
        name="calculator",
        arguments={"expression": "(20 / 100) * 500"},
    )


def test_local_router_ignores_unsupported_requests_without_state() -> None:
    router = LocalIntentRouter()
    assert router.route("hello there") is None
    assert router.route("what time is it?") == ToolCall(name="time", arguments={})
    assert router.route("tell me a joke") is None


def test_local_router_rejects_ambiguous_or_false_positive_requests() -> None:
    router = LocalIntentRouter()
    assert router.route("my laptop is hot") is None
    assert router.route("my CPU temperature is high") is None
    assert router.route("what time is the event?") is None
    assert router.route("what is the date of the event?") is None


def test_local_router_exposes_confidence_before_tool_call_construction() -> None:
    match = LocalIntentRouter().recognize("weather in Indore")

    assert isinstance(match, IntentMatch)
    assert match.intent == "weather"
    assert match.entities == {"location": "Indore"}
    assert match.confidence >= 0.8
    assert LocalIntentRouter().route("what's the weather?") is None
