from app.llm import ToolCall
from app.local_router import LocalIntentRouter


def test_local_router_extracts_weather_location_variants() -> None:
    router = LocalIntentRouter()

    for prompt in (
        "what's the weather in Ujjain?",
        "tell me Ujjain's weather",
        "how is the weather in Ujjain?",
        "how's Ujjain weather?",
        "is it raining in Ujjain?",
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


def test_local_router_ignores_non_weather_requests() -> None:
    assert LocalIntentRouter().route("hello there") is None
