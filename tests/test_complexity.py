from app.complexity import TaskComplexity, TaskComplexityRouter


def test_simple_tasks_disable_thinking() -> None:
    router = TaskComplexityRouter()
    for prompt in ("what is 2 + 2", "what is the weather in Indore", "open Brave"):
        assert router.classify(prompt) is TaskComplexity.SIMPLE
        assert router.should_think(prompt) is False


def test_complex_tasks_enable_thinking() -> None:
    router = TaskComplexityRouter()
    for prompt in (
        "analyze why my authentication middleware returns 401",
        "compare these designs and plan a migration",
    ):
        assert router.classify(prompt) is TaskComplexity.COMPLEX
        assert router.should_think(prompt) is True


def test_ambiguous_request_defaults_to_simple() -> None:
    assert TaskComplexityRouter().classify("tell me something useful") is TaskComplexity.SIMPLE
