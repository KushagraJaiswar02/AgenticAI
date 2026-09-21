import logging

import pytest

from app.errors import ApplicationError
from app.input import normalize_input
from app.llm import LLMProvider, LLMResponse
from app.logging_config import SecretRedactingFilter
from app.orchestrator import Orchestrator


class FakeLLM(LLMProvider):
    def generate(self, prompt: str, tools=None, *, tool_call=None, tool_result=None) -> LLMResponse:
        return LLMResponse(text=f"received {prompt}")


def test_input_normalization() -> None:
    assert normalize_input("  hello   from  JARVIS  ") == "hello from JARVIS"


def test_orchestrator_processes_message() -> None:
    assert Orchestrator(FakeLLM()).process("  hello  ") == "received hello"


def test_orchestrator_rejects_empty_message() -> None:
    with pytest.raises(ApplicationError, match="must not be empty"):
        Orchestrator(FakeLLM()).process("  ")


def test_logging_filter_redacts_secret(caplog) -> None:
    logger = logging.getLogger("secret-test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactingFilter())
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO, logger="secret-test"):
            logger.info("request failed for OPENAI_API_KEY=test-secret")
        assert "test-secret" not in caplog.text
        assert "OPENAI_API_KEY" not in caplog.text
    finally:
        logger.removeHandler(handler)
