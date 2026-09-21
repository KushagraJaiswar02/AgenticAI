"""JARVIS V0.1 terminal entry point."""

from __future__ import annotations

import logging

from app.config import ConfigurationError, load_settings
from app.database import Database
from app.errors import ApplicationError
from app.input import read_input
from app.logging_config import configure_logging
from app.orchestrator import Orchestrator
from app.provider_factory import create_llm_provider
from app.response import render_response
from app.tools import ToolRegistry, WeatherTool


def main() -> int:
    logger = configure_logging()
    try:
        settings = load_settings()
        Database(settings.database_url)
        llm_provider = create_llm_provider(settings)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except ApplicationError as exc:
        logger.error("Application startup failed: %s", exc)
        return 1

    tool_registry = ToolRegistry([WeatherTool()])
    orchestrator = Orchestrator(create_llm_provider(settings), tool_registry=tool_registry, logger=logger)

    print("JARVIS V0.1. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            message = read_input()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.lower() in {"exit", "quit"}:
            return 0
        try:
            print(render_response(orchestrator.process(message)))
        except ApplicationError as exc:
            logger.exception("Request failed: %s", exc)
            print("JARVIS: I could not process that request.")


if __name__ == "__main__":
    raise SystemExit(main())
