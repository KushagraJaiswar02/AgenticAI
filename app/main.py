"""JARVIS V0.1 terminal entry point."""

from __future__ import annotations

import logging

from app.config import ConfigurationError, load_settings
from app.database import Database
from app.errors import ApplicationError, LLMProviderError, UnexpectedApplicationError
from app.input import read_input
from app.logging_config import configure_logging
from app.orchestrator import Orchestrator
from app.provider_factory import create_llm_provider
from app.response import render_response
from app.tools import (
    CalculatorTool,
    DateTool,
    OpenApplicationTool,
    SystemInfoTool,
    TimeTool,
    ToolRegistry,
    WeatherTool,
)
from app.filesystem_security import FilesystemBoundary
from app.filesystem_tools import create_filesystem_tools
from app.filesystem_context import FilesystemContext
from app.session import ConversationState


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

    boundary = FilesystemBoundary(list(settings.filesystem_allowed_roots))
    filesystem_context = FilesystemContext.create(settings.jarvis_workspace, boundary, settings.project_root)
    tool_registry = ToolRegistry(
        [
            WeatherTool(),
            TimeTool(),
            DateTool(),
            CalculatorTool(),
            SystemInfoTool(),
            OpenApplicationTool(),
            *create_filesystem_tools(boundary, filesystem_context),
        ]
    )
    state = ConversationState(filesystem_context=filesystem_context)
    orchestrator = Orchestrator(llm_provider, tool_registry=tool_registry, logger=logger, confirmation_callback=lambda prompt: input(f"JARVIS: {prompt}\n> "), state=state)

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
        except LLMProviderError:
            logger.warning("LLM provider unavailable and no local capability matched")
            logger.debug("LLM provider failure details", exc_info=True)
            print("JARVIS: I could not process that request.")
        except UnexpectedApplicationError as exc:
            logger.exception("Request failed: %s", exc)
            print("JARVIS: I could not process that request.")
        except ApplicationError as exc:
            logger.error("Request failed: %s", exc)
            print("JARVIS: I could not process that request.")


if __name__ == "__main__":
    raise SystemExit(main())
