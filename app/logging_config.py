"""Secret-safe application logging."""

from __future__ import annotations

import logging
import re


class SecretRedactingFilter(logging.Filter):
    """Redact API-key-like values from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        message = re.sub(r"(?i)OPENAI_API_KEY\s*=\s*\S+", "[REDACTED]", message)
        message = re.sub(r"(?i)GEMINI_API_KEY\s*=\s*\S+", "[REDACTED]", message)
        message = re.sub(r"(?i)(api_key|authorization)\s*[:=]\s*\S+", "[REDACTED]", message)
        message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
        record.msg = message
        record.args = ()
        return True


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the JARVIS logger once and return it."""
    logger = logging.getLogger("jarvis")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(SecretRedactingFilter())
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
