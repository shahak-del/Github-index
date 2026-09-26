"""Structured logging setup.

A logging filter redacts anything that looks like one of our known secret
values, as a defense-in-depth measure on top of never passing secrets to
log calls in the first place.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

_REDACTED = "***REDACTED***"


class SecretRedactionFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._patterns = [re.compile(re.escape(s)) for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pattern in self._patterns:
            if pattern.search(msg):
                msg = pattern.sub(_REDACTED, msg)
        record.msg = msg
        record.args = ()
        return True


def setup_logging(log_dir: Path, secrets: list[str], level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("knowledge_brain")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "knowledge_brain.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    redaction = SecretRedactionFilter(secrets)
    for handler in logger.handlers:
        handler.addFilter(redaction)

    return logger
