"""Structured logging with PII redaction applied at the handler.

Redacting at the logging layer rather than at each call site means a future
``logger.info(user_message)`` added by someone in a hurry still cannot write an
employee's email address or phone number into a log file.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from .services.guardrails import redact


class RedactingJsonFormatter(logging.Formatter):
    def __init__(self, *, redact_pii: bool = True) -> None:
        super().__init__()
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if self.redact_pii:
            message = redact(message)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }
        for field in ("trace_id", "session_id", "route", "latency_ms", "status_code"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, redact_pii: bool = True) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(RedactingJsonFormatter(redact_pii=redact_pii))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # uvicorn's access log duplicates our request log line and is not redacted.
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
