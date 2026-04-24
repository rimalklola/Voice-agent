from __future__ import annotations

import logging
import os

try:  # optional dependency
    from opentelemetry import trace
except ModuleNotFoundError:  # pragma: no cover - optional dependency absent
    trace = None  # type: ignore

from src.utils.telemetry import init_telemetry, telemetry_enabled
from src.utils.pii_redaction import PIIRedactingFormatter


class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - exercised via logging
        record.trace_id = getattr(record, "trace_id", "0" * 32)
        record.span_id = getattr(record, "span_id", "0" * 16)
        record.trace_flags = getattr(record, "trace_flags", 0)
        if trace is None:
            return True
        span = trace.get_current_span()
        if span is None:
            return True
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return True
        record.trace_id = f"{ctx.trace_id:032x}"
        record.span_id = f"{ctx.span_id:016x}"
        record.trace_flags = int(ctx.trace_flags)
        return True


def _add_trace_filter(handler: logging.Handler, flt: logging.Filter) -> None:
    for existing in handler.filters:
        if isinstance(existing, TraceContextFilter):
            return
    handler.addFilter(flt)


def setup_logging() -> None:
    if getattr(setup_logging, "_configured", False):  # pragma: no cover - idempotence guard
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s %(levelname)s [%(name)s] trace_id=%(trace_id)s span_id=%(span_id)s %(message)s"
    
    # Create PII redacting formatter
    formatter = PIIRedactingFormatter(fmt=fmt)
    
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
    )
    
    # Update root logger handlers with PII redacting formatter
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    trace_filter = TraceContextFilter()
    root = logging.getLogger()
    root.addFilter(trace_filter)
    for handler in root.handlers:
        _add_trace_filter(handler, trace_filter)

    # Suppress noisy third-party loggers
    for noisy in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "sentence_transformers",
        "transformers",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    setup_logging._configured = True  # type: ignore[attr-defined]

    if telemetry_enabled():
        init_telemetry()
