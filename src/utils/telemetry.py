from __future__ import annotations

import logging
import os
import threading
from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    try:
        from opentelemetry.sdk.trace.export import OTLPSpanExporter  # type: ignore
    except ImportError:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    trace = None  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore

logger = logging.getLogger(__name__)

_initialized = False
_tracer: Optional["trace.Tracer"] = None
_lock = threading.Lock()


def telemetry_enabled() -> bool:
    flag = os.getenv("ENABLE_OTEL", os.getenv("ENABLE_TELEMETRY", "0"))
    return str(flag).lower() in {"1", "true", "yes", "on"}


def _configure_exporter() -> Optional["OTLPSpanExporter"]:
    if OTLPSpanExporter is None:
        return None
    exporter_kwargs = {}
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter_kwargs["endpoint"] = endpoint
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if headers:
        parsed = {}
        for item in headers.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            parsed[key.strip()] = value.strip()
        if parsed:
            exporter_kwargs["headers"] = parsed
    try:
        return OTLPSpanExporter(**exporter_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to configure OTLP exporter: %s", exc)
        return None


def init_telemetry() -> Optional["trace.Tracer"]:
    global _initialized, _tracer
    if _initialized:
        return _tracer
    if not telemetry_enabled():
        _initialized = True
        return None
    if trace is None or TracerProvider is None or BatchSpanProcessor is None or Resource is None:
        logger.warning("OpenTelemetry packages not available; telemetry disabled.")
        _initialized = True
        return None

    with _lock:
        if _initialized:
            return _tracer
        resource_attrs = {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "s2s-poc"),
        }
        env = os.getenv("OTEL_ENVIRONMENT") or os.getenv("ENV")
        if env:
            resource_attrs["deployment.environment"] = env
        provider = TracerProvider(resource=Resource.create(resource_attrs))
        exporter = _configure_exporter()
        if exporter is None:
            logger.warning("OTLP exporter unavailable; telemetry disabled.")
            _initialized = True
            return None
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(os.getenv("OTEL_SERVICE_NAME", "s2s-poc"))
        _initialized = True
        logger.info("OpenTelemetry tracer initialized", extra={"service": resource_attrs["service.name"]})
        return _tracer


def get_tracer() -> Optional["trace.Tracer"]:
    tracer = init_telemetry()
    return tracer


__all__ = ["init_telemetry", "get_tracer", "telemetry_enabled"]
