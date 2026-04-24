from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from src.utils.call_ledger import upsert_call_record

PILOT_CALL_FIELDS = [
    "call_id",
    "call_started_utc",
    "call_finished_utc",
    "duration_ms",
    "session_status",
    "failure_reason",
    "retry_attempts",
    "call_sid",
    "stream_sid",
    "voice",
    "model",
    "temperature",
    "user_turns",
    "assistant_responses",
    "rate_limit_events",
    "realtime_error_events",
    "twilio_idle_timeouts",
    "twilio_disconnects",
    "openai_disconnects",
    "tool_calls_total",
    "tool_ask_to_reserve",
    "tool_get_project_info",
    "tool_get_property_specs",
    "tool_get_project_facts",
    "tool_end_call",
    "caller_wants_reservation",
    "reservation_status",
    "reservation_request_type",
    "turn_count",
    "customer_turn_count",
    "assistant_turn_count",
    "transcript_path",
    "transcript_preview",
    "reservation_ok",
    "reservation_duplicate",
    "reservation_error",
]

_TOOL_COUNTER_KEYS = {
    "ask_to_reserve": "tool_ask_to_reserve",
    "get_project_info": "tool_get_project_info",
    "get_property_specs": "tool_get_property_specs",
    "get_project_facts": "tool_get_project_facts",
    "end_call": "tool_end_call",
}

_NUMERIC_FIELDS = {
    "duration_ms",
    "retry_attempts",
    "user_turns",
    "assistant_responses",
    "rate_limit_events",
    "realtime_error_events",
    "twilio_idle_timeouts",
    "twilio_disconnects",
    "openai_disconnects",
    "tool_calls_total",
    "tool_ask_to_reserve",
    "tool_get_project_info",
    "tool_get_property_specs",
    "tool_get_project_facts",
    "tool_end_call",
    "caller_wants_reservation",
    "turn_count",
    "customer_turn_count",
    "assistant_turn_count",
    "reservation_ok",
    "reservation_duplicate",
    "reservation_error",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_pilot_call_row(*, voice: str, model: str, temperature: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "call_id": "",
        "call_started_utc": _utc_now_iso(),
        "call_finished_utc": "",
        "duration_ms": 0,
        "session_status": "unknown",
        "failure_reason": "",
        "retry_attempts": 0,
        "call_sid": "",
        "stream_sid": "",
        "voice": voice,
        "model": model,
        "temperature": temperature,
        "reservation_status": "not_requested",
        "reservation_request_type": "",
        "transcript_path": "",
        "transcript_preview": "",
    }
    for field in _NUMERIC_FIELDS:
        row[field] = 0
    return row


def set_call_identifiers(row: Dict[str, Any], *, call_sid: str | None, stream_sid: str | None) -> None:
    if call_sid:
        row["call_sid"] = str(call_sid)
    if stream_sid:
        row["stream_sid"] = str(stream_sid)


def increment_counter(row: Dict[str, Any], key: str, amount: int = 1) -> None:
    current = int(row.get(key, 0) or 0)
    row[key] = max(0, current + amount)


def record_tool_call(row: Dict[str, Any], tool_name: str | None) -> None:
    increment_counter(row, "tool_calls_total")
    if not tool_name:
        return
    mapped = _TOOL_COUNTER_KEYS.get(tool_name)
    if mapped:
        increment_counter(row, mapped)


def record_reservation_status(row: Dict[str, Any], status: str | None) -> None:
    normalized = str(status or "").strip().lower()
    row["reservation_status"] = normalized or "not_requested"
    if normalized in {"ok", "duplicate", "needs_confirmation", "missing_fields", "invalid_fields", "rate_limited", "error"}:
        row["caller_wants_reservation"] = 1
    if normalized == "ok":
        increment_counter(row, "reservation_ok")
        return
    if normalized == "duplicate":
        increment_counter(row, "reservation_duplicate")
        return
    if normalized:
        increment_counter(row, "reservation_error")


def finalize_pilot_call_row(
    row: Dict[str, Any],
    *,
    session_status: str,
    failure_reason: str,
    retry_attempts: int,
    duration_ms: int,
) -> Dict[str, Any]:
    row["session_status"] = (session_status or "unknown").strip() or "unknown"
    row["failure_reason"] = (failure_reason or "").strip()
    row["retry_attempts"] = max(0, int(retry_attempts))
    row["duration_ms"] = max(0, int(duration_ms))
    row["call_finished_utc"] = _utc_now_iso()
    row["turn_count"] = max(0, int(row.get("customer_turn_count", 0) or 0) + int(row.get("assistant_turn_count", 0) or 0))
    return row


def _coerce_row_for_storage(row: Dict[str, Any]) -> Dict[str, Any]:
    safe_row: Dict[str, Any] = {}
    for field in PILOT_CALL_FIELDS:
        value = row.get(field, "")
        if field in _NUMERIC_FIELDS:
            try:
                safe_row[field] = int(value)
            except (TypeError, ValueError):
                safe_row[field] = 0
        else:
            safe_row[field] = "" if value is None else str(value)
    return safe_row


def append_pilot_call_row(sqlite_path: str, row: Dict[str, Any]) -> None:
    serialized = _coerce_row_for_storage(row)
    upsert_call_record(sqlite_path, serialized)


async def append_pilot_call_row_async(sqlite_path: str, row: Dict[str, Any]) -> None:
    await asyncio.to_thread(append_pilot_call_row, sqlite_path, row)


__all__ = [
    "PILOT_CALL_FIELDS",
    "append_pilot_call_row",
    "append_pilot_call_row_async",
    "finalize_pilot_call_row",
    "increment_counter",
    "new_pilot_call_row",
    "record_reservation_status",
    "record_tool_call",
    "set_call_identifiers",
]
