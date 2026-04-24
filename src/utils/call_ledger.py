from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CALL_COLUMNS = [
    "call_id",
    "call_sid",
    "stream_sid",
    "call_started_utc",
    "call_finished_utc",
    "duration_ms",
    "session_status",
    "failure_reason",
    "retry_attempts",
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
    "reservation_ok",
    "reservation_duplicate",
    "reservation_error",
    "turn_count",
    "customer_turn_count",
    "assistant_turn_count",
    "transcript_path",
    "transcript_preview",
]

CALL_INTEGER_COLUMNS = {
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
    "reservation_ok",
    "reservation_duplicate",
    "reservation_error",
    "turn_count",
    "customer_turn_count",
    "assistant_turn_count",
}

CALL_REAL_COLUMNS = {"temperature"}

RESERVATION_COLUMNS = [
    "call_id",
    "timestamp_utc",
    "full_name",
    "email",
    "phone",
    "phone_display",
    "lead_source",
    "request_type",
    "requested_date",
    "requested_time",
    "guests",
    "notes",
    "source",
    "status",
    "fallback_used",
]

_DB_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = _ensure_parent_dir(db_path)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _initialize_schema(conn)
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                call_sid TEXT NOT NULL DEFAULT '',
                stream_sid TEXT NOT NULL DEFAULT '',
                call_started_utc TEXT NOT NULL DEFAULT '',
                call_finished_utc TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER NOT NULL DEFAULT 0,
                session_status TEXT NOT NULL DEFAULT 'unknown',
                failure_reason TEXT NOT NULL DEFAULT '',
                retry_attempts INTEGER NOT NULL DEFAULT 0,
                voice TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                temperature REAL NOT NULL DEFAULT 0,
                user_turns INTEGER NOT NULL DEFAULT 0,
                assistant_responses INTEGER NOT NULL DEFAULT 0,
                rate_limit_events INTEGER NOT NULL DEFAULT 0,
                realtime_error_events INTEGER NOT NULL DEFAULT 0,
                twilio_idle_timeouts INTEGER NOT NULL DEFAULT 0,
                twilio_disconnects INTEGER NOT NULL DEFAULT 0,
                openai_disconnects INTEGER NOT NULL DEFAULT 0,
                tool_calls_total INTEGER NOT NULL DEFAULT 0,
                tool_ask_to_reserve INTEGER NOT NULL DEFAULT 0,
                tool_get_project_info INTEGER NOT NULL DEFAULT 0,
                tool_get_property_specs INTEGER NOT NULL DEFAULT 0,
                tool_get_project_facts INTEGER NOT NULL DEFAULT 0,
                tool_end_call INTEGER NOT NULL DEFAULT 0,
                caller_wants_reservation INTEGER NOT NULL DEFAULT 0,
                reservation_status TEXT NOT NULL DEFAULT 'not_requested',
                reservation_request_type TEXT NOT NULL DEFAULT '',
                reservation_ok INTEGER NOT NULL DEFAULT 0,
                reservation_duplicate INTEGER NOT NULL DEFAULT 0,
                reservation_error INTEGER NOT NULL DEFAULT 0,
                turn_count INTEGER NOT NULL DEFAULT 0,
                customer_turn_count INTEGER NOT NULL DEFAULT 0,
                assistant_turn_count INTEGER NOT NULL DEFAULT 0,
                transcript_path TEXT NOT NULL DEFAULT '',
                transcript_preview TEXT NOT NULL DEFAULT '',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL DEFAULT '',
                timestamp_utc TEXT NOT NULL,
                full_name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                phone_display TEXT NOT NULL DEFAULT '',
                lead_source TEXT NOT NULL DEFAULT '',
                request_type TEXT NOT NULL DEFAULT 'rappel',
                requested_date TEXT NOT NULL DEFAULT '',
                requested_time TEXT NOT NULL DEFAULT '',
                guests INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'twilio',
                status TEXT NOT NULL DEFAULT '',
                fallback_used INTEGER NOT NULL DEFAULT 0,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_started_at ON calls(call_started_utc DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_call_sid ON calls(call_sid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(session_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_reservation_status ON calls(reservation_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reservations_call_id ON reservations(call_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reservations_phone_date ON reservations(phone, requested_date, requested_time, request_type)")


def _serialize_call_row(row: Dict[str, Any]) -> Dict[str, Any]:
    now = _utc_now_iso()
    serialized: Dict[str, Any] = {"created_at_utc": now, "updated_at_utc": now}
    for column in CALL_COLUMNS:
        value = row.get(column, "")
        if column in CALL_INTEGER_COLUMNS:
            try:
                serialized[column] = int(value)
            except (TypeError, ValueError):
                serialized[column] = 0
        elif column in CALL_REAL_COLUMNS:
            try:
                serialized[column] = float(value)
            except (TypeError, ValueError):
                serialized[column] = 0.0
        else:
            serialized[column] = "" if value is None else str(value)

    if not serialized["call_id"]:
        raise ValueError("call_id is required")
    if not serialized["reservation_status"]:
        serialized["reservation_status"] = "not_requested"
    return serialized


def upsert_call_record(db_path: str | Path, row: Dict[str, Any]) -> None:
    serialized = _serialize_call_row(row)
    insert_columns = CALL_COLUMNS + ["created_at_utc", "updated_at_utc"]
    placeholders = ", ".join([":" + column for column in insert_columns])
    update_clause = ", ".join([f"{column} = excluded.{column}" for column in CALL_COLUMNS] + ["updated_at_utc = excluded.updated_at_utc"])

    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO calls ({", ".join(insert_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(call_id) DO UPDATE SET {update_clause}
                    """,
                    serialized,
                )
        finally:
            conn.close()


def insert_reservation_record(db_path: str | Path, row: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {"created_at_utc": _utc_now_iso()}
    for column in RESERVATION_COLUMNS:
        value = row.get(column, "")
        if column in {"guests", "fallback_used"}:
            try:
                payload[column] = int(value)
            except (TypeError, ValueError):
                payload[column] = 0
        else:
            payload[column] = "" if value is None else str(value)

    if not payload.get("timestamp_utc"):
        payload["timestamp_utc"] = _utc_now_iso()

    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO reservations ({", ".join(RESERVATION_COLUMNS + ['created_at_utc'])})
                    VALUES ({", ".join(':' + column for column in RESERVATION_COLUMNS + ['created_at_utc'])})
                    """,
                    payload,
                )
        finally:
            conn.close()


def reservation_exists(
    db_path: str | Path,
    *,
    phone: str,
    requested_date: str,
    requested_time: str,
    request_type: str,
) -> bool:
    normalized_phone = str(phone or "")
    normalized_date = str(requested_date or "")
    normalized_time = str(requested_time or "")
    normalized_type = str(request_type or "")

    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            row = conn.execute(
                """
                SELECT 1
                FROM reservations
                WHERE phone = ?
                  AND (? = '' OR requested_date = ?)
                  AND (? = '' OR requested_time = ?)
                  AND (? = '' OR request_type = ?)
                LIMIT 1
                """,
                (
                    normalized_phone,
                    normalized_date,
                    normalized_date,
                    normalized_time,
                    normalized_time,
                    normalized_type,
                    normalized_type,
                ),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


def list_call_records(
    db_path: str | Path,
    *,
    limit: int = 50,
    call_sid: Optional[str] = None,
    stream_sid: Optional[str] = None,
    query: Optional[str] = None,
    session_status: Optional[str] = None,
    require_transcript: bool = False,
) -> List[Dict[str, Any]]:
    normalized_limit = max(1, min(int(limit), 500))
    conditions: List[str] = []
    params: List[Any] = []

    if call_sid:
        conditions.append("lower(call_sid) = lower(?)")
        params.append(call_sid)
    if stream_sid:
        conditions.append("lower(stream_sid) = lower(?)")
        params.append(stream_sid)
    if session_status:
        conditions.append("lower(session_status) = lower(?)")
        params.append(session_status)
    if require_transcript:
        conditions.append("transcript_path <> ''")
    if query:
        conditions.append(
            "(lower(call_id) LIKE lower(?) OR lower(call_sid) LIKE lower(?) OR lower(stream_sid) LIKE lower(?) OR lower(transcript_preview) LIKE lower(?))"
        )
        search = f"%{query}%"
        params.extend([search, search, search, search])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM calls {where_clause} ORDER BY call_started_utc DESC LIMIT ?"
    params.append(normalized_limit)

    with _DB_LOCK:
        conn = _connect(db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


__all__ = [
    "CALL_COLUMNS",
    "insert_reservation_record",
    "list_call_records",
    "reservation_exists",
    "upsert_call_record",
]