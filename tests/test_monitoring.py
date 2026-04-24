import asyncio
import sqlite3

from src.utils.monitoring import (
    append_pilot_call_row_async,
    finalize_pilot_call_row,
    increment_counter,
    new_pilot_call_row,
    record_reservation_status,
    record_tool_call,
    set_call_identifiers,
)


def test_monitoring_row_helpers_and_sqlite_write(tmp_path):
    row = new_pilot_call_row(voice="sage", model="gpt-realtime-2025-08-28", temperature=0.65)
    row["call_id"] = "call_test_monitoring"
    set_call_identifiers(row, call_sid="CA123", stream_sid="MS456")

    increment_counter(row, "user_turns", 2)
    increment_counter(row, "assistant_responses", 3)
    record_tool_call(row, "get_project_info")
    record_tool_call(row, "ask_to_reserve")
    record_reservation_status(row, "ok")
    record_reservation_status(row, "duplicate")
    record_reservation_status(row, "error")

    finalized = finalize_pilot_call_row(
        row,
        session_status="completed",
        failure_reason="",
        retry_attempts=1,
        duration_ms=4200,
    )

    sqlite_path = tmp_path / "call_ledger.sqlite3"
    asyncio.run(append_pilot_call_row_async(str(sqlite_path), finalized))

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM calls WHERE call_id = ?", ("call_test_monitoring",)).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    saved = dict(rows[0])
    assert saved["call_sid"] == "CA123"
    assert saved["stream_sid"] == "MS456"
    assert saved["session_status"] == "completed"
    assert int(saved["duration_ms"]) == 4200
    assert int(saved["retry_attempts"]) == 1
    assert int(saved["user_turns"]) == 2
    assert int(saved["assistant_responses"]) == 3
    assert int(saved["tool_calls_total"]) == 2
    assert int(saved["tool_get_project_info"]) == 1
    assert int(saved["tool_ask_to_reserve"]) == 1
    assert int(saved["reservation_ok"]) == 1
    assert int(saved["reservation_duplicate"]) == 1
    assert int(saved["reservation_error"]) == 1
