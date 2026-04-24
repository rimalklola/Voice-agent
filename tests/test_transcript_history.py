import json

import pytest

from src.utils.transcript_history import (
    TranscriptSessionBuffer,
    get_transcript_record,
    list_transcript_records,
    render_transcript_text,
    save_transcript_record,
)


def test_transcript_session_buffer_orders_committed_items_first():
    buffer = TranscriptSessionBuffer(voice="sage", model="gpt-realtime-2025-08-28", temperature=0.65)
    buffer.set_call_identifiers(call_sid="CA111", stream_sid="MS222")

    buffer.mark_item_committed("item_user_1", "")
    added = buffer.add_turn(
        speaker="customer",
        text="Bonjour, je cherche une villa.",
        event_type="conversation.item.input_audio_transcription.completed",
        item_id="item_user_1",
    )
    assert added

    buffer.add_turn(
        speaker="assistant",
        text="Bonjour, je peux vous aider.",
        event_type="response.audio_transcript.done",
        response_id="resp_1",
    )

    record = buffer.to_record(session_status="completed", failure_reason="", duration_ms=1200)

    assert record["call_sid"] == "CA111"
    assert record["stream_sid"] == "MS222"
    assert record["turn_count"] == 2
    assert record["turns"][0]["speaker"] == "customer"
    assert "Customer:" in record["transcript_full_text"]
    assert "Assistant:" in record["transcript_full_text"]


def test_transcript_history_save_list_get_and_render(tmp_path):
    sqlite_path = tmp_path / "call_ledger.sqlite3"
    buffer = TranscriptSessionBuffer(voice="sage", model="gpt-realtime-2025-08-28", temperature=0.65)
    buffer.set_call_identifiers(call_sid="CA555", stream_sid="MS999")
    buffer.mark_item_committed("item_1", "")

    buffer.add_turn(
        speaker="customer",
        text="Je veux un appartement F2.",
        event_type="conversation.item.input_audio_transcription.completed",
        item_id="item_1",
    )
    buffer.add_turn(
        speaker="assistant",
        text="Très bien, je vous propose une option.",
        event_type="response.output_text.done",
        response_id="resp_1",
    )

    record = buffer.to_record(session_status="completed", failure_reason="", duration_ms=2400)
    output_path = save_transcript_record(tmp_path, record, sqlite_path=sqlite_path)
    assert output_path.exists()

    fetched = get_transcript_record(tmp_path, record["call_id"])
    assert fetched["call_id"] == record["call_id"]
    assert fetched["turn_count"] == 2

    listed = list_transcript_records(tmp_path, limit=10, call_sid="CA555", sqlite_path=sqlite_path)
    assert len(listed) == 1
    assert listed[0]["call_id"] == record["call_id"]

    plain_text = render_transcript_text(fetched)
    assert "Call ID:" in plain_text
    assert "Customer:" in plain_text
    assert "Assistant:" in plain_text


def test_transcript_history_list_filters_by_query_and_status(tmp_path):
    sqlite_path = tmp_path / "call_ledger.sqlite3"
    first = {
        "call_id": "call_a",
        "call_started_utc": "2026-03-24T10:00:00+00:00",
        "call_finished_utc": "2026-03-24T10:01:00+00:00",
        "duration_ms": 60000,
        "session_status": "completed",
        "failure_reason": "",
        "call_sid": "CA_A",
        "stream_sid": "MS_A",
        "voice": "sage",
        "model": "gpt-realtime-2025-08-28",
        "temperature": 0.65,
        "turn_count": 1,
        "customer_turn_count": 1,
        "assistant_turn_count": 0,
        "transcript_full_text": "Customer: Bonjour",
        "turns": [{"speaker": "customer", "text": "Bonjour"}],
    }
    second = {
        "call_id": "call_b",
        "call_started_utc": "2026-03-24T11:00:00+00:00",
        "call_finished_utc": "2026-03-24T11:01:00+00:00",
        "duration_ms": 60000,
        "session_status": "aborted",
        "failure_reason": "network",
        "call_sid": "CA_B",
        "stream_sid": "MS_B",
        "voice": "sage",
        "model": "gpt-realtime-2025-08-28",
        "temperature": 0.65,
        "turn_count": 1,
        "customer_turn_count": 0,
        "assistant_turn_count": 1,
        "transcript_full_text": "Assistant: Au revoir",
        "turns": [{"speaker": "assistant", "text": "Au revoir"}],
    }

    save_transcript_record(tmp_path, first, sqlite_path=sqlite_path)
    save_transcript_record(tmp_path, second, sqlite_path=sqlite_path)

    by_status = list_transcript_records(tmp_path, session_status="completed", limit=10, sqlite_path=sqlite_path)
    assert len(by_status) == 1
    assert by_status[0]["call_id"] == "call_a"

    by_query = list_transcript_records(tmp_path, query="au revoir", limit=10, sqlite_path=sqlite_path)
    assert len(by_query) == 1
    assert by_query[0]["call_id"] == "call_b"


def test_get_transcript_record_raises_for_missing_call(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_transcript_record(tmp_path, "missing_call")


def test_saved_transcript_json_is_valid(tmp_path):
    sqlite_path = tmp_path / "call_ledger.sqlite3"
    record = {
        "call_id": "call_json",
        "call_started_utc": "2026-03-24T12:00:00+00:00",
        "call_finished_utc": "2026-03-24T12:01:00+00:00",
        "duration_ms": 60000,
        "session_status": "completed",
        "failure_reason": "",
        "call_sid": "CA_JSON",
        "stream_sid": "MS_JSON",
        "voice": "sage",
        "model": "gpt-realtime-2025-08-28",
        "temperature": 0.65,
        "turn_count": 0,
        "customer_turn_count": 0,
        "assistant_turn_count": 0,
        "transcript_full_text": "",
        "turns": [],
    }

    path = save_transcript_record(tmp_path, record, sqlite_path=sqlite_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["call_id"] == "call_json"


def test_transcript_record_is_chronological_and_resequenced():
    buffer = TranscriptSessionBuffer(voice="sage", model="gpt-realtime-2025-08-28", temperature=0.65)

    # Assistant speaks first in time.
    buffer.add_turn(
        speaker="assistant",
        text="Bonjour.",
        event_type="response.output_audio_transcript.done",
        item_id="item_a1",
    )

    # Customer turn can have committed metadata but must still appear chronologically.
    buffer.mark_item_committed("item_u1", "item_a1")
    buffer.add_turn(
        speaker="customer",
        text="Bonjour, un appartement.",
        event_type="conversation.item.input_audio_transcription.completed",
        item_id="item_u1",
    )

    record = buffer.to_record(session_status="completed", failure_reason="", duration_ms=1200)

    turns = record["turns"]
    assert len(turns) == 2
    assert turns[0]["speaker"] == "assistant"
    assert turns[1]["speaker"] == "customer"
    assert [t["sequence"] for t in turns] == [1, 2]


def test_transcript_fallback_customer_marker_is_supported():
    buffer = TranscriptSessionBuffer(voice="sage", model="gpt-realtime-2025-08-28", temperature=0.65)
    buffer.add_turn(
        speaker="customer",
        text="(audio détecté, transcription indisponible)",
        event_type="input_audio_buffer.committed",
        item_id="item_u_missing",
    )

    record = buffer.to_record(session_status="completed", failure_reason="", duration_ms=500)
    assert record["customer_turn_count"] == 1
    assert "Customer: (audio détecté, transcription indisponible)" in record["transcript_full_text"]
