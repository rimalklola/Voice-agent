import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from src.utils import tools as tools_mod


class FakeOpenAIWS:
    def __init__(self):
        self.sent = []

    async def send(self, payload: str):
        try:
            self.sent.append(json.loads(payload))
        except Exception:
            self.sent.append(payload)


def _fetch_sqlite_rows(db_path, query, params=()):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def test_initialize_session_registers_tools():
    from src.utils import realtime as rt

    if not hasattr(rt, "handle_media_stream"):
        pytest.skip("FastAPI not installed; handle_media_stream unavailable")

    fake = FakeOpenAIWS()

    async def run():
        await rt.initialize_session(fake)

    asyncio.run(run())
    updates = [m for m in fake.sent if isinstance(m, dict) and m.get("type") == "session.update"]
    assert updates, "Missing session.update message"
    tools = updates[-1]["session"].get("tools", [])
    names = sorted(t.get("name") for t in tools if t.get("type") == "function")
    assert names == ["ask_to_reserve", "end_call", "get_project_facts", "get_project_info", "get_property_specs"]
    reserve_tool = next(t for t in tools if t.get("name") == "ask_to_reserve")
    assert reserve_tool["parameters"]["required"] == ["full_name", "phone", "lead_source"]


def test_initialize_session_enables_input_audio_transcription():
    from src.utils import realtime as rt

    fake = FakeOpenAIWS()

    async def run():
        await rt.initialize_session(fake)

    asyncio.run(run())

    updates = [m for m in fake.sent if isinstance(m, dict) and m.get("type") == "session.update"]
    assert updates, "Missing session.update message"
    session = updates[-1]["session"]
    input_cfg = (session.get("audio") or {}).get("input") or {}
    transcription_cfg = input_cfg.get("transcription") or {}
    assert transcription_cfg.get("model")
    assert transcription_cfg.get("language")


def test_initialize_session_adds_opening_message():
    from src.utils import realtime as rt

    fake = FakeOpenAIWS()

    async def run():
        await rt.initialize_session(fake)

    asyncio.run(run())
    greetings = [m for m in fake.sent if isinstance(m, dict) and m.get("type") == "conversation.item.create"]
    assert greetings, "Expected initial conversation item"
    payload = greetings[0]["item"]["content"][0]["text"]
    assert "Alma Resort" in payload


def test_initialize_session_instructions_cover_lot_delivery_dates():
    from src.utils import realtime as rt

    fake = FakeOpenAIWS()

    async def run():
        await rt.initialize_session(fake)

    asyncio.run(run())
    updates = [m for m in fake.sent if isinstance(m, dict) and m.get("type") == "session.update"]
    assert updates, "Missing session.update message"
    instructions = updates[-1]["session"].get("instructions", "")
    assert "category=villa_lots" in instructions
    assert "date de livraison" in instructions.lower()
    assert "n'indique pas qu'il faut vérifier avec l'équipe commerciale" in instructions.lower()


def test_initialize_session_property_specs_tool_mentions_villa_lots_for_terrain():
    from src.utils import realtime as rt

    fake = FakeOpenAIWS()

    async def run():
        await rt.initialize_session(fake)

    asyncio.run(run())
    updates = [m for m in fake.sent if isinstance(m, dict) and m.get("type") == "session.update"]
    assert updates, "Missing session.update message"
    tools = updates[-1]["session"].get("tools", [])
    specs_tool = next(t for t in tools if t.get("name") == "get_property_specs")
    category_desc = specs_tool["parameters"]["properties"]["category"]["description"]
    attrs_desc = specs_tool["parameters"]["properties"]["attributes"]["description"]
    assert "terrain" in category_desc.lower()
    assert "villa_lots" in category_desc
    assert "delivery_timeline" in attrs_desc


def test_validate_realtime_model_accepts_known_variants():
    from src.utils import realtime as rt

    assert rt._validate_realtime_model("gpt-realtime-2025-08-28") == "gpt-realtime-2025-08-28"
    assert rt._validate_realtime_model("gpt-4o-realtime-preview-2024-12-17") == "gpt-4o-realtime-preview-2024-12-17"


def test_validate_realtime_model_rejects_non_realtime_model():
    from src.utils import realtime as rt

    with pytest.raises(RuntimeError, match="Invalid realtime model configured"):
        rt._validate_realtime_model("gpt-4o-mini-transcribe")


def test_build_openai_realtime_url_uses_only_model_query():
    from src.utils import realtime as rt

    url = rt._build_openai_realtime_url("gpt-realtime-2025-08-28")
    assert url == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2025-08-28"
    assert "temperature" not in url


def test_run_session_tasks_cancels_sibling_on_failure():
    from src.utils import realtime as rt

    events = []

    async def receive_from_twilio():
        events.append("receive_started")
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def send_to_twilio():
        events.append("send_started")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append("send_cancelled")
            raise

    async def run():
        with pytest.raises(RuntimeError, match="boom"):
            await rt._run_session_tasks(receive_from_twilio, send_to_twilio)

    asyncio.run(run())
    assert events[:2] == ["receive_started", "send_started"]
    assert "send_cancelled" in events


def test_receive_twilio_message_times_out():
    from src.utils import realtime as rt

    class SlowTwilioSocket:
        async def receive_text(self):
            await asyncio.sleep(0.05)
            return "late"

    async def run():
        with pytest.raises(rt.TwilioIdleTimeoutError):
            await rt._receive_twilio_message(SlowTwilioSocket(), timeout_s=0.01)

    asyncio.run(run())


def test_session_buffer_helpers_cap_growth():
    from src.utils import realtime as rt

    response_text_buffers = {}
    for idx in range(rt.SESSION_BUFFER_MAX_ENTRIES + 1):
        rt._append_response_text_delta(response_text_buffers, f"rid-{idx}", "x")
    assert len(response_text_buffers) == rt.SESSION_BUFFER_MAX_ENTRIES

    fn_args_buffers = {}
    rt._append_function_call_delta(fn_args_buffers, "call-1", "a" * (rt.SESSION_FUNCTION_ARGS_MAX_CHARS + 10))
    assert len(fn_args_buffers["call-1"]) == rt.SESSION_FUNCTION_ARGS_MAX_CHARS

    tool_attempts = {}
    for idx in range(rt.SESSION_TOOL_ATTEMPTS_MAX_ENTRIES + 2):
        rt._record_tool_attempt(tool_attempts, f"tool-{idx}")
    assert len(tool_attempts) == rt.SESSION_TOOL_ATTEMPTS_MAX_ENTRIES


def test_should_drop_inbound_media_frame_respects_first_turn_window():
    from src.utils import realtime as rt

    assert not rt._should_drop_inbound_media_frame(
        now=10.0,
        is_playback_active=True,
        playback_guard_until=12.0,
        first_turn_listening_until=11.0,
    )

    assert rt._should_drop_inbound_media_frame(
        now=12.1,
        is_playback_active=False,
        playback_guard_until=13.0,
        first_turn_listening_until=11.0,
    )


def test_should_ignore_greeting_echo_is_exact_and_time_bounded():
    from src.utils import realtime as rt

    assert rt._should_ignore_greeting_echo(
        normalized_transcript="bonjour, alma resort à l'appareil. je vous écoute.",
        ignore_next_transcript=True,
        greeting_signature="bonjour, alma resort à l'appareil. je vous écoute.",
        greeting_completed_at=10.0,
        now=10.5,
        echo_window_s=1.2,
    )

    assert not rt._should_ignore_greeting_echo(
        normalized_transcript="bonjour, je cherche un appartement",
        ignore_next_transcript=True,
        greeting_signature="bonjour, alma resort à l'appareil. je vous écoute.",
        greeting_completed_at=10.0,
        now=10.5,
        echo_window_s=1.2,
    )

    assert not rt._should_ignore_greeting_echo(
        normalized_transcript="bonjour, alma resort à l'appareil. je vous écoute.",
        ignore_next_transcript=True,
        greeting_signature="bonjour, alma resort à l'appareil. je vous écoute.",
        greeting_completed_at=10.0,
        now=12.0,
        echo_window_s=1.2,
    )


def test_should_ignore_greeting_phase_audio_only_while_playback_active():
    from src.utils import realtime as rt

    assert rt._should_ignore_greeting_phase_audio(
        greeting_response_pending=True,
        is_playback_active=True,
    )

    assert not rt._should_ignore_greeting_phase_audio(
        greeting_response_pending=True,
        is_playback_active=False,
    )

    assert not rt._should_ignore_greeting_phase_audio(
        greeting_response_pending=False,
        is_playback_active=True,
    )


def test_get_project_info_runs_via_thread(monkeypatch):
    captured = {}

    tools_mod._PROJECT_INFO_ATTEMPTS = deque()
    tools_mod._PROJECT_INFO_RECENT = {}
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_BLOCKLIST", tuple())
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_LIMIT", 5)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_WINDOW", 60.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_DUPLICATE_WINDOW", 30.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_MIN_LENGTH", 3)

    async def fake_to_thread(func, *args, **kwargs):
        captured["called"] = True
        return func(*args, **kwargs)

    monkeypatch.setattr(tools_mod.asyncio, "to_thread", fake_to_thread)

    called = {}

    def fake_retrieve(query, top_k):
        called["query"] = query
        called["top_k"] = top_k
        return [{"text": "stub", "source_type": "csv", "source_path": "stub", "row_idx": 0, "page": None}]

    monkeypatch.setattr(tools_mod, "retrieve_context", fake_retrieve)

    async def run():
        return await tools_mod.get_project_info("hello", top_k=2)

    result = asyncio.run(run())
    assert captured.get("called")
    assert called == {"query": "hello", "top_k": 2}
    assert result["status"] == "ok"
    assert result["snippets"]


def test_get_project_info_blocked_keyword(monkeypatch):
    tools_mod._PROJECT_INFO_ATTEMPTS = deque()
    tools_mod._PROJECT_INFO_RECENT = {}
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_BLOCKLIST", ("motsecret",))
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_MIN_LENGTH", 3)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_DUPLICATE_WINDOW", 30.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_LIMIT", 5)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_WINDOW", 60.0)

    async def passthrough(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(tools_mod.asyncio, "to_thread", passthrough)
    monkeypatch.setattr(tools_mod, "retrieve_context", lambda *_: [])

    async def run():
        return await tools_mod.get_project_info("Veuillez me donner le motsecret", top_k=2)

    result = asyncio.run(run())
    assert result["status"] == "blocked_keyword"
    assert result["snippets"] == []


def test_get_project_info_rate_limited(monkeypatch):
    tools_mod._PROJECT_INFO_ATTEMPTS = deque()
    tools_mod._PROJECT_INFO_RECENT = {}
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_BLOCKLIST", tuple())
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_LIMIT", 1)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_WINDOW", 60.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_DUPLICATE_WINDOW", 0.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_MIN_LENGTH", 3)

    async def passthrough(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(tools_mod.asyncio, "to_thread", passthrough)
    monkeypatch.setattr(tools_mod, "retrieve_context", lambda *_: [])

    async def first_run():
        return await tools_mod.get_project_info("Quelle est la politique", top_k=2)

    async def second_run():
        return await tools_mod.get_project_info("Encore une question", top_k=2)

    first = asyncio.run(first_run())
    assert first["status"] == "ok"

    second = asyncio.run(second_run())
    assert second["status"] == "rate_limited"
    assert second["snippets"] == []


def test_get_project_info_duplicate_guard(monkeypatch):
    tools_mod._PROJECT_INFO_ATTEMPTS = deque()
    tools_mod._PROJECT_INFO_RECENT = {}
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_BLOCKLIST", tuple())
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_LIMIT", 5)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_RATE_WINDOW", 60.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_DUPLICATE_WINDOW", 3600.0)
    monkeypatch.setattr(tools_mod, "PROJECT_INFO_MIN_LENGTH", 3)

    async def passthrough(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(tools_mod.asyncio, "to_thread", passthrough)

    def fake_retrieve_duplicate(query, top_k):
        return [{"text": "stub", "source_type": "csv", "source_path": "stub", "row_idx": 0, "page": None}]

    monkeypatch.setattr(tools_mod, "retrieve_context", fake_retrieve_duplicate)

    async def run():
        return await tools_mod.get_project_info("Parle-moi des chambres", top_k=2)

    first = asyncio.run(run())
    assert first["status"] == "ok"

    second = asyncio.run(run())
    assert second["status"] == "duplicate"
    assert second["snippets"] == []


def test_get_property_specs_success():
    async def run():
        return await tools_mod.get_property_specs("apartments_ript", "f2", ["interior_area_m2", "terrace_area_m2"])

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert result["variant"] == "f2"
    assert len(result["attributes"]) == 2
    assert "50" in result["answer"]


def test_get_property_specs_unknown_variant():
    async def run():
        return await tools_mod.get_property_specs("villas_premium", "type99")

    result = asyncio.run(run())
    assert result["status"] == "unknown_variant"
    assert "catégorie" in result["message"].lower() or "type" in result["message"].lower()


def test_get_property_specs_aggregate_variants():
    async def run():
        return await tools_mod.get_property_specs("apartments_residence")

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert result.get("variants")
    labels = [v["variant_label"] for v in result["variants"]]
    assert "F2 Résidence" in labels


def test_get_property_specs_marks_past_delivery_as_delivered(monkeypatch):
    monkeypatch.setattr(tools_mod, "_utcnow", lambda: datetime(2026, 3, 24, tzinfo=timezone.utc))

    async def run():
        return await tools_mod.get_property_specs("lots", "bande", ["delivery_timeline"])

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert "déjà livrés en décembre 2025" in result["answer"].lower()
    assert result["attributes"][0]["value"].lower() == "lots titrés déjà livrés en décembre 2025"


def test_get_project_facts_by_topic():
    async def run():
        return await tools_mod.get_project_facts(section="location", topic="exact_location")

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert "Cap Malabata" in result["answer"]


def test_get_project_facts_delivery_uses_past_tense_when_date_has_passed(monkeypatch):
    monkeypatch.setattr(tools_mod, "_utcnow", lambda: datetime(2026, 3, 24, tzinfo=timezone.utc))

    async def run():
        return await tools_mod.get_project_facts(section="project_overview", topic="progress")

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert "lotissement déjà livré fin décembre 2025" in result["answer"].lower()
    assert "septembre 2027" in result["answer"].lower()


def test_get_project_facts_unknown_section():
    async def run():
        return await tools_mod.get_project_facts(section="unknown")

    result = asyncio.run(run())
    assert result["status"] == "unknown_section"


def test_ask_to_reserve_fallback_on_retries(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "call_ledger.sqlite3"
    monkeypatch.setattr(tools_mod, "CALL_LEDGER_SQLITE_PATH", str(sqlite_path))
    tools_mod._RESERVATION_CONFIRM_FAILS = {}
    tools_mod._RESERVATION_RECENT = {}
    tools_mod._RESERVATION_ATTEMPTS = deque()

    # First attempt: invalid email triggers retry
    async def first_attempt():
        return await tools_mod.ask_to_reserve({
            "call_id": "call_fallback_1",
            "full_name": "Test User",
            "email": "not-an-email",
            "phone": "0612345678",
            "lead_source": "reseaux",
            "date": "2025-01-01",
            "time": "10:00",
            "guests": 2,
            "notes": "",
            "confirmed": True,
        })

    first_result = asyncio.run(first_attempt())
    assert first_result["status"] == "invalid_fields"

    # Second attempt should fallback and record reservation
    async def second_attempt():
        return await tools_mod.ask_to_reserve({
            "call_id": "call_fallback_1",
            "full_name": "Test User",
            "email": "not-an-email",
            "phone": "0612345678",
            "lead_source": "reseaux",
            "date": "2025-01-01",
            "time": "10:00",
            "guests": 2,
            "notes": "",
            "confirmed": True,
        })

    second_result = asyncio.run(second_attempt())
    assert second_result["status"] == "ok"
    assert second_result.get("fallback")
    rows = _fetch_sqlite_rows(sqlite_path, "SELECT * FROM reservations WHERE call_id = ?", ("call_fallback_1",))
    assert len(rows) == 1
    assert "coordonnées à confirmer" in rows[0]["notes"]
    assert rows[0]["fallback_used"] == 1


def test_ask_to_reserve_writes_sqlite(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "call_ledger.sqlite3"
    monkeypatch.setattr(tools_mod, "CALL_LEDGER_SQLITE_PATH", str(sqlite_path))
    monkeypatch.setattr(tools_mod, "_RESERVATION_RECENT", {})
    monkeypatch.setattr(tools_mod, "_RESERVATION_ATTEMPTS", deque())

    async def run():
        return await tools_mod.ask_to_reserve({
            "call_id": "call_reservation_1",
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+212612345678",
            "lead_source": "ami",
            "request_type": "visite",
            "date": "2024-12-25",
            "time": "19:00",
            "guests": 2,
            "notes": "Corner table",
            "confirmed": True,
        })

    result = asyncio.run(run())
    assert result["status"] == "ok"
    rows = _fetch_sqlite_rows(sqlite_path, "SELECT * FROM reservations WHERE call_id = ?", ("call_reservation_1",))
    assert len(rows) == 1
    assert rows[0]["full_name"] == "Jane Doe"
    assert rows[0]["phone"] == "+212612345678"


def test_ask_to_reserve_missing_fields():
    tools_mod._RESERVATION_RECENT = {}
    tools_mod._RESERVATION_ATTEMPTS = deque()

    async def run():
        return await tools_mod.ask_to_reserve({"full_name": "", "phone": "", "lead_source": ""})

    result = asyncio.run(run())
    assert result["status"] == "missing_fields"
    assert set(result["missing"]) == {"full_name", "phone", "lead_source", "confirmed"}


def test_ask_to_reserve_invalid_email():
    tools_mod._RESERVATION_RECENT = {}
    tools_mod._RESERVATION_ATTEMPTS = deque()

    async def run():
        return await tools_mod.ask_to_reserve({
            "full_name": "John Doe",
            "email": "not-an-email",
            "phone": "0612345678",
            "lead_source": "presse",
            "confirmed": True,
        })

    result = asyncio.run(run())
    assert result["status"] == "invalid_fields"
    assert "email" in result["invalid"]


def test_ask_to_reserve_requires_confirmation():
    tools_mod._RESERVATION_RECENT = {}
    tools_mod._RESERVATION_ATTEMPTS = deque()

    async def run():
        return await tools_mod.ask_to_reserve({
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "0612345678",
            "lead_source": "ami",
            "date": "2024-12-25",
            "time": "19:00",
            "guests": 2,
            "notes": "Corner table",
            "confirmed": False,
        })

    result = asyncio.run(run())
    assert result["status"] == "needs_confirmation"


def test_ask_to_reserve_duplicate_guard(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "call_ledger.sqlite3"
    monkeypatch.setattr(tools_mod, "CALL_LEDGER_SQLITE_PATH", str(sqlite_path))
    monkeypatch.setattr(tools_mod, "RESERVATION_DUPLICATE_WINDOW", 3600.0)
    monkeypatch.setattr(tools_mod, "RESERVATION_RATE_LIMIT", 10)
    monkeypatch.setattr(tools_mod, "_RESERVATION_RECENT", {})
    monkeypatch.setattr(tools_mod, "_RESERVATION_ATTEMPTS", deque())
    tools_mod.config.guardrails.reservations.allow_duplicate_without_prompt = False

    async def call(**extra):
        payload = {
            "call_id": "call_duplicate_test",
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "0612345678",
            "lead_source": "reseaux",
            "request_type": "rappel",
            "date": "2024-12-25",
            "time": "19:00",
            "confirmed": True,
        }
        payload.update(extra)
        return await tools_mod.ask_to_reserve(payload)

    first = asyncio.run(call())
    assert first["status"] == "ok"

    second = asyncio.run(call())
    assert second["status"] == "duplicate"

    third = asyncio.run(call(override_duplicate=True))
    assert third["status"] == "ok"


def test_handle_media_stream_smoke(monkeypatch):
    from src.utils import realtime as rt

    if not hasattr(rt, "handle_media_stream"):
        pytest.skip("FastAPI not installed; handle_media_stream unavailable")

    sent_to_twilio = []

    class FakeTwilioWebSocket:
        def __init__(self):
            self.accepted = False
            self.closed = False
            self._messages = [
                json.dumps({"event": "connected"}),
                json.dumps({"event": "start", "start": {"streamSid": "sid-123"}}),
                json.dumps({"event": "stop"}),
            ]

        async def accept(self):
            self.accepted = True

        async def iter_text(self):
            for msg in self._messages:
                yield msg

        async def send_json(self, payload):
            sent_to_twilio.append(payload)

        async def close(self, *_, **__):
            self.closed = True

    class FakeOpenAIState:
        name = "OPEN"

    class FakeOpenAIWS:
        def __init__(self):
            self.sent = []
            self.state = FakeOpenAIState()

        async def send(self, payload: str):
            try:
                self.sent.append(json.loads(payload))
            except Exception:
                self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            return None

    fake_openai = FakeOpenAIWS()

    class DummyContextManager:
        def __init__(self, client):
            self.client = client

        async def __aenter__(self):
            return self.client

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def dummy_connect(*_args, **_kwargs):
        return DummyContextManager(fake_openai)

    monkeypatch.setattr(rt.websockets, "connect", dummy_connect)

    async def run():
        websocket = FakeTwilioWebSocket()
        await rt.handle_media_stream(websocket)
        return websocket

    websocket = asyncio.run(run())

    assert websocket.accepted
    message_types = [m.get("type") for m in fake_openai.sent if isinstance(m, dict)]
    assert "session.update" in message_types
    assert sent_to_twilio == []


def test_transcript_history_routes_exist():
    from src.utils import realtime as rt

    if not hasattr(rt, "create_app"):
        pytest.skip("FastAPI not installed; app routes unavailable")

    app = rt.create_app()
    route_paths = {route.path for route in app.routes}
    assert "/transcripts" in route_paths
    assert "/transcripts/{call_id}" in route_paths
    assert "/transcripts/{call_id}/text" in route_paths


def test_transcript_history_endpoints(tmp_path, monkeypatch):
    from src.utils import realtime as rt
    from src.utils.transcript_history import save_transcript_record

    if not hasattr(rt, "create_app"):
        pytest.skip("FastAPI not installed; transcript endpoints unavailable")

    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover - optional dependency
        pytest.skip("fastapi.testclient unavailable")

    sqlite_path = tmp_path / "call_ledger.sqlite3"
    monkeypatch.setattr(rt, "TRANSCRIPT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setattr(rt, "CALL_LEDGER_SQLITE_PATH", str(sqlite_path))

    save_transcript_record(
        tmp_path,
        {
            "call_id": "call_test_1",
            "call_started_utc": "2026-03-24T10:00:00+00:00",
            "call_finished_utc": "2026-03-24T10:01:00+00:00",
            "duration_ms": 60000,
            "session_status": "completed",
            "failure_reason": "",
            "call_sid": "CA_TEST",
            "stream_sid": "MS_TEST",
            "voice": "sage",
            "model": "gpt-realtime-2025-08-28",
            "temperature": 0.65,
            "turn_count": 1,
            "customer_turn_count": 1,
            "assistant_turn_count": 0,
            "transcript_full_text": "Customer: Bonjour",
            "turns": [
                {
                    "sequence": 1,
                    "speaker": "customer",
                    "text": "Bonjour",
                    "timestamp_utc": "2026-03-24T10:00:10+00:00",
                    "event_type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_1",
                    "previous_item_id": "",
                    "response_id": "",
                }
            ],
        },
        sqlite_path=sqlite_path,
    )

    app = rt.create_app()
    with TestClient(app) as client:
        list_resp = client.get("/transcripts?limit=5&call_sid=CA_TEST")
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["count"] == 1
        assert payload["items"][0]["call_id"] == "call_test_1"

        detail_resp = client.get("/transcripts/call_test_1")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["call_sid"] == "CA_TEST"

        text_resp = client.get("/transcripts/call_test_1/text")
        assert text_resp.status_code == 200
        assert "Transcript:" in text_resp.text
        assert "Customer: Bonjour" in text_resp.text
