from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.call_ledger import list_call_records, upsert_call_record
from src.utils.config import config

TRANSCRIPT_INDEX_FIELDS = [
    "call_id",
    "call_started_utc",
    "call_finished_utc",
    "duration_ms",
    "session_status",
    "failure_reason",
    "call_sid",
    "stream_sid",
    "voice",
    "model",
    "temperature",
    "turn_count",
    "customer_turn_count",
    "assistant_turn_count",
    "transcript_preview",
    "transcript_path",
]


@dataclass
class TranscriptTurn:
    sequence: int
    speaker: str
    text: str
    timestamp_utc: str
    event_type: str
    item_id: str = ""
    previous_item_id: str = ""
    response_id: str = ""
    committed_order: int = -1


@dataclass
class TranscriptSessionBuffer:
    voice: str
    model: str
    temperature: Any
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    call_started_utc: str = field(default_factory=lambda: _utc_now_iso())
    call_sid: str = ""
    stream_sid: str = ""
    turns: List[TranscriptTurn] = field(default_factory=list)
    _next_sequence: int = 1
    _next_commit_order: int = 1
    _committed_previous_by_item: Dict[str, str] = field(default_factory=dict)
    _committed_order_by_item: Dict[str, int] = field(default_factory=dict)

    def set_call_identifiers(self, *, call_sid: Optional[str], stream_sid: Optional[str]) -> None:
        if call_sid:
            self.call_sid = str(call_sid)
        if stream_sid:
            self.stream_sid = str(stream_sid)

    def mark_item_committed(self, item_id: Optional[str], previous_item_id: Optional[str]) -> None:
        normalized_item = (item_id or "").strip()
        if not normalized_item:
            return
        self._committed_previous_by_item[normalized_item] = (previous_item_id or "").strip()
        if normalized_item not in self._committed_order_by_item:
            self._committed_order_by_item[normalized_item] = self._next_commit_order
            self._next_commit_order += 1

    def add_turn(
        self,
        *,
        speaker: str,
        text: str,
        event_type: str,
        item_id: Optional[str] = None,
        previous_item_id: Optional[str] = None,
        response_id: Optional[str] = None,
    ) -> bool:
        normalized_text = (text or "").strip()
        normalized_speaker = (speaker or "").strip().lower()
        if not normalized_text:
            return False
        if normalized_speaker not in {"customer", "assistant"}:
            raise ValueError("speaker must be either 'customer' or 'assistant'")

        normalized_item_id = (item_id or "").strip()
        normalized_previous_item_id = (previous_item_id or "").strip()

        if normalized_item_id and not normalized_previous_item_id:
            normalized_previous_item_id = self._committed_previous_by_item.get(normalized_item_id, "")

        committed_order = -1
        if normalized_item_id:
            committed_order = self._committed_order_by_item.get(normalized_item_id, -1)

        self.turns.append(
            TranscriptTurn(
                sequence=self._next_sequence,
                speaker=normalized_speaker,
                text=normalized_text,
                timestamp_utc=_utc_now_iso(),
                event_type=(event_type or "").strip() or "unknown",
                item_id=normalized_item_id,
                previous_item_id=normalized_previous_item_id,
                response_id=(response_id or "").strip(),
                committed_order=committed_order,
            )
        )
        self._next_sequence += 1
        return True

    def has_assistant_turn_for_response(self, response_id: Optional[str]) -> bool:
        normalized = (response_id or "").strip()
        if not normalized:
            return False
        return any(t.speaker == "assistant" and t.response_id == normalized for t in self.turns)

    def to_record(
        self,
        *,
        session_status: str,
        failure_reason: str,
        duration_ms: int,
        call_finished_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        sorted_turns = _sorted_turns(self.turns)

        customer_count = sum(1 for turn in sorted_turns if turn.speaker == "customer")
        assistant_count = sum(1 for turn in sorted_turns if turn.speaker == "assistant")
        transcript_full_text = _build_full_text(sorted_turns)

        serialized_turns: List[Dict[str, Any]] = []
        for idx, turn in enumerate(sorted_turns, start=1):
            serialized_turns.append(
                {
                    "sequence": idx,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "timestamp_utc": turn.timestamp_utc,
                    "event_type": turn.event_type,
                    "item_id": turn.item_id,
                    "previous_item_id": turn.previous_item_id,
                    "response_id": turn.response_id,
                }
            )

        return {
            "call_id": self.call_id,
            "call_started_utc": self.call_started_utc,
            "call_finished_utc": call_finished_utc or _utc_now_iso(),
            "duration_ms": max(0, int(duration_ms)),
            "session_status": (session_status or "unknown").strip() or "unknown",
            "failure_reason": (failure_reason or "").strip(),
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "voice": self.voice,
            "model": self.model,
            "temperature": self.temperature,
            "turn_count": len(sorted_turns),
            "customer_turn_count": customer_count,
            "assistant_turn_count": assistant_count,
            "transcript_full_text": transcript_full_text,
            "turns": serialized_turns,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc_iso(raw: str) -> datetime:
    value = (raw or "").strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sorted_turns(turns: List[TranscriptTurn]) -> List[TranscriptTurn]:
    return sorted(turns, key=lambda t: (_parse_utc_iso(t.timestamp_utc), t.sequence))


def _build_full_text(turns: List[TranscriptTurn]) -> str:
    lines = [f"{turn.speaker.capitalize()}: {turn.text}" for turn in turns]
    return "\n".join(lines)


def _preview(text: str, max_chars: int = 200) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _coerce_index_row(record: Dict[str, Any], transcript_path: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for field in TRANSCRIPT_INDEX_FIELDS:
        if field == "transcript_path":
            row[field] = transcript_path
            continue
        value = record.get(field, "")
        row[field] = "" if value is None else value

    if not row.get("transcript_preview"):
        row["transcript_preview"] = _preview(str(record.get("transcript_full_text") or ""))
    return row


def _transcripts_dir(base_dir: str | Path) -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _calls_dir(base_dir: str | Path) -> Path:
    calls_path = _transcripts_dir(base_dir) / "calls"
    calls_path.mkdir(parents=True, exist_ok=True)
    return calls_path


def _default_sqlite_path() -> Path:
    monitoring_cfg = config.get("monitoring", {}) if isinstance(config, dict) else {}
    if not isinstance(monitoring_cfg, dict):
        monitoring_cfg = {}
    return Path(str(monitoring_cfg.get("sqlite_path", "./data/monitoring/call_ledger.sqlite3")))


def save_transcript_record(base_dir: str | Path, record: Dict[str, Any], *, sqlite_path: str | Path | None = None) -> Path:
    calls_dir = _calls_dir(base_dir)
    call_id = str(record.get("call_id") or "").strip()
    if not call_id:
        raise ValueError("record.call_id is required")

    target = calls_dir / f"{call_id}.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    row = _coerce_index_row(record, str(target))
    upsert_call_record(sqlite_path or _default_sqlite_path(), row)

    return target


def get_transcript_record(base_dir: str | Path, call_id: str) -> Dict[str, Any]:
    normalized_id = (call_id or "").strip()
    if not normalized_id:
        raise ValueError("call_id is required")
    path = _calls_dir(base_dir) / f"{normalized_id}.json"
    if not path.exists():
        raise FileNotFoundError(normalized_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_transcript_records(
    base_dir: str | Path,
    *,
    limit: int = 50,
    call_sid: Optional[str] = None,
    stream_sid: Optional[str] = None,
    query: Optional[str] = None,
    session_status: Optional[str] = None,
    sqlite_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    return list_call_records(
        sqlite_path or _default_sqlite_path(),
        limit=limit,
        call_sid=call_sid,
        stream_sid=stream_sid,
        query=query,
        session_status=session_status,
        require_transcript=True,
    )


def render_transcript_text(record: Dict[str, Any]) -> str:
    header = [
        f"Call ID: {record.get('call_id', '')}",
        f"Call SID: {record.get('call_sid', '')}",
        f"Stream SID: {record.get('stream_sid', '')}",
        f"Started: {record.get('call_started_utc', '')}",
        f"Finished: {record.get('call_finished_utc', '')}",
        f"Duration (ms): {record.get('duration_ms', 0)}",
        f"Status: {record.get('session_status', '')}",
        "",
        "Transcript:",
    ]

    body: List[str] = []
    for turn in record.get("turns") or []:
        ts = str(turn.get("timestamp_utc") or "")
        speaker = str(turn.get("speaker") or "unknown").capitalize()
        text = str(turn.get("text") or "")
        body.append(f"[{ts}] {speaker}: {text}")

    if not body:
        body.append("(No transcript turns captured)")

    return "\n".join(header + body)


__all__ = [
    "TRANSCRIPT_INDEX_FIELDS",
    "TranscriptSessionBuffer",
    "get_transcript_record",
    "list_transcript_records",
    "render_transcript_text",
    "save_transcript_record",
]
