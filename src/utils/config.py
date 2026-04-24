from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE_ENV = "ALMA_CONFIG_FILE"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"

REQUIRED_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("project", "gemini_api_key"),
    ("project", "public_http_base_url"),
    ("project", "suppress_audio_logs"),
    ("realtime", "temperature"),
    ("realtime", "voice"),
    ("realtime", "connect_attempts"),
    ("realtime", "connect_backoff"),
    ("realtime", "connect_backoff_max"),
    ("realtime", "asr_model"),
    ("realtime", "playback_guard_ms"),
    ("elevenlabs", "enabled"),
    ("elevenlabs", "api_key"),
    ("elevenlabs", "voice_id"),
    ("elevenlabs", "model_id"),
    ("elevenlabs", "base_url"),
    ("elevenlabs", "optimize_latency"),
    ("elevenlabs", "stability"),
    ("elevenlabs", "similarity"),
    ("ingestion", "embed_model_name"),
    ("ingestion", "lancedb_path"),
    ("ingestion", "doc_csv_path"),
    ("ingestion", "doc_txt_path"),
    ("ingestion", "doc_pdf_path"),
    ("ingestion", "chunk_chars"),
    ("ingestion", "chunk_overlap"),
    ("retrieval", "top_k"),
    ("retrieval", "distance"),
    ("retrieval", "init_k"),
    ("retrieval", "use_mmr"),
    ("retrieval", "mmr_lambda"),
    ("retrieval", "cross_encoder_model"),
    ("retrieval", "ef_search"),
    ("retrieval", "use_lexical_boost"),
    ("retrieval", "lexical_weight"),
    ("retrieval", "csv_bias"),
    ("retrieval", "pdf_bias"),
    ("retrieval", "session_system_prompt_path"),
    ("guardrails", "reservations", "rate_limit"),
    ("guardrails", "reservations", "rate_window"),
    ("guardrails", "reservations", "duplicate_window"),
    ("guardrails", "reservations", "min_phone_digits"),
    ("guardrails", "reservations", "max_phone_digits"),
    ("guardrails", "reservations", "max_notes_length"),
    ("guardrails", "reservations", "allow_duplicate_without_prompt"),
    ("guardrails", "knowledge", "min_length"),
    ("guardrails", "knowledge", "max_query_length"),
    ("guardrails", "knowledge", "rate_limit"),
    ("guardrails", "knowledge", "rate_window"),
    ("guardrails", "knowledge", "duplicate_window"),
    ("guardrails", "knowledge", "blocklist"),
    ("guardrails", "rate_limit_hold_message"),
    ("tools", "end_call_prompt"),
    ("twilio", "account_sid"),
    ("twilio", "auth_token"),
    ("monitoring", "sqlite_path"),
)

ENVIRONMENT_OVERRIDES: Dict[str, Tuple[str, ...]] = {
    "GEMINI_API_KEY": ("project", "gemini_api_key"),
    "PUBLIC_HTTP_BASE_URL": ("project", "public_http_base_url"),
    "SUPPRESS_AUDIO_LOGS": ("project", "suppress_audio_logs"),
    "TEMPERATURE": ("realtime", "temperature"),
    "VOICE": ("realtime", "voice"),
    "REALTIME_CONNECT_ATTEMPTS": ("realtime", "connect_attempts"),
    "REALTIME_CONNECT_BACKOFF": ("realtime", "connect_backoff"),
    "REALTIME_CONNECT_BACKOFF_MAX": ("realtime", "connect_backoff_max"),
    "ASR_MODEL": ("realtime", "asr_model"),
    "REALTIME_INPUT_TRANSCRIPTION_LANGUAGE": ("realtime", "input_transcription_language"),
    "REALTIME_PLAYBACK_GUARD_MS": ("realtime", "playback_guard_ms"),
    "USE_ELEVENLABS_TTS": ("elevenlabs", "enabled"),
    "ELEVENLABS_API_KEY": ("elevenlabs", "api_key"),
    "ELEVENLABS_VOICE_ID": ("elevenlabs", "voice_id"),
    "ELEVENLABS_MODEL_ID": ("elevenlabs", "model_id"),
    "ELEVENLABS_BASE_URL": ("elevenlabs", "base_url"),
    "ELEVENLABS_OPTIMIZE_LATENCY": ("elevenlabs", "optimize_latency"),
    "ELEVENLABS_STABILITY": ("elevenlabs", "stability"),
    "ELEVENLABS_SIMILARITY": ("elevenlabs", "similarity"),
    "EMBED_MODEL_NAME": ("ingestion", "embed_model_name"),
    "LANCEDB_PATH": ("ingestion", "lancedb_path"),
    "DOC_CSV_PATH": ("ingestion", "doc_csv_path"),
    "DOC_TXT_PATH": ("ingestion", "doc_txt_path"),
    "DOC_PDF_PATH": ("ingestion", "doc_pdf_path"),
    "CHUNK_CHARS": ("ingestion", "chunk_chars"),
    "CHUNK_OVERLAP": ("ingestion", "chunk_overlap"),
    "TOP_K": ("retrieval", "top_k"),
    "DISTANCE": ("retrieval", "distance"),
    "INIT_K": ("retrieval", "init_k"),
    "USE_MMR": ("retrieval", "use_mmr"),
    "MMR_LAMBDA": ("retrieval", "mmr_lambda"),
    "CROSS_ENCODER_MODEL": ("retrieval", "cross_encoder_model"),
    "EF_SEARCH": ("retrieval", "ef_search"),
    "USE_LEXICAL_BOOST": ("retrieval", "use_lexical_boost"),
    "LEXICAL_WEIGHT": ("retrieval", "lexical_weight"),
    "CSV_BIAS": ("retrieval", "csv_bias"),
    "PDF_BIAS": ("retrieval", "pdf_bias"),
    "SESSION_SYSTEM_PROMPT_PATH": ("retrieval", "session_system_prompt_path"),
    "RESERVATION_RATE_LIMIT": ("guardrails", "reservations", "rate_limit"),
    "RESERVATION_RATE_WINDOW": ("guardrails", "reservations", "rate_window"),
    "RESERVATION_DUPLICATE_WINDOW": ("guardrails", "reservations", "duplicate_window"),
    "RESERVATION_MIN_PHONE_DIGITS": ("guardrails", "reservations", "min_phone_digits"),
    "RESERVATION_MAX_PHONE_DIGITS": ("guardrails", "reservations", "max_phone_digits"),
    "RESERVATION_MAX_NOTES_LENGTH": ("guardrails", "reservations", "max_notes_length"),
    "RESERVATION_ALLOW_DUPLICATE_WITHOUT_PROMPT": ("guardrails", "reservations", "allow_duplicate_without_prompt"),
    "PROJECT_INFO_MIN_LENGTH": ("guardrails", "knowledge", "min_length"),
    "PROJECT_INFO_MAX_QUERY_LENGTH": ("guardrails", "knowledge", "max_query_length"),
    "PROJECT_INFO_RATE_LIMIT": ("guardrails", "knowledge", "rate_limit"),
    "PROJECT_INFO_RATE_WINDOW": ("guardrails", "knowledge", "rate_window"),
    "PROJECT_INFO_DUPLICATE_WINDOW": ("guardrails", "knowledge", "duplicate_window"),
    "PROJECT_INFO_BLOCKLIST": ("guardrails", "knowledge", "blocklist"),
    "RATE_LIMIT_HOLD_MESSAGE": ("guardrails", "rate_limit_hold_message"),
    "TWILIO_ACCOUNT_SID": ("twilio", "account_sid"),
    "TWILIO_AUTH_TOKEN": ("twilio", "auth_token"),
    "META_SIP_USERNAME": ("whatsapp_sip", "username"),
    "META_SIP_PASSWORD": ("whatsapp_sip", "password"),
    "META_SIP_LISTEN_PORT": ("whatsapp_sip", "listen_port"),
    "META_SIP_MY_IP": ("whatsapp_sip", "my_ip"),
    "META_SIP_SERVER": ("whatsapp_sip", "meta_sip_server"),
    "PILOT_MONITORING_ENABLED": ("monitoring", "enabled"),
    "CALL_LEDGER_SQLITE_PATH": ("monitoring", "sqlite_path"),
    "TRANSCRIPT_HISTORY_ENABLED": ("monitoring", "transcript_history_enabled"),
    "TRANSCRIPT_HISTORY_DIR": ("monitoring", "transcript_history_dir"),
}


class ConfigNode(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            super().__setitem__(item, value)
        return value

    __setattr__ = dict.__setitem__  # type: ignore[assignment]

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            super().__setitem__(key, value)
        return value

    def copy(self) -> "ConfigNode":
        return ConfigNode({k: (v.copy() if isinstance(v, ConfigNode) else v) for k, v in self.items()})


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def _get_nested(data: Dict[str, Any], path: Iterable[str]) -> Any:
    node: Any = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            raise KeyError("Missing configuration key: " + ".".join(path))
        node = node[part]
    return node


def _set_nested(data: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    parts = list(path)
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _coerce_env_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(current, tuple):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    return raw


def _ensure_required_keys(data: Dict[str, Any]) -> None:
    missing = []
    for path in REQUIRED_PATHS:
        try:
            _get_nested(data, path)
        except KeyError:
            missing.append(".".join(path))
    if missing:
        raise KeyError(f"Missing configuration keys: {', '.join(missing)}")


def _apply_env_overrides(data: Dict[str, Any]) -> None:
    for env_name, path in ENVIRONMENT_OVERRIDES.items():
        if env_name in os.environ:
            try:
                current = _get_nested(data, path)
            except KeyError:
                continue
            overridden = _coerce_env_value(os.environ[env_name], current)
            _set_nested(data, path, overridden)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _augment_configuration(data: Dict[str, Any]) -> None:
    prompt_path_str = _get_nested(data, ("retrieval", "session_system_prompt_path"))
    prompt_path = _resolve_path(prompt_path_str)
    data.setdefault("retrieval", {})["session_system_prompt_path"] = str(prompt_path)
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        prompt_text = ""
    data["retrieval"]["session_system_prompt"] = prompt_text


def _to_confignode(obj: Any) -> Any:
    if isinstance(obj, dict) and not isinstance(obj, ConfigNode):
        return ConfigNode({k: _to_confignode(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_confignode(v) for v in obj]
    return obj


@lru_cache(maxsize=1)
def _load_from_source(path: Path) -> ConfigNode:
    data = _load_yaml(path)
    _ensure_required_keys(data)
    _apply_env_overrides(data)
    _augment_configuration(data)
    return _to_confignode(data)


def load_settings(path: str | Path | None = None, *, force: bool = False) -> ConfigNode:
    config_path = Path(path or os.getenv(CONFIG_FILE_ENV) or DEFAULT_CONFIG_PATH)
    if force:
        _load_from_source.cache_clear()
    return _load_from_source(config_path)


def reload_settings(path: str | Path | None = None) -> ConfigNode:
    return load_settings(path, force=True)


config = load_settings()

__all__ = ["ConfigNode", "config", "load_settings", "reload_settings"]
