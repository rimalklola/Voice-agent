"""Realtime session server bridging Twilio media streams and Google Gemini Live."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
import requests

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    genai = None  # type: ignore
    gtypes = None  # type: ignore

try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = None  # type: ignore

try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.websockets import WebSocketDisconnect
except ModuleNotFoundError:
    FastAPI = None  # type: ignore
    WebSocket = object  # type: ignore
    Request = object  # type: ignore
    HTMLResponse = None  # type: ignore
    JSONResponse = None  # type: ignore
    PlainTextResponse = None  # type: ignore

from src.utils.config import config
from src.utils.logging import setup_logging
from src.utils.prompt import build_instructions, build_temporal_guidance
from src.utils.telemetry import get_tracer
from src.utils.tools import ask_to_reserve, get_project_info, get_property_specs, get_project_facts
from src.utils.circuit_breaker import gemini_live_breaker, CircuitBreakerOpenError
from src.utils.tool_executor import execute_tool_with_timeout, get_tool_timeout_config
from src.utils.health import perform_health_check
from src.utils.monitoring import (
    append_pilot_call_row_async,
    finalize_pilot_call_row,
    increment_counter,
    new_pilot_call_row,
    record_reservation_status,
    record_tool_call,
    set_call_identifiers,
)
from src.utils.transcript_history import (
    TranscriptSessionBuffer,
    get_transcript_record,
    list_transcript_records,
    render_transcript_text,
    save_transcript_record,
)

# -----------------------
# Config / Globals
# -----------------------
GEMINI_API_KEY = config.project.gemini_api_key
TEMPERATURE = config.realtime.temperature
VOICE = config.realtime.voice
GEMINI_MODEL = config.realtime.asr_model
REALTIME_CONNECT_ATTEMPTS = config.realtime.connect_attempts
REALTIME_CONNECT_BACKOFF = config.realtime.connect_backoff
REALTIME_CONNECT_BACKOFF_MAX = config.realtime.connect_backoff_max
REALTIME_PLAYBACK_GUARD = config.realtime.playback_guard_ms / 1000.0
TWILIO_IDLE_TIMEOUT_S = float(config.realtime.get("twilio_idle_timeout_s", 30.0))
SESSION_BUFFER_MAX_ENTRIES = int(config.realtime.get("session_buffer_max_entries", 128))
SESSION_RESPONSE_TEXT_MAX_CHARS = int(config.realtime.get("session_response_text_max_chars", 16000))
SESSION_FUNCTION_ARGS_MAX_CHARS = int(config.realtime.get("session_function_args_max_chars", 16000))
SESSION_TOOL_ATTEMPTS_MAX_ENTRIES = int(config.realtime.get("session_tool_attempt_max_entries", 32))

ELEVENLABS_API_KEY = config.elevenlabs.api_key
ELEVENLABS_VOICE_ID = config.elevenlabs.voice_id
ELEVENLABS_MODEL_ID = config.elevenlabs.model_id
ELEVENLABS_BASE_URL = config.elevenlabs.base_url
ELEVENLABS_OPTIMIZE_LATENCY = str(config.elevenlabs.optimize_latency)
ELEVENLABS_STABILITY = config.elevenlabs.stability
ELEVENLABS_SIMILARITY = config.elevenlabs.similarity

USE_ELEVENLABS_TTS = config.elevenlabs.enabled
RATE_LIMIT_HOLD_MESSAGE = config.guardrails.rate_limit_hold_message
TWILIO_ACCOUNT_SID = config.twilio.account_sid
TWILIO_AUTH_TOKEN = config.twilio.auth_token

# Audio constants
_MULAW_RATE = 8000        # Twilio sends/receives 8kHz G.711 mu-law
_GEMINI_IN_RATE = 16000   # Gemini Live expects 16kHz PCM
_GEMINI_OUT_RATE = 24000  # Gemini Live outputs 24kHz PCM
_TWILIO_FRAME_BYTES = 320  # 40ms of mu-law @ 8kHz

def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_monitoring_cfg = config.get("monitoring", {})
if not isinstance(_monitoring_cfg, dict):
    _monitoring_cfg = {}
PILOT_MONITORING_ENABLED = _parse_bool(_monitoring_cfg.get("enabled", True), True)
CALL_LEDGER_SQLITE_PATH = str(_monitoring_cfg.get("sqlite_path", "./data/monitoring/call_ledger.sqlite3"))
TRANSCRIPT_HISTORY_ENABLED = _parse_bool(_monitoring_cfg.get("transcript_history_enabled", True), True)
TRANSCRIPT_HISTORY_DIR = str(_monitoring_cfg.get("transcript_history_dir", "./data/monitoring/transcripts"))

setup_logging()
logger = logging.getLogger(__name__)

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set; realtime endpoints will reject connections until provided.")

if USE_ELEVENLABS_TTS and (not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID):
    logger.warning("ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID missing; ElevenLabs TTS disabled.")

# -----------------------
# Audio conversion (mu-law 8kHz ↔ PCM 16kHz/24kHz) using numpy
# -----------------------
_ULAW_DECODE_TABLE: Optional[np.ndarray] = None


def _get_ulaw_decode_table() -> np.ndarray:
    global _ULAW_DECODE_TABLE
    if _ULAW_DECODE_TABLE is None:
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            b = (~i) & 0xFF
            sign = b & 0x80
            exp = (b >> 4) & 0x07
            mantissa = b & 0x0F
            magnitude = ((mantissa << 3) | 0x84) << exp
            magnitude -= 0x84
            table[i] = max(-32768, min(32767, -magnitude if sign else magnitude))
        _ULAW_DECODE_TABLE = table
    return _ULAW_DECODE_TABLE


def _mulaw_b64_to_pcm16k(mulaw_b64: str) -> bytes:
    """Base64 mu-law 8kHz → raw PCM int16 bytes at 16kHz."""
    mulaw_bytes = base64.b64decode(mulaw_b64)
    table = _get_ulaw_decode_table()
    indices = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    pcm_8k = table[indices].astype(np.float32)
    n = len(pcm_8k)
    if n == 0:
        return b""
    # Linear interpolation 8kHz → 16kHz (exact 2× upsample)
    pcm_16k = np.empty(n * 2, dtype=np.float32)
    pcm_16k[0::2] = pcm_8k
    if n > 1:
        pcm_16k[1::2][:-1] = (pcm_8k[:-1] + pcm_8k[1:]) / 2.0
    pcm_16k[1::2][-1] = pcm_8k[-1]
    return pcm_16k.clip(-32768, 32767).astype(np.int16).tobytes()


def _pcm24k_to_mulaw_b64(pcm_bytes: bytes) -> str:
    """Raw PCM int16 bytes at 24kHz → base64 mu-law 8kHz."""
    if not pcm_bytes:
        return ""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    # Decimate 24kHz → 8kHz (3:1 ratio)
    n_out = n // 3
    if n_out == 0:
        return ""
    truncated = samples[: n_out * 3].reshape(n_out, 3)
    pcm_8k = truncated.mean(axis=1).clip(-32768, 32767).astype(np.int16)
    mulaw = _pcm16_to_ulaw(pcm_8k)
    return base64.b64encode(mulaw).decode("ascii")


def _pcm16_to_ulaw(samples: np.ndarray) -> bytes:
    """Encode int16 numpy array to G.711 mu-law bytes."""
    BIAS = 132
    CLIP = 32635
    s32 = samples.astype(np.int32)
    sign = np.where(s32 < 0, np.uint8(0x80), np.uint8(0)).astype(np.uint8)
    abs_s = np.clip(np.abs(s32), 0, CLIP) + BIAS
    exp = np.floor(np.log2(abs_s.clip(1))).astype(np.int32) - 3
    exp = np.clip(exp, 0, 7).astype(np.uint8)
    mantissa = ((abs_s >> (exp.astype(np.int32) + 3)) & 0x0F).astype(np.uint8)
    ulaw = (~(sign | (exp << 4) | mantissa)).astype(np.uint8)
    return ulaw.tobytes()


# -----------------------
# Twilio helpers
# -----------------------
class TwilioIdleTimeoutError(Exception):
    """Raised when Twilio stops sending media events for too long."""


def _hangup_twilio_call(call_sid: str) -> None:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and call_sid):
        logger.info("Skipping Twilio hangup; credentials or call_sid missing")
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls/{call_sid}.json"
    try:
        resp = requests.post(
            url,
            data={"Status": "completed"},
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10,
        )
        if resp.status_code >= 400:
            logger.warning("Twilio hangup request failed", extra={"status": resp.status_code})
        else:
            logger.info("Twilio call hangup requested", extra={"call_sid": call_sid})
    except Exception as exc:
        logger.exception("Error while calling Twilio hangup API", exc_info=exc)


async def _receive_twilio_message(websocket: WebSocket, timeout_s: float) -> str:  # type: ignore[valid-type]
    try:
        return await asyncio.wait_for(websocket.receive_text(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise TwilioIdleTimeoutError(f"No Twilio media event for {timeout_s:.1f}s") from exc


async def _close_twilio_websocket(websocket: WebSocket, code: int = 1000) -> None:  # type: ignore[valid-type]
    try:
        await websocket.close(code=code)
    except Exception:
        logger.debug("Twilio websocket already closed during cleanup")


async def _close_gemini_session(session_cm: Any, session: Any) -> None:
    if session is not None:
        try:
            await session.close()
        except Exception:
            logger.debug("Failed to close Gemini session cleanly", exc_info=True)
    if session_cm is not None:
        try:
            await session_cm.__aexit__(None, None, None)
        except Exception:
            logger.debug("Failed to exit Gemini session context cleanly", exc_info=True)


# -----------------------
# Session task runner
# -----------------------
async def _run_session_tasks(
    receive_from_twilio: Callable[[], Awaitable[None]],
    receive_from_gemini: Callable[[], Awaitable[None]],
) -> None:
    tasks = [
        asyncio.create_task(receive_from_twilio(), name="receive_from_twilio"),
        asyncio.create_task(receive_from_gemini(), name="receive_from_gemini"),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        first_error: Optional[BaseException] = None
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                first_error = exc
                break
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if first_error is not None:
            raise first_error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# -----------------------
# Gemini session setup
# -----------------------
_GEMINI_CLIENT: Optional[Any] = None


def _get_gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        if genai is None:
            raise RuntimeError("google-genai package is not installed. Run: pip install google-genai")
        _GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return _GEMINI_CLIENT


def _build_gemini_tools() -> List[Any]:
    if gtypes is None:
        return []

    declarations = [
        gtypes.FunctionDeclaration(
            name="ask_to_reserve",
            description=(
                "Démarrer une demande de visite/rappel/brochure en fin d'appel. "
                "Exiger la confirmation explicite des informations collectées."
            ),
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "full_name": gtypes.Schema(type="STRING", description="Nom et prénom du client."),
                    "phone": gtypes.Schema(type="STRING", description="Téléphone du client (obligatoire)."),
                    "lead_source": gtypes.Schema(type="STRING", description="Source déclarée par le client (presse, réseaux, ami, etc.)."),
                    "request_type": gtypes.Schema(type="STRING", description="Type de demande: visite, rappel ou brochure."),
                    "email": gtypes.Schema(type="STRING", description="Adresse e-mail (optionnelle, seulement si donnée spontanément)."),
                    "date": gtypes.Schema(type="STRING", description="Date souhaitée (optionnelle)."),
                    "time": gtypes.Schema(type="STRING", description="Heure souhaitée (optionnelle)."),
                    "guests": gtypes.Schema(type="INTEGER", description="Nombre de personnes (optionnel)."),
                    "notes": gtypes.Schema(type="STRING", description="Informations complémentaires (optionnel)."),
                },
                required=["full_name", "phone", "lead_source"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_project_info",
            description="Répond brièvement aux questions sur Alma Resort en s'appuyant sur la base documentaire.",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "query": gtypes.Schema(type="STRING", description="Question utilisateur."),
                    "top_k": gtypes.Schema(type="INTEGER", description="Nombre de résultats (par défaut 3)."),
                },
                required=["query"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_property_specs",
            description=(
                "Donne les surfaces, typologies, dates de livraison et attributs clés d'un produit depuis la fiche structurée. "
                "Pour lots/terrains, utiliser la catégorie villa_lots."
            ),
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "category": gtypes.Schema(
                        type="STRING",
                        description="Catégorie cible: apartments_ript, apartments_residence, villas_premium ou villa_lots.",
                    ),
                    "variant": gtypes.Schema(type="STRING", description="Typologie précise (ex: F2, Type 1, bande, jumele). Optionnel."),
                    "attributes": gtypes.Schema(
                        type="ARRAY",
                        items=gtypes.Schema(type="STRING"),
                        description="Attributs attendus (ex: interior_area_m2, delivery_timeline).",
                    ),
                    "question": gtypes.Schema(type="STRING", description="Question utilisateur initiale."),
                },
                required=["category"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_project_facts",
            description=(
                "Récupère les informations générales (localisation, promoteurs, sécurité, commercialisation, "
                "avancement et livraisons globales) depuis la fiche structurée."
            ),
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "section": gtypes.Schema(
                        type="STRING",
                        description="Rubrique ciblée: location, project_overview, legal, lifestyle ou commercial.",
                    ),
                    "topic": gtypes.Schema(type="STRING", description="Entrée précise dans la rubrique."),
                    "question": gtypes.Schema(type="STRING", description="Question utilisateur pour aider à sélectionner la bonne entrée."),
                },
            ),
        ),
        gtypes.FunctionDeclaration(
            name="end_call",
            description=(
                "Met fin à l'appel en cours. Utilise ce tool uniquement lorsque la conversation est terminée "
                "et que l'appelant n'a plus besoin d'aide."
            ),
            parameters=gtypes.Schema(type="OBJECT", properties={}),
        ),
    ]
    return [gtypes.Tool(function_declarations=declarations)]


def _build_gemini_config() -> Any:
    if gtypes is None:
        raise RuntimeError("google-genai package is not installed")

    system_prompt = (
        config.retrieval.session_system_prompt
        + "\n\n" + build_temporal_guidance()
        + "\n\nRègle supplémentaire: Réponses ≤ 2 phrases, coupe dès que l'information essentielle est dite."
        + "\nSi le prospect demande une date de livraison, utilise d'abord les outils. Pour lots, lotissements, terrains: "
        + "utilise `get_property_specs` avec `category=villa_lots` et `attributes=[delivery_timeline]`, ou "
        + "`get_project_facts` avec `section=project_overview` et `topic=progress` si la question est globale."
        + "\nN'indique pas qu'il faut vérifier avec l'équipe commerciale pour une date de livraison déjà présente dans les fiches."
        + "\nAvant d'appeler l'outil `ask_to_reserve`, reformule la demande et attends une validation explicite de l'utilisateur."
        + "\nSi l'utilisateur confirme qu'il s'agit d'une modification d'une réservation existante, passe override_duplicate=true."
        + "\nUtilise `get_property_specs` pour les surfaces, `get_project_facts` pour les informations générales structurées, "
        + "et `get_project_info` pour le reste."
        + "\n" + config.tools.get("end_call_prompt", "")
    )

    return gtypes.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=gtypes.Content(parts=[gtypes.Part(text=system_prompt)]),
        tools=_build_gemini_tools(),
        speech_config=gtypes.SpeechConfig(
            voice_config=gtypes.VoiceConfig(
                prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
    )


async def _connect_gemini_live_session():
    client = _get_gemini_client()
    live_config = _build_gemini_config()
    session_cm = client.aio.live.connect(model=GEMINI_MODEL, config=live_config)
    session = await session_cm.__aenter__()
    return session_cm, session


# -----------------------
# ElevenLabs TTS (unchanged from original)
# -----------------------
async def _stream_tts_to_twilio(
    websocket: WebSocket,  # type: ignore[valid-type]
    stream_sid: Optional[str],
    text: str,
    *,
    session_span=None,
) -> None:
    if not USE_ELEVENLABS_TTS:
        return
    if not stream_sid or not text.strip():
        return
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        logger.error("Cannot stream TTS: ElevenLabs credentials missing")
        return
    if aiohttp is None:
        logger.error("Cannot stream TTS: aiohttp is not installed")
        return

    params = {
        "output_format": "ulaw_8000",
        "optimize_streaming_latency": ELEVENLABS_OPTIMIZE_LATENCY,
    }
    url = f"{ELEVENLABS_BASE_URL.rstrip('/')}/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    payload: Dict[str, Any] = {"text": text, "model_id": ELEVENLABS_MODEL_ID}
    voice_settings: Dict[str, float] = {}
    if ELEVENLABS_STABILITY:
        try:
            voice_settings["stability"] = float(ELEVENLABS_STABILITY)
        except ValueError:
            pass
    if ELEVENLABS_SIMILARITY:
        try:
            voice_settings["similarity_boost"] = float(ELEVENLABS_SIMILARITY)
        except ValueError:
            pass
    if voice_settings:
        payload["voice_settings"] = voice_settings

    headers = {"Accept": "*/*", "xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    logger.info("Streaming response via ElevenLabs", extra={"text_preview": text[:120]})

    buffer = bytearray()
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers, params=params) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(_TWILIO_FRAME_BYTES):
                    if not chunk:
                        continue
                    buffer.extend(chunk)
                    while len(buffer) >= _TWILIO_FRAME_BYTES:
                        frame = bytes(buffer[:_TWILIO_FRAME_BYTES])
                        del buffer[:_TWILIO_FRAME_BYTES]
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(frame).decode("ascii")},
                        })
                        await asyncio.sleep(0)
        if buffer:
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(bytes(buffer)).decode("ascii")},
            })
        await websocket.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": "tts_complete"}})
    except Exception:
        logger.exception("Failed streaming ElevenLabs TTS")


# -----------------------
# FastAPI route helpers
# -----------------------
async def startup_warmup() -> None:
    try:
        from src.ingestion.embedder import get_embedder
        get_embedder(config.ingestion.embed_model_name).embed(["warmup"])
    except Exception:
        pass
    try:
        from src.retrieval.lancedb_client import connect_table
        _ = connect_table()
    except Exception:
        pass


async def health() -> Dict[str, Any]:
    health_status = await perform_health_check(
        check_openai=True,
        check_lancedb=True,
        check_config=True,
        timeout_s=5.0,
    )
    return health_status.to_dict()


async def incoming_call(request: Request):  # type: ignore[valid-type]
    if config.project.public_http_base_url:
        public_url = config.project.public_http_base_url.rstrip("/")
        if public_url.startswith("http"):
            ws_url = public_url.replace("http", "ws", 1) + "/media-stream"
        else:
            ws_url = "wss://" + public_url.lstrip("ws://").lstrip("wss://") + "/media-stream"
    else:
        host = (
            request.headers.get("x-forwarded-host")
            or request.headers.get("host")
            or request.url.hostname
            or "localhost"
        )
        proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "https").lower()
        ws_scheme = "ws" if host in {"localhost", "127.0.0.1", "0.0.0.0"} else "wss"
        ws_url = f"{ws_scheme}://{host}/media-stream"
    logger.info("Issuing Twilio stream connect to %s", ws_url)
    twiml = f'<Response><Connect><Stream url="{ws_url}"/></Connect></Response>'
    return HTMLResponse(content=twiml, media_type="application/xml")


async def list_transcripts(request: Request):  # type: ignore[valid-type]
    params = request.query_params
    try:
        limit = int(params.get("limit", "50"))
    except ValueError:
        limit = 50
    records = list_transcript_records(
        TRANSCRIPT_HISTORY_DIR,
        limit=limit,
        call_sid=params.get("call_sid"),
        stream_sid=params.get("stream_sid"),
        query=params.get("q"),
        session_status=params.get("status"),
        sqlite_path=CALL_LEDGER_SQLITE_PATH,
    )
    return JSONResponse({
        "count": len(records),
        "items": records,
        "transcript_history_enabled": TRANSCRIPT_HISTORY_ENABLED,
        "transcript_history_dir": TRANSCRIPT_HISTORY_DIR,
    })


async def get_transcript(call_id: str):
    try:
        record = get_transcript_record(TRANSCRIPT_HISTORY_DIR, call_id)
    except FileNotFoundError:
        return JSONResponse({"error": "transcript_not_found", "call_id": call_id}, status_code=404)
    except ValueError:
        return JSONResponse({"error": "invalid_call_id"}, status_code=400)
    return JSONResponse(record)


async def get_transcript_text(call_id: str):
    try:
        record = get_transcript_record(TRANSCRIPT_HISTORY_DIR, call_id)
    except FileNotFoundError:
        return PlainTextResponse(f"Transcript not found for call_id={call_id}", status_code=404)
    except ValueError:
        return PlainTextResponse("Invalid call_id", status_code=400)
    return PlainTextResponse(render_transcript_text(record), media_type="text/plain")


# -----------------------
# Main WebSocket handler
# -----------------------
async def handle_media_stream(websocket: WebSocket):  # type: ignore[valid-type]
    await websocket.accept()
    logger.info("Twilio client connected")
    session_started_at = time.perf_counter()

    tracer = get_tracer()
    span_cm = (
        tracer.start_as_current_span(
            "realtime.session",
            attributes={"realtime.voice": VOICE, "realtime.model": GEMINI_MODEL},
        )
        if tracer
        else nullcontext()
    )

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY missing; closing media stream connection")
        await websocket.close(code=1011)
        return

    with span_cm as session_span:
        attempt = 0
        backoff = REALTIME_CONNECT_BACKOFF
        session_tracer = tracer
        monitoring_row = new_pilot_call_row(voice=VOICE, model=GEMINI_MODEL, temperature=TEMPERATURE)
        transcript_buffer = TranscriptSessionBuffer(voice=VOICE, model=GEMINI_MODEL, temperature=TEMPERATURE)
        monitoring_row["call_id"] = transcript_buffer.call_id
        monitoring_status = "unknown"
        monitoring_failure_reason = ""

        try:
            while True:
                session_started = False
                gemini_session_cm = None
                gemini_session = None
                tool_attempts: Dict[str, bool] = {}

                try:
                    gemini_session_cm, gemini_session = await gemini_live_breaker.call(
                        _connect_gemini_live_session,
                    )
                    session_started = True

                    # ---- Per-connection state ----
                    stream_sid: Optional[str] = None
                    call_sid: Optional[str] = None
                    is_playback_active: bool = False
                    playback_guard_until: float = 0.0
                    twilio_closed: bool = False
                    greeting_sent: bool = False
                    end_call_requested: bool = False

                    # Accumulated assistant text for ElevenLabs / transcript
                    current_assistant_text: List[str] = []

                    # -------- Twilio → Gemini (audio in) --------
                    async def receive_from_twilio():
                        nonlocal stream_sid, call_sid, greeting_sent, is_playback_active
                        nonlocal twilio_closed, monitoring_status, monitoring_failure_reason

                        try:
                            while True:
                                message = await _receive_twilio_message(websocket, TWILIO_IDLE_TIMEOUT_S)
                                try:
                                    data = json.loads(message)
                                except json.JSONDecodeError:
                                    logger.warning("Ignoring malformed Twilio event")
                                    continue
                                etype = data.get("event")

                                if etype == "connected":
                                    logger.info("Twilio media stream connected")

                                elif etype == "start":
                                    start_info = data.get("start", {})
                                    stream_sid = start_info.get("streamSid")
                                    call_sid = start_info.get("callSid")
                                    set_call_identifiers(monitoring_row, call_sid=call_sid, stream_sid=stream_sid)
                                    transcript_buffer.set_call_identifiers(call_sid=call_sid, stream_sid=stream_sid)
                                    logger.info(
                                        "Incoming stream started",
                                        extra={"stream_sid": stream_sid, "call_sid": call_sid},
                                    )
                                    if session_span is not None:
                                        if stream_sid:
                                            session_span.set_attribute("twilio.stream_sid", stream_sid)
                                        if call_sid:
                                            session_span.set_attribute("twilio.call_sid", call_sid)
                                    if not greeting_sent:
                                        try:
                                            await gemini_session.send(
                                                input=gtypes.LiveClientContent(
                                                    turns=[
                                                        gtypes.Content(
                                                            parts=[gtypes.Part(
                                                                text="Dis simplement : 'Bonjour, Alma Resort à l'appareil. Je vous écoute.'"
                                                            )],
                                                            role="user",
                                                        )
                                                    ],
                                                    turn_complete=True,
                                                )
                                            )
                                        except Exception:
                                            logger.exception("Failed to send greeting to Gemini")
                                        greeting_sent = True

                                elif etype == "media":
                                    media_payload = str((data.get("media") or {}).get("payload") or "").strip()
                                    if not media_payload:
                                        continue
                                    try:
                                        pcm_16k = _mulaw_b64_to_pcm16k(media_payload)
                                        await gemini_session.send(
                                            input=gtypes.LiveClientRealtimeInput(
                                                media_chunks=[gtypes.Blob(
                                                    data=pcm_16k,
                                                    mime_type="audio/pcm;rate=16000",
                                                )]
                                            )
                                        )
                                    except Exception:
                                        logger.debug("Failed forwarding media frame to Gemini")

                                elif etype == "stop":
                                    logger.info("Twilio stream stopped")
                                    twilio_closed = True
                                    if monitoring_status == "unknown":
                                        monitoring_status = "completed"
                                    break

                        except TwilioIdleTimeoutError:
                            increment_counter(monitoring_row, "twilio_idle_timeouts")
                            monitoring_status = "twilio_idle_timeout"
                            monitoring_failure_reason = "twilio_idle_timeout"
                            logger.warning(
                                "Twilio idle timeout",
                                extra={"idle_timeout_s": TWILIO_IDLE_TIMEOUT_S},
                            )
                            twilio_closed = True
                            try:
                                await gemini_session.close()
                            except Exception:
                                pass
                        except WebSocketDisconnect:
                            increment_counter(monitoring_row, "twilio_disconnects")
                            if monitoring_status == "unknown":
                                monitoring_status = "twilio_disconnected"
                                monitoring_failure_reason = "twilio_websocket_disconnect"
                            logger.info("Twilio client disconnected")
                            twilio_closed = True
                            try:
                                await gemini_session.close()
                            except Exception:
                                pass
                        except Exception:
                            logger.exception("Unhandled exception in receive_from_twilio")
                            raise

                    # -------- Gemini → Twilio (audio out + tool calls) --------
                    async def receive_from_gemini():
                        nonlocal is_playback_active, playback_guard_until, twilio_closed
                        nonlocal tool_attempts, monitoring_status, monitoring_failure_reason
                        nonlocal current_assistant_text, end_call_requested

                        try:
                            async for response in gemini_session.receive():
                                if end_call_requested:
                                    break

                                # ---- Audio output ----
                                if response.data:
                                    b64 = _pcm24k_to_mulaw_b64(response.data)
                                    if b64 and stream_sid and not twilio_closed:
                                        await websocket.send_json({
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": b64},
                                        })
                                        is_playback_active = True
                                        playback_guard_until = time.monotonic() + REALTIME_PLAYBACK_GUARD

                                # ---- Server content (turn events + text) ----
                                if response.server_content:
                                    sc = response.server_content

                                    # Collect model text for transcript / ElevenLabs
                                    if sc.model_turn:
                                        for part in sc.model_turn.parts:
                                            if part.text:
                                                current_assistant_text.append(part.text)

                                    if sc.interrupted:
                                        # User barged in — clear Twilio audio buffer
                                        is_playback_active = False
                                        playback_guard_until = 0.0
                                        current_assistant_text.clear()
                                        if stream_sid and not twilio_closed:
                                            try:
                                                await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                                            except Exception:
                                                pass
                                        logger.debug("Gemini response interrupted by user barge-in")

                                    if sc.turn_complete:
                                        increment_counter(monitoring_row, "assistant_responses")
                                        is_playback_active = False
                                        playback_guard_until = time.monotonic() + REALTIME_PLAYBACK_GUARD

                                        aggregated = "".join(current_assistant_text).strip()
                                        current_assistant_text.clear()

                                        if aggregated:
                                            transcript_buffer.add_turn(
                                                speaker="assistant",
                                                text=aggregated,
                                                event_type="turn_complete",
                                            )
                                            if USE_ELEVENLABS_TTS and stream_sid and not twilio_closed:
                                                await _stream_tts_to_twilio(
                                                    websocket,
                                                    stream_sid,
                                                    aggregated,
                                                    session_span=session_span,
                                                )

                                        if session_span is not None:
                                            session_span.add_event("gemini.turn_complete")

                                # ---- Tool calls ----
                                if response.tool_call:
                                    for fc in response.tool_call.function_calls:
                                        name = fc.name
                                        call_id = fc.id
                                        args = dict(fc.args or {})
                                        record_tool_call(monitoring_row, name)

                                        logger.info(
                                            "Tool call received",
                                            extra={"tool_name": name, "call_id": call_id, "args": args},
                                        )
                                        if session_span is not None:
                                            session_span.add_event(
                                                "tool.completed",
                                                attributes={"tool_name": name or "unknown"},
                                            )

                                        if name == "ask_to_reserve":
                                            ask_args = dict(args)
                                            ask_args.setdefault(
                                                "call_id",
                                                monitoring_row.get("call_id") or transcript_buffer.call_id,
                                            )
                                            monitoring_row["caller_wants_reservation"] = 1
                                            requested_type = str(ask_args.get("request_type") or "").strip()
                                            if requested_type:
                                                monitoring_row["reservation_request_type"] = requested_type
                                            tool_cm = (
                                                session_tracer.start_as_current_span(
                                                    "tool.ask_to_reserve",
                                                    attributes={"tool.name": "ask_to_reserve"},
                                                )
                                                if session_tracer
                                                else nullcontext()
                                            )
                                            with tool_cm as tool_span:
                                                try:
                                                    timeout = get_tool_timeout_config("ask_to_reserve")
                                                    result = await execute_tool_with_timeout(
                                                        ask_to_reserve,
                                                        "ask_to_reserve",
                                                        ask_args,
                                                        timeout_s=timeout,
                                                        fallback_error_message="Désolée, un problème est survenu pendant la réservation. Réessayons.",
                                                    )
                                                except Exception as exc:
                                                    if tool_span is not None:
                                                        tool_span.record_exception(exc)
                                                    result = {
                                                        "status": "error",
                                                        "answer": "Désolée, un problème est survenu pendant la réservation. Réessayons.",
                                                    }
                                            record_reservation_status(monitoring_row, result.get("status"))
                                            await gemini_session.send(
                                                input=gtypes.LiveClientToolResponse(
                                                    function_responses=[gtypes.FunctionResponse(
                                                        id=call_id,
                                                        name=name,
                                                        response={"result": result},
                                                    )]
                                                )
                                            )

                                        elif name == "get_project_info":
                                            user_query = args.get("query", "") or ""
                                            top_k_arg = args.get("top_k")
                                            try:
                                                top_k = int(top_k_arg) if top_k_arg is not None else config.retrieval.top_k
                                            except (TypeError, ValueError):
                                                top_k = config.retrieval.top_k
                                            top_k = max(1, min(top_k, 3))
                                            tool_cm = (
                                                session_tracer.start_as_current_span(
                                                    "tool.get_project_info",
                                                    attributes={"tool.name": "get_project_info", "tool.top_k": top_k},
                                                )
                                                if session_tracer
                                                else nullcontext()
                                            )
                                            with tool_cm as tool_span:
                                                try:
                                                    timeout = get_tool_timeout_config("get_project_info")
                                                    payload = await execute_tool_with_timeout(
                                                        lambda a: get_project_info(user_query, top_k),
                                                        "get_project_info",
                                                        {},
                                                        timeout_s=timeout,
                                                        fallback_error_message="La recherche a pris trop de temps. Veuillez reformuler.",
                                                    )
                                                except Exception as exc:
                                                    if tool_span is not None:
                                                        tool_span.record_exception(exc)
                                                    payload = {"query": user_query, "snippets": []}
                                            tool_attempts["get_project_info"] = True
                                            await gemini_session.send(
                                                input=gtypes.LiveClientToolResponse(
                                                    function_responses=[gtypes.FunctionResponse(
                                                        id=call_id,
                                                        name=name,
                                                        response={"result": payload},
                                                    )]
                                                )
                                            )

                                        elif name == "get_property_specs":
                                            category = args.get("category", "") or ""
                                            variant = args.get("variant")
                                            question = args.get("question")
                                            raw_attrs = args.get("attributes")
                                            attr_list = None
                                            if isinstance(raw_attrs, list):
                                                attr_list = [str(a) for a in raw_attrs if isinstance(a, str)] or None
                                            tool_cm = (
                                                session_tracer.start_as_current_span(
                                                    "tool.get_property_specs",
                                                    attributes={"tool.name": "get_property_specs"},
                                                )
                                                if session_tracer
                                                else nullcontext()
                                            )
                                            with tool_cm as tool_span:
                                                try:
                                                    timeout = get_tool_timeout_config("get_property_specs")
                                                    payload = await execute_tool_with_timeout(
                                                        lambda a: get_property_specs(category, variant, attr_list, question),
                                                        "get_property_specs",
                                                        {},
                                                        timeout_s=timeout,
                                                        fallback_error_message="Je n'ai pas retrouvé cette fiche produit.",
                                                    )
                                                except Exception as exc:
                                                    if tool_span is not None:
                                                        tool_span.record_exception(exc)
                                                    payload = {"status": "error", "answer": "Je n'ai pas retrouvé cette fiche produit."}
                                            tool_attempts["get_property_specs"] = True
                                            await gemini_session.send(
                                                input=gtypes.LiveClientToolResponse(
                                                    function_responses=[gtypes.FunctionResponse(
                                                        id=call_id,
                                                        name=name,
                                                        response={"result": payload},
                                                    )]
                                                )
                                            )

                                        elif name == "get_project_facts":
                                            section = args.get("section")
                                            topic = args.get("topic")
                                            question = args.get("question")
                                            tool_cm = (
                                                session_tracer.start_as_current_span(
                                                    "tool.get_project_facts",
                                                    attributes={"tool.name": "get_project_facts"},
                                                )
                                                if session_tracer
                                                else nullcontext()
                                            )
                                            with tool_cm as tool_span:
                                                try:
                                                    timeout = get_tool_timeout_config("get_project_facts")
                                                    payload = await execute_tool_with_timeout(
                                                        lambda a: get_project_facts(section, topic, question),
                                                        "get_project_facts",
                                                        {},
                                                        timeout_s=timeout,
                                                        fallback_error_message="Je ne retrouve pas cette information.",
                                                    )
                                                except Exception as exc:
                                                    if tool_span is not None:
                                                        tool_span.record_exception(exc)
                                                    payload = {"status": "error", "answer": "Je ne retrouve pas cette information."}
                                            tool_attempts["get_project_facts"] = True
                                            await gemini_session.send(
                                                input=gtypes.LiveClientToolResponse(
                                                    function_responses=[gtypes.FunctionResponse(
                                                        id=call_id,
                                                        name=name,
                                                        response={"result": payload},
                                                    )]
                                                )
                                            )

                                        elif name == "end_call":
                                            logger.info("Agent requested end_call")
                                            if monitoring_status == "unknown":
                                                monitoring_status = "ended_by_agent"
                                            if session_span is not None:
                                                session_span.add_event("tool.end_call")
                                            if stream_sid and not twilio_closed:
                                                try:
                                                    await websocket.send_json({
                                                        "event": "mark",
                                                        "streamSid": stream_sid,
                                                        "mark": {"name": "call_end"},
                                                    })
                                                except Exception:
                                                    pass
                                                twilio_closed = True
                                            if call_sid:
                                                await asyncio.to_thread(_hangup_twilio_call, call_sid)
                                            try:
                                                await gemini_session.send(
                                                    input=gtypes.LiveClientToolResponse(
                                                        function_responses=[gtypes.FunctionResponse(
                                                            id=call_id,
                                                            name=name,
                                                            response={"result": {"status": "ok"}},
                                                        )]
                                                    )
                                                )
                                            except Exception:
                                                pass
                                            end_call_requested = True
                                            return

                        except WebSocketDisconnect:
                            increment_counter(monitoring_row, "twilio_disconnects")
                            if monitoring_status == "unknown":
                                monitoring_status = "twilio_disconnected"
                                monitoring_failure_reason = "twilio_websocket_disconnect"
                            twilio_closed = True
                            logger.info("Twilio socket closed; stopping Gemini receive loop")
                            return
                        except RuntimeError:
                            increment_counter(monitoring_row, "twilio_disconnects")
                            if monitoring_status == "unknown":
                                monitoring_status = "twilio_runtime_closed"
                                monitoring_failure_reason = "twilio_runtime_closed"
                            twilio_closed = True
                            return
                        except Exception:
                            logger.exception("Unhandled exception in receive_from_gemini")
                            raise

                    await _run_session_tasks(receive_from_twilio, receive_from_gemini)
                    duration_ms = int((time.perf_counter() - session_started_at) * 1000)
                    if monitoring_status == "unknown":
                        monitoring_status = "completed"
                    if session_span is not None:
                        session_span.set_attribute("realtime.retry_attempts", attempt)
                        session_span.set_attribute("realtime.session_duration_ms", duration_ms)
                    logger.info("Gemini Live session finished", extra={"duration_ms": duration_ms})
                    break

                except CircuitBreakerOpenError as exc:
                    monitoring_status = "connect_rejected"
                    monitoring_failure_reason = "circuit_breaker_open"
                    if session_span is not None:
                        session_span.record_exception(exc)
                    logger.error("Gemini Live circuit breaker is open; rejecting session")
                    await _close_twilio_websocket(websocket, code=1013)
                    return

                except Exception as exc:
                    if session_started:
                        monitoring_status = "aborted"
                        monitoring_failure_reason = f"session_exception:{exc.__class__.__name__}"
                        duration_ms = int((time.perf_counter() - session_started_at) * 1000)
                        if session_span is not None:
                            session_span.record_exception(exc)
                        logger.exception(
                            "Gemini Live session aborted",
                            extra={"duration_ms": duration_ms},
                        )
                        await websocket.close(code=1011)
                        return

                    attempt += 1
                    delay = min(backoff, REALTIME_CONNECT_BACKOFF_MAX)
                    if attempt >= REALTIME_CONNECT_ATTEMPTS:
                        monitoring_status = "connect_failed"
                        monitoring_failure_reason = f"connect_failed:{exc.__class__.__name__}"
                        if session_span is not None:
                            session_span.record_exception(exc)
                        logger.exception("Failed to establish Gemini Live session after retries")
                        await websocket.close(code=1011)
                        return
                    logger.warning(
                        "Gemini Live connect attempt failed; retrying",
                        extra={"attempt": attempt, "delay": delay, "error": str(exc)},
                    )
                    await asyncio.sleep(delay)
                    backoff = min(backoff * 2, REALTIME_CONNECT_BACKOFF_MAX)
                finally:
                    tool_attempts.clear()
                    await _close_gemini_session(gemini_session_cm, gemini_session)

        finally:
            if TRANSCRIPT_HISTORY_ENABLED:
                duration_ms = int((time.perf_counter() - session_started_at) * 1000)
                transcript_record = transcript_buffer.to_record(
                    session_status=monitoring_status,
                    failure_reason=monitoring_failure_reason,
                    duration_ms=duration_ms,
                )
                try:
                    transcript_path = await asyncio.to_thread(
                        save_transcript_record,
                        TRANSCRIPT_HISTORY_DIR,
                        transcript_record,
                        sqlite_path=CALL_LEDGER_SQLITE_PATH,
                    )
                    monitoring_row["transcript_path"] = str(transcript_path)
                    monitoring_row["transcript_preview"] = str(
                        transcript_record.get("transcript_full_text") or ""
                    )[:200].strip()
                    monitoring_row["turn_count"] = int(transcript_record.get("turn_count") or 0)
                    monitoring_row["customer_turn_count"] = int(transcript_record.get("customer_turn_count") or 0)
                    monitoring_row["assistant_turn_count"] = int(transcript_record.get("assistant_turn_count") or 0)
                except Exception:
                    logger.exception("Failed writing transcript history record")

            if PILOT_MONITORING_ENABLED:
                duration_ms = int((time.perf_counter() - session_started_at) * 1000)
                finalized_row = finalize_pilot_call_row(
                    monitoring_row,
                    session_status=monitoring_status,
                    failure_reason=monitoring_failure_reason,
                    retry_attempts=attempt,
                    duration_ms=duration_ms,
                )
                try:
                    await append_pilot_call_row_async(CALL_LEDGER_SQLITE_PATH, finalized_row)
                except Exception:
                    logger.exception("Failed writing call ledger row")


# -----------------------
# FastAPI app factory
# -----------------------
def create_app() -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("FastAPI is required to create the realtime app.")
    if TWILIO_IDLE_TIMEOUT_S <= 0:
        raise RuntimeError("realtime.twilio_idle_timeout_s must be > 0")
    if SESSION_BUFFER_MAX_ENTRIES <= 0:
        raise RuntimeError("realtime.session_buffer_max_entries must be > 0")

    @asynccontextmanager
    async def lifespan(_: "FastAPI"):
        await startup_warmup()
        yield

    app = FastAPI(title="Twilio ↔ Gemini Live (Local RAG)", lifespan=lifespan)
    app.get("/", response_class=JSONResponse)(health)
    app.get("/transcripts", response_class=JSONResponse)(list_transcripts)
    app.get("/transcripts/{call_id}", response_class=JSONResponse)(get_transcript)
    app.get("/transcripts/{call_id}/text", response_class=PlainTextResponse)(get_transcript_text)
    app.api_route("/incoming-call", methods=["GET", "POST"])(incoming_call)
    app.websocket("/media-stream")(handle_media_stream)
    return app


app = create_app() if FastAPI else None
