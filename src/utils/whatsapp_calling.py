"""
WhatsApp Business Calling API → Gemini Live bridge via SIP/RTP.

Call flow:
  1. User calls your WhatsApp Business number.
  2. Meta/WhatsApp sends a TLS SIP INVITE to your configured SIP endpoint.
  3. pyVoIP answers the call and hands RTP G.711 mu-law audio to this module.
  4. Audio is resampled 8kHz → 16kHz and streamed to Gemini Live.
  5. Gemini's 24kHz PCM response is resampled to 8kHz mu-law and sent back via RTP.

TLS requirement:
  Meta requires TLS SIP (port 5061). Use nginx or HAProxy to terminate TLS and
  forward plain SIP to whatsapp_sip.listen_port (default 5060).

  Example nginx stream block:
    stream {
      server {
        listen 5061 ssl;
        ssl_certificate /etc/ssl/certs/your.crt;
        ssl_certificate_key /etc/ssl/private/your.key;
        proxy_pass 127.0.0.1:5060;
      }
    }

Configuration (config/settings.yaml > whatsapp_sip):
  username    - Your WhatsApp phone_number_id (META_SIP_USERNAME env var)
  password    - SIP password from Meta API   (META_SIP_PASSWORD env var)
  listen_port - Port pyVoIP listens on       (default 5060)
  my_ip       - Your server's public IP      (META_SIP_MY_IP env var)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from pyVoIP.VoIP import VoIPPhone, CallState
    HAS_PYVOIP = True
except ImportError:
    HAS_PYVOIP = False
    VoIPPhone = None  # type: ignore
    CallState = None  # type: ignore

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    genai = None  # type: ignore
    gtypes = None  # type: ignore

from src.utils.config import config
from src.utils.logging import setup_logging
from src.utils.prompt import build_temporal_guidance
from src.utils.tools import ask_to_reserve, get_project_info, get_property_specs, get_project_facts
from src.utils.tool_executor import execute_tool_with_timeout, get_tool_timeout_config
from src.utils.monitoring import (
    finalize_pilot_call_row,
    increment_counter,
    new_pilot_call_row,
    record_reservation_status,
    record_tool_call,
)
from src.utils.transcript_history import (
    TranscriptSessionBuffer,
    save_transcript_record,
)

setup_logging()
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
GEMINI_API_KEY = config.project.gemini_api_key
GEMINI_MODEL = config.realtime.asr_model
VOICE = config.realtime.voice
TEMPERATURE = config.realtime.temperature

_wa_cfg = config.get("whatsapp_sip", {}) or {}
META_SIP_USERNAME: str = str(_wa_cfg.get("username", ""))
META_SIP_PASSWORD: str = str(_wa_cfg.get("password", ""))
SIP_LISTEN_PORT: int = int(_wa_cfg.get("listen_port", 5060))
SIP_MY_IP: str = str(_wa_cfg.get("my_ip", ""))
SIP_META_SERVER: str = str(_wa_cfg.get("meta_sip_server", ""))

_mon_cfg = config.get("monitoring", {}) or {}
CALL_LEDGER_SQLITE_PATH = str(_mon_cfg.get("sqlite_path", "./data/monitoring/call_ledger.sqlite3"))
TRANSCRIPT_HISTORY_DIR = str(_mon_cfg.get("transcript_history_dir", "./data/monitoring/transcripts"))
TRANSCRIPT_HISTORY_ENABLED = bool(_mon_cfg.get("transcript_history_enabled", True))

# -----------------------
# Audio conversion  (G.711 mu-law 8kHz ↔ PCM 16kHz/24kHz)
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


def _mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """G.711 mu-law raw bytes at 8kHz → PCM int16 bytes at 16kHz."""
    table = _get_ulaw_decode_table()
    indices = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    pcm_8k = table[indices].astype(np.float32)
    n = len(pcm_8k)
    if n == 0:
        return b""
    # 2× linear interpolation upsample
    pcm_16k = np.empty(n * 2, dtype=np.float32)
    pcm_16k[0::2] = pcm_8k
    if n > 1:
        pcm_16k[1::2][:-1] = (pcm_8k[:-1] + pcm_8k[1:]) / 2.0
    pcm_16k[1::2][-1] = pcm_8k[-1]
    return pcm_16k.clip(-32768, 32767).astype(np.int16).tobytes()


def _pcm16_to_ulaw(samples: np.ndarray) -> bytes:
    """Encode int16 numpy array → G.711 mu-law bytes."""
    BIAS, CLIP = 132, 32635
    s32 = samples.astype(np.int32)
    sign = np.where(s32 < 0, np.uint8(0x80), np.uint8(0)).astype(np.uint8)
    abs_s = np.clip(np.abs(s32), 0, CLIP) + BIAS
    exp = np.floor(np.log2(abs_s.clip(1))).astype(np.int32) - 3
    exp = np.clip(exp, 0, 7).astype(np.uint8)
    mantissa = ((abs_s >> (exp.astype(np.int32) + 3)) & 0x0F).astype(np.uint8)
    ulaw = (~(sign | (exp << 4) | mantissa)).astype(np.uint8)
    return ulaw.tobytes()


def _pcm24k_to_mulaw(pcm_bytes: bytes) -> bytes:
    """PCM int16 bytes at 24kHz → G.711 mu-law bytes at 8kHz (3:1 decimation)."""
    if not pcm_bytes:
        return b""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    n_out = n // 3
    if n_out == 0:
        return b""
    pcm_8k = samples[: n_out * 3].reshape(n_out, 3).mean(axis=1).clip(-32768, 32767).astype(np.int16)
    return _pcm16_to_ulaw(pcm_8k)


# -----------------------
# Gemini helpers
# -----------------------
_GEMINI_CLIENT: Optional[Any] = None


def _get_gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        if genai is None:
            raise RuntimeError("google-genai is not installed. Run: pip install google-genai")
        _GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return _GEMINI_CLIENT


def _build_gemini_tools() -> List[Any]:
    if gtypes is None:
        return []
    declarations = [
        gtypes.FunctionDeclaration(
            name="ask_to_reserve",
            description="Démarrer une demande de visite/rappel/brochure en fin d'appel. Exiger la confirmation explicite.",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "full_name": gtypes.Schema(type="STRING", description="Nom et prénom du client."),
                    "phone": gtypes.Schema(type="STRING", description="Téléphone (obligatoire)."),
                    "lead_source": gtypes.Schema(type="STRING", description="Source déclarée par le client."),
                    "request_type": gtypes.Schema(type="STRING", description="visite, rappel ou brochure."),
                    "email": gtypes.Schema(type="STRING", description="E-mail (optionnel)."),
                    "date": gtypes.Schema(type="STRING", description="Date souhaitée (optionnel)."),
                    "time": gtypes.Schema(type="STRING", description="Heure souhaitée (optionnel)."),
                    "guests": gtypes.Schema(type="INTEGER", description="Nombre de personnes (optionnel)."),
                    "notes": gtypes.Schema(type="STRING", description="Informations complémentaires (optionnel)."),
                },
                required=["full_name", "phone", "lead_source"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_project_info",
            description="Répond aux questions sur Alma Resort via la base documentaire.",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "query": gtypes.Schema(type="STRING", description="Question utilisateur."),
                    "top_k": gtypes.Schema(type="INTEGER", description="Nombre de résultats (défaut 3)."),
                },
                required=["query"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_property_specs",
            description="Surfaces, typologies, dates de livraison depuis la fiche structurée.",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "category": gtypes.Schema(type="STRING", description="apartments_ript, apartments_residence, villas_premium ou villa_lots."),
                    "variant": gtypes.Schema(type="STRING", description="Typologie précise (optionnel)."),
                    "attributes": gtypes.Schema(type="ARRAY", items=gtypes.Schema(type="STRING"), description="Attributs attendus."),
                    "question": gtypes.Schema(type="STRING", description="Question utilisateur."),
                },
                required=["category"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_project_facts",
            description="Informations générales structurées (localisation, promoteurs, avancement).",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "section": gtypes.Schema(type="STRING", description="location, project_overview, legal, lifestyle ou commercial."),
                    "topic": gtypes.Schema(type="STRING", description="Entrée précise dans la rubrique."),
                    "question": gtypes.Schema(type="STRING", description="Question utilisateur."),
                },
            ),
        ),
        gtypes.FunctionDeclaration(
            name="end_call",
            description="Met fin à l'appel. Utilise uniquement quand la conversation est terminée.",
            parameters=gtypes.Schema(type="OBJECT", properties={}),
        ),
    ]
    return [gtypes.Tool(function_declarations=declarations)]


def _build_gemini_live_config() -> Any:
    if gtypes is None:
        raise RuntimeError("google-genai is not installed")
    system_prompt = (
        config.retrieval.session_system_prompt
        + "\n\n" + build_temporal_guidance()
        + "\n\nRègle supplémentaire: Réponses ≤ 2 phrases."
        + "\nUtilise `get_property_specs` pour les surfaces, `get_project_facts` pour les informations générales structurées, et `get_project_info` pour le reste."
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


# -----------------------
# Tool execution
# -----------------------
async def _run_tool(fc: Any, monitoring_row: Dict[str, Any]) -> Any:
    """Dispatch a Gemini function call and return the result."""
    name = fc.name
    args: Dict[str, Any] = dict(fc.args or {})
    record_tool_call(monitoring_row, name)

    if name == "ask_to_reserve":
        args.setdefault("call_id", monitoring_row.get("call_id", ""))
        monitoring_row["caller_wants_reservation"] = 1
        rt = str(args.get("request_type") or "").strip()
        if rt:
            monitoring_row["reservation_request_type"] = rt
        try:
            result = await execute_tool_with_timeout(
                ask_to_reserve, "ask_to_reserve", args,
                timeout_s=get_tool_timeout_config("ask_to_reserve"),
                fallback_error_message="Désolée, un problème est survenu pendant la réservation.",
            )
        except Exception:
            result = {"status": "error", "answer": "Désolée, un problème est survenu pendant la réservation."}
        record_reservation_status(monitoring_row, result.get("status"))
        return result

    if name == "get_project_info":
        query = args.get("query", "") or ""
        top_k = max(1, min(int(args.get("top_k") or config.retrieval.top_k), 3))
        try:
            return await execute_tool_with_timeout(
                lambda a: get_project_info(query, top_k), "get_project_info", {},
                timeout_s=get_tool_timeout_config("get_project_info"),
                fallback_error_message="La recherche a pris trop de temps.",
            )
        except Exception:
            return {"query": query, "snippets": []}

    if name == "get_property_specs":
        cat = args.get("category", "") or ""
        variant = args.get("variant")
        question = args.get("question")
        raw_attrs = args.get("attributes")
        attr_list = [str(a) for a in raw_attrs if isinstance(a, str)] if isinstance(raw_attrs, list) else None
        try:
            return await execute_tool_with_timeout(
                lambda a: get_property_specs(cat, variant, attr_list, question),
                "get_property_specs", {},
                timeout_s=get_tool_timeout_config("get_property_specs"),
                fallback_error_message="Je n'ai pas retrouvé cette fiche produit.",
            )
        except Exception:
            return {"status": "error", "answer": "Je n'ai pas retrouvé cette fiche produit."}

    if name == "get_project_facts":
        section = args.get("section")
        topic = args.get("topic")
        question = args.get("question")
        try:
            return await execute_tool_with_timeout(
                lambda a: get_project_facts(section, topic, question),
                "get_project_facts", {},
                timeout_s=get_tool_timeout_config("get_project_facts"),
                fallback_error_message="Je ne retrouve pas cette information.",
            )
        except Exception:
            return {"status": "error", "answer": "Je ne retrouve pas cette information."}

    return {"status": "error", "message": f"Unknown tool: {name}"}


# -----------------------
# Core async call handler
# -----------------------
async def handle_whatsapp_call(call: Any) -> None:
    """
    Bridge one WhatsApp SIP call to Gemini Live.

    This coroutine is started via asyncio.run() inside the pyVoIP callback thread,
    so it gets its own event loop fully isolated from the FastAPI/uvicorn loop.

    Audio path (round-trip):
      pyVoIP RTP (G.711 mu-law 8kHz)
        → _mulaw_to_pcm16k → Gemini Live (PCM 16kHz input)
        → Gemini Live audio output (PCM 24kHz)
        → _pcm24k_to_mulaw → pyVoIP RTP (G.711 mu-law 8kHz)
    """
    t0 = time.perf_counter()
    monitoring_row = new_pilot_call_row(voice=VOICE, model=GEMINI_MODEL, temperature=TEMPERATURE)
    transcript_buffer = TranscriptSessionBuffer(voice=VOICE, model=GEMINI_MODEL, temperature=TEMPERATURE)
    monitoring_row["call_id"] = transcript_buffer.call_id
    monitoring_status = "unknown"
    monitoring_failure_reason = ""

    loop = asyncio.get_running_loop()

    # Thread-safe bridge: RTP reader thread → asyncio queue
    rtp_in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    stop_evt = threading.Event()

    def _rtp_reader() -> None:
        """Thread: pull G.711 frames from pyVoIP, push to async queue."""
        RTP_CHUNK = 320  # 40ms × 8000 Hz × 1 byte/sample (mu-law)
        while not stop_evt.is_set() and call.state == CallState.ANSWERED:
            try:
                audio = call.readAudio(RTP_CHUNK)
                if audio:
                    asyncio.run_coroutine_threadsafe(rtp_in.put(audio), loop)
            except Exception:
                break

    reader_thread = threading.Thread(target=_rtp_reader, daemon=True, name="rtp-reader")
    reader_thread.start()

    try:
        client = _get_gemini_client()
        live_cfg = _build_gemini_live_config()

        async with client.aio.live.connect(model=GEMINI_MODEL, config=live_cfg) as session:
            # Trigger greeting
            await session.send(
                input=gtypes.LiveClientContent(
                    turns=[gtypes.Content(
                        parts=[gtypes.Part(
                            text="Dis simplement : 'Bonjour, Alma Resort à l'appareil. Je vous écoute.'"
                        )],
                        role="user",
                    )],
                    turn_complete=True,
                )
            )

            end_requested = False

            async def _forward_rtp_to_gemini() -> None:
                """Drain the inbound queue and stream audio to Gemini."""
                while not end_requested and call.state == CallState.ANSWERED:
                    try:
                        mulaw = await asyncio.wait_for(rtp_in.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    pcm_16k = _mulaw_to_pcm16k(mulaw)
                    try:
                        await session.send(
                            input=gtypes.LiveClientRealtimeInput(
                                media_chunks=[gtypes.Blob(data=pcm_16k, mime_type="audio/pcm;rate=16000")]
                            )
                        )
                    except Exception:
                        logger.debug("Failed sending audio chunk to Gemini")

            async def _forward_gemini_to_rtp() -> None:
                """Receive from Gemini and write audio + handle tool calls."""
                nonlocal end_requested, monitoring_status, monitoring_failure_reason
                assistant_chunks: List[str] = []

                async for response in session.receive():
                    if end_requested:
                        break

                    # ── Audio → RTP ──
                    if response.data:
                        mulaw_out = _pcm24k_to_mulaw(response.data)
                        if mulaw_out and call.state == CallState.ANSWERED:
                            # writeAudio is synchronous — off-load to thread pool
                            await asyncio.to_thread(call.writeAudio, mulaw_out)

                    # ── Server content ──
                    if response.server_content:
                        sc = response.server_content
                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.text:
                                    assistant_chunks.append(part.text)
                        if sc.interrupted:
                            assistant_chunks.clear()
                        if sc.turn_complete:
                            increment_counter(monitoring_row, "assistant_responses")
                            aggregated = "".join(assistant_chunks).strip()
                            assistant_chunks.clear()
                            if aggregated:
                                transcript_buffer.add_turn(
                                    speaker="assistant",
                                    text=aggregated,
                                    event_type="turn_complete",
                                )

                    # ── Tool calls ──
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            if fc.name == "end_call":
                                record_tool_call(monitoring_row, "end_call")
                                logger.info("Agent requested end_call (WhatsApp)")
                                if monitoring_status == "unknown":
                                    monitoring_status = "ended_by_agent"
                                # Acknowledge the tool call before hanging up
                                try:
                                    await session.send(
                                        input=gtypes.LiveClientToolResponse(
                                            function_responses=[gtypes.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"result": {"status": "ok"}},
                                            )]
                                        )
                                    )
                                except Exception:
                                    pass
                                end_requested = True
                                return

                            result = await _run_tool(fc, monitoring_row)
                            await session.send(
                                input=gtypes.LiveClientToolResponse(
                                    function_responses=[gtypes.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"result": result},
                                    )]
                                )
                            )

            await asyncio.gather(_forward_rtp_to_gemini(), _forward_gemini_to_rtp())

        if monitoring_status == "unknown":
            monitoring_status = "completed"

    except Exception as exc:
        logger.exception("WhatsApp Gemini session error")
        monitoring_status = "aborted"
        monitoring_failure_reason = exc.__class__.__name__
    finally:
        stop_evt.set()
        reader_thread.join(timeout=2)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "WhatsApp call finished",
            extra={"status": monitoring_status, "duration_ms": duration_ms},
        )

        if TRANSCRIPT_HISTORY_ENABLED:
            record = transcript_buffer.to_record(
                session_status=monitoring_status,
                failure_reason=monitoring_failure_reason,
                duration_ms=duration_ms,
            )
            try:
                save_transcript_record(
                    TRANSCRIPT_HISTORY_DIR, record, sqlite_path=CALL_LEDGER_SQLITE_PATH
                )
            except Exception:
                logger.exception("Failed saving transcript")


# -----------------------
# SIP server
# -----------------------
class WhatsAppCallServer:
    """
    SIP server that listens for WhatsApp Business Calling API calls via pyVoIP.

    Meta delivers calls as SIP INVITE over TLS. Run nginx/HAProxy in front to
    terminate TLS, then forward plain SIP traffic to listen_port (default 5060).
    """

    def __init__(self) -> None:
        self._phone: Optional[Any] = None

    def _on_call(self, call: Any) -> None:
        """pyVoIP calls this in a new thread for every incoming INVITE."""
        call_id = getattr(call, "call_id", "unknown")
        logger.info("Incoming WhatsApp call", extra={"call_id": call_id})
        try:
            call.answer()
            # asyncio.run() creates a dedicated event loop for this call
            asyncio.run(handle_whatsapp_call(call))
        except Exception:
            logger.exception("Unhandled error in WhatsApp call thread")
        finally:
            try:
                if call.state != CallState.ENDED:
                    call.hangup()
            except Exception:
                pass
            logger.info("WhatsApp call ended", extra={"call_id": call_id})

    def start(self) -> None:
        if not HAS_PYVOIP:
            logger.error(
                "pyVoIP is not installed. WhatsApp SIP server will not start. "
                "Run: pip install pyVoIP"
            )
            return
        if not META_SIP_USERNAME:
            logger.warning(
                "whatsapp_sip.username is not configured. "
                "WhatsApp SIP server will not start."
            )
            return

        kwargs: Dict[str, Any] = {
            "server": SIP_META_SERVER or "0.0.0.0",
            "port": SIP_LISTEN_PORT,
            "username": META_SIP_USERNAME,
            "password": META_SIP_PASSWORD,
            "callCallback": self._on_call,
        }
        if SIP_MY_IP:
            kwargs["myIP"] = SIP_MY_IP

        try:
            self._phone = VoIPPhone(**kwargs)
            self._phone.start()
            logger.info(
                "WhatsApp SIP server started",
                extra={"port": SIP_LISTEN_PORT, "username": META_SIP_USERNAME},
            )
        except Exception:
            logger.exception("Failed to start WhatsApp SIP server")

    def stop(self) -> None:
        if self._phone is not None:
            try:
                self._phone.stop()
                logger.info("WhatsApp SIP server stopped")
            except Exception:
                logger.debug("Error stopping WhatsApp SIP server", exc_info=True)


# Module-level singleton — imported and started by main.py
whatsapp_server = WhatsAppCallServer()