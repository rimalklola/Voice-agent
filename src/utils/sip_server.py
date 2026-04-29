"""
Generic SIP server — accepts calls from any SIP client (MicroSIP, Linphone, etc.)
and bridges them to Gemini Live.

Call flow:
  1. SIP client (e.g. MicroSIP) dials  sip:alma@<YOUR_LOCAL_IP>:5060
  2. pyVoIP receives the INVITE and puts the call in RINGING state.
  3. AlmaVoIPCall.ringing() auto-answers and spawns the bridge thread.
  4. Audio: RTP G.711 mu-law 8kHz ↔ Gemini Live PCM 16kHz/24kHz
  5. Gemini can call agent tools mid-conversation.

MicroSIP setup:
  - Add account: Display name = Alma, Username = alma, Domain = <YOUR_LOCAL_IP>:5060
  - No password, Transport = UDP
  - Dial: alma@<YOUR_LOCAL_IP>  (or just press call)

Configuration (.env):
  SIP_LISTEN_PORT   internal UDP port pyVoIP listens on  (default 5060)
  SIP_MY_IP         your local/LAN IP so RTP replies route correctly
  SIP_ENABLED       set to "true" to start the server (default true)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from pyVoIP.VoIP.phone import VoIPPhone, VoIPPhoneParameter
    from pyVoIP.VoIP.call import VoIPCall, CallState
    from pyVoIP.VoIP.status import PhoneStatus
    from pyVoIP.SIP.client import SIPClient
    from pyVoIP.SIP.message import SIPMessage, SIPMessageType
    HAS_PYVOIP = True
except Exception as exc:
    HAS_PYVOIP = False
    VoIPPhone = None          # type: ignore
    VoIPPhoneParameter = None # type: ignore
    VoIPCall = object         # type: ignore
    CallState = None          # type: ignore
    SIPClient = object        # type: ignore
    SIPMessage = None         # type: ignore
    SIPMessageType = None     # type: ignore
    _PYVOIP_ERR = str(exc)

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    genai = None   # type: ignore
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
from src.utils.transcript_history import TranscriptSessionBuffer, save_transcript_record

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = config.project.gemini_api_key
GEMINI_MODEL   = config.realtime.asr_model
VOICE          = config.realtime.voice
TEMPERATURE    = config.realtime.temperature

SIP_LISTEN_PORT: int  = int(os.getenv("SIP_LISTEN_PORT", "5060"))
SIP_MY_IP: str        = os.getenv("SIP_MY_IP", "")
SIP_ENABLED: bool     = os.getenv("SIP_ENABLED", "true").lower() == "true"

_mon_cfg = config.get("monitoring", {}) or {}
CALL_LEDGER_SQLITE_PATH    = str(_mon_cfg.get("sqlite_path", "./data/monitoring/call_ledger.sqlite3"))
TRANSCRIPT_HISTORY_DIR     = str(_mon_cfg.get("transcript_history_dir", "./data/monitoring/transcripts"))
TRANSCRIPT_HISTORY_ENABLED = bool(_mon_cfg.get("transcript_history_enabled", True))

# ---------------------------------------------------------------------------
# Audio conversion  (G.711 mu-law 8 kHz ↔ PCM 16 kHz / 24 kHz)
# ---------------------------------------------------------------------------
_ULAW_DECODE_TABLE: Optional[np.ndarray] = None


def _get_ulaw_decode_table() -> np.ndarray:
    global _ULAW_DECODE_TABLE
    if _ULAW_DECODE_TABLE is None:
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            b = (~i) & 0xFF
            sign = b & 0x80
            exp  = (b >> 4) & 0x07
            mant = b & 0x0F
            mag  = ((mant << 3) | 0x84) << exp
            mag -= 0x84
            table[i] = max(-32768, min(32767, -mag if sign else mag))
        _ULAW_DECODE_TABLE = table
    return _ULAW_DECODE_TABLE


def _mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """G.711 mu-law 8 kHz → PCM int16 16 kHz (2× linear interpolation)."""
    table  = _get_ulaw_decode_table()
    idx    = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    pcm_8k = table[idx].astype(np.float32)
    n = len(pcm_8k)
    if n == 0:
        return b""
    pcm_16k = np.empty(n * 2, dtype=np.float32)
    pcm_16k[0::2] = pcm_8k
    if n > 1:
        pcm_16k[1::2][:-1] = (pcm_8k[:-1] + pcm_8k[1:]) / 2.0
    pcm_16k[1::2][-1] = pcm_8k[-1]
    return pcm_16k.clip(-32768, 32767).astype(np.int16).tobytes()


def _pcm16_to_ulaw(samples: np.ndarray) -> bytes:
    """int16 numpy array → G.711 mu-law bytes."""
    BIAS, CLIP = 132, 32635
    s32  = samples.astype(np.int32)
    sign = np.where(s32 < 0, np.uint8(0x80), np.uint8(0)).astype(np.uint8)
    abs_s = np.clip(np.abs(s32), 0, CLIP) + BIAS
    exp   = np.floor(np.log2(abs_s.clip(1))).astype(np.int32) - 3
    exp   = np.clip(exp, 0, 7).astype(np.uint8)
    mant  = ((abs_s >> (exp.astype(np.int32) + 3)) & 0x0F).astype(np.uint8)
    return (~(sign | (exp << 4) | mant)).astype(np.uint8).tobytes()


def _pcm24k_to_mulaw(pcm_bytes: bytes) -> bytes:
    """PCM int16 24 kHz → G.711 mu-law 8 kHz (3:1 decimation)."""
    if not pcm_bytes:
        return b""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n_out   = len(samples) // 3
    if n_out == 0:
        return b""
    pcm_8k = samples[: n_out * 3].reshape(n_out, 3).mean(axis=1).clip(-32768, 32767).astype(np.int16)
    return _pcm16_to_ulaw(pcm_8k)


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
_GEMINI_CLIENT: Optional[Any] = None


def _get_gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        if genai is None:
            raise RuntimeError("google-genai not installed")
        _GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return _GEMINI_CLIENT


def _build_gemini_tools() -> List[Any]:
    if gtypes is None:
        return []
    declarations = [
        gtypes.FunctionDeclaration(
            name="ask_to_reserve",
            description="Démarrer une demande de visite/rappel/brochure en fin d'appel.",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "full_name":    gtypes.Schema(type="STRING"),
                    "phone":        gtypes.Schema(type="STRING"),
                    "lead_source":  gtypes.Schema(type="STRING"),
                    "request_type": gtypes.Schema(type="STRING"),
                    "email":        gtypes.Schema(type="STRING"),
                    "date":         gtypes.Schema(type="STRING"),
                    "time":         gtypes.Schema(type="STRING"),
                    "guests":       gtypes.Schema(type="INTEGER"),
                    "notes":        gtypes.Schema(type="STRING"),
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
                    "query": gtypes.Schema(type="STRING"),
                    "top_k": gtypes.Schema(type="INTEGER"),
                },
                required=["query"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_property_specs",
            description="Surfaces, typologies, dates de livraison.",
            parameters=gtypes.Schema(
                type="OBJECT",
                properties={
                    "category":   gtypes.Schema(type="STRING"),
                    "variant":    gtypes.Schema(type="STRING"),
                    "attributes": gtypes.Schema(type="ARRAY", items=gtypes.Schema(type="STRING")),
                    "question":   gtypes.Schema(type="STRING"),
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
                    "section":  gtypes.Schema(type="STRING"),
                    "topic":    gtypes.Schema(type="STRING"),
                    "question": gtypes.Schema(type="STRING"),
                },
            ),
        ),
        gtypes.FunctionDeclaration(
            name="end_call",
            description="Met fin à l'appel.",
            parameters=gtypes.Schema(type="OBJECT", properties={}),
        ),
    ]
    return [gtypes.Tool(function_declarations=declarations)]


def _build_live_config() -> Any:
    if gtypes is None:
        raise RuntimeError("google-genai not installed")
    system_prompt = (
        config.retrieval.session_system_prompt
        + "\n\n" + build_temporal_guidance()
        + "\n\nRègle supplémentaire: Réponses ≤ 2 phrases."
        + "\nUtilise `get_property_specs` pour les surfaces, `get_project_facts` pour les"
          " informations générales structurées, et `get_project_info` pour le reste."
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


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
async def _run_tool(fc: Any, monitoring_row: Dict[str, Any]) -> Any:
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
        cat      = args.get("category", "") or ""
        variant  = args.get("variant")
        question = args.get("question")
        raw_attr = args.get("attributes")
        attr_list = [str(a) for a in raw_attr if isinstance(a, str)] if isinstance(raw_attr, list) else None
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
        section  = args.get("section")
        topic    = args.get("topic")
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


# ---------------------------------------------------------------------------
# Gemini Live bridge
# ---------------------------------------------------------------------------
async def _bridge_call_to_gemini(call: Any) -> None:
    """
    Bridge one answered SIP call to Gemini Live.
    Runs in its own asyncio event loop (via asyncio.run in a thread).
    """
    t0 = time.perf_counter()
    monitoring_row   = new_pilot_call_row(voice=VOICE, model=GEMINI_MODEL, temperature=TEMPERATURE)
    transcript_buf   = TranscriptSessionBuffer(voice=VOICE, model=GEMINI_MODEL, temperature=TEMPERATURE)
    monitoring_row["call_id"] = transcript_buf.call_id
    monitoring_status = "unknown"
    monitoring_failure_reason = ""

    loop     = asyncio.get_running_loop()
    rtp_in: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    stop_evt = threading.Event()

    RTP_CHUNK = 320  # 40 ms × 8000 Hz × 1 byte/sample

    def _rtp_reader() -> None:
        while not stop_evt.is_set() and call.state == CallState.ANSWERED:
            try:
                audio = call.read_audio(RTP_CHUNK, blocking=True)
                if audio:
                    asyncio.run_coroutine_threadsafe(rtp_in.put(audio), loop)
            except Exception:
                break

    reader_thread = threading.Thread(target=_rtp_reader, daemon=True, name="rtp-reader")
    reader_thread.start()

    try:
        client   = _get_gemini_client()
        live_cfg = _build_live_config()

        async with client.aio.live.connect(model=GEMINI_MODEL, config=live_cfg) as session:
            await session.send(
                input=gtypes.LiveClientContent(
                    turns=[gtypes.Content(
                        parts=[gtypes.Part(text="Dis simplement : 'Bonjour, Alma Resort à l'appareil. Je vous écoute.'")],
                        role="user",
                    )],
                    turn_complete=True,
                )
            )

            end_requested = False

            async def _mic_to_gemini() -> None:
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
                        pass

            async def _gemini_to_speaker() -> None:
                nonlocal end_requested, monitoring_status, monitoring_failure_reason
                assistant_chunks: List[str] = []

                async for response in session.receive():
                    if end_requested:
                        break

                    if response.data and call.state == CallState.ANSWERED:
                        mulaw_out = _pcm24k_to_mulaw(response.data)
                        if mulaw_out:
                            await asyncio.to_thread(call.write_audio, mulaw_out)

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
                            agg = "".join(assistant_chunks).strip()
                            assistant_chunks.clear()
                            if agg:
                                transcript_buf.add_turn("assistant", agg, "turn_complete")

                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            if fc.name == "end_call":
                                record_tool_call(monitoring_row, "end_call")
                                if monitoring_status == "unknown":
                                    monitoring_status = "ended_by_agent"
                                try:
                                    await session.send(
                                        input=gtypes.LiveClientToolResponse(
                                            function_responses=[gtypes.FunctionResponse(
                                                id=fc.id, name=fc.name,
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
                                        id=fc.id, name=fc.name,
                                        response={"result": result},
                                    )]
                                )
                            )

            await asyncio.gather(_mic_to_gemini(), _gemini_to_speaker())

        if monitoring_status == "unknown":
            monitoring_status = "completed"

    except Exception as exc:
        logger.exception("SIP Gemini session error")
        monitoring_status = "aborted"
        monitoring_failure_reason = exc.__class__.__name__
    finally:
        stop_evt.set()
        reader_thread.join(timeout=2)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("SIP call finished", extra={"status": monitoring_status, "duration_ms": duration_ms})

        if TRANSCRIPT_HISTORY_ENABLED:
            record = transcript_buf.to_record(
                session_status=monitoring_status,
                failure_reason=monitoring_failure_reason,
                duration_ms=duration_ms,
            )
            try:
                save_transcript_record(TRANSCRIPT_HISTORY_DIR, record, sqlite_path=CALL_LEDGER_SQLITE_PATH)
            except Exception:
                logger.exception("Failed saving transcript")


# ---------------------------------------------------------------------------
# Custom SIPClient — responds 200 OK to REGISTER so MicroSIP can connect
# ---------------------------------------------------------------------------
if HAS_PYVOIP:
    class RegistrarSIPClient(SIPClient):
        """
        SIPClient that acts as its own registrar:
        - Skips self-registration on startup (no outbound REGISTER sent).
        - Responds 200 OK to any inbound REGISTER (e.g. from MicroSIP).
        - All other messages handled normally by the parent.
        """

        def register(self) -> None:
            """Skip outbound registration — we are the server, not a client."""
            self.phone._status = PhoneStatus.REGISTERED
            logger.debug("SIP self-registration skipped (server mode)")

        def deregister(self) -> bool:
            """Skip deregistration on shutdown."""
            return True

        def handle_new_connection(self, conn: Any) -> None:
            raw = conn.peak()
            try:
                msg = SIPMessage(raw)
            except Exception:
                super().handle_new_connection(conn)
                return
            if (
                msg.type == SIPMessageType.REQUEST
                and msg.method == "REGISTER"
            ):
                response = self.gen_ok(msg)
                conn.send(response)
                logger.debug("REGISTER from %s → 200 OK", msg.headers.get("From", {}).get("raw", "?"))
                return
            super().handle_new_connection(conn)

else:
    RegistrarSIPClient = None  # type: ignore


# ---------------------------------------------------------------------------
# Custom VoIPCall — auto-answers every incoming INVITE
# ---------------------------------------------------------------------------
if HAS_PYVOIP:
    class AlmaVoIPCall(VoIPCall):
        """
        Auto-answering call class.  pyVoIP calls ringing() 1 s after the INVITE.
        We override it to answer immediately and start the Gemini bridge thread.
        """

        def ringing(self, request: Any) -> None:
            if self.state != CallState.RINGING:
                self.request = request
                return
            try:
                self.answer()
            except Exception:
                logger.exception("Failed to answer SIP call")
                return
            logger.info("SIP call answered", extra={"call_id": self.call_id})
            threading.Thread(
                target=lambda: asyncio.run(_bridge_call_to_gemini(self)),
                daemon=True,
                name=f"sip-bridge-{self.call_id}",
            ).start()

else:
    AlmaVoIPCall = None  # type: ignore


# ---------------------------------------------------------------------------
# SIP server
# ---------------------------------------------------------------------------
class SIPCallServer:
    """
    Listens on UDP port SIP_LISTEN_PORT for incoming SIP INVITEs.
    No registration, no credentials — any SIP client can call in directly.

    MicroSIP: add account with domain = <your local IP>:5060, no password.
    """

    def __init__(self) -> None:
        self._phone: Optional[Any] = None

    def start(self) -> None:
        if not HAS_PYVOIP:
            logger.error("pyVoIP failed to import: %s — SIP server will not start.", _PYVOIP_ERR if not HAS_PYVOIP else "")
            return
        if not SIP_ENABLED:
            logger.info("SIP server disabled (SIP_ENABLED=false)")
            return

        my_ip = SIP_MY_IP or "0.0.0.0"
        params = VoIPPhoneParameter(
            server=my_ip,
            port=SIP_LISTEN_PORT,
            user="alma",
            credentials_manager=None,
            bind_ip="0.0.0.0",
            bind_port=SIP_LISTEN_PORT,
            call_class=AlmaVoIPCall,
            sip_class=RegistrarSIPClient,  # responds 200 OK to REGISTER
            rtp_port_low=10000,
            rtp_port_high=20000,
        )
        try:
            self._phone = VoIPPhone(params)
            self._phone.start()
            logger.info("SIP server started on UDP port %d", SIP_LISTEN_PORT)
        except Exception:
            logger.exception("Failed to start SIP server")

    def stop(self) -> None:
        if self._phone is not None:
            try:
                self._phone.stop()
                logger.info("SIP server stopped")
            except Exception:
                logger.debug("Error stopping SIP server", exc_info=True)


sip_server = SIPCallServer()
