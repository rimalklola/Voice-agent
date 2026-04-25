# Alma AI Voice Agent

A production-grade multilingual voice AI assistant for real estate customer engagement. Handles incoming calls via Twilio and WhatsApp Business SIP, streams audio through Google Gemini 2.0 Live for real-time conversation, and uses a local RAG pipeline (LanceDB + SentenceTransformers) for property knowledge retrieval.

Built for Alma Resort (Tangier, Morocco) — qualifies leads, answers property questions, and books reservations in French, with support for Arabic, Darija, Spanish, and English.

---

## What's implemented

### Voice conversation engine
- Real-time audio streaming: Twilio (8 kHz µ-law) ↔ Gemini 2.0 Live (16 kHz input / 24 kHz output)
- WebSocket reconnect with configurable retry/backoff
- Playback guard to prevent audio overlap (default 1.5 s)
- Configurable voice activity detection (VAD threshold, silence padding)
- Optional ElevenLabs TTS fallback (multilingual v2)

### WhatsApp Business SIP bridge
- pyVoIP-based SIP/RTP bridge on port 5060 (TLS terminated upstream by nginx/HAProxy on 5061)
- Shares the same Gemini Live engine and tool set as the Twilio path

### Local RAG knowledge base
- Document ingestion: CSV rows, PDF pages, and TXT files → chunked (800 chars, 120-char overlap)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors) stored in LanceDB
- Hybrid search: vector similarity (cosine) + lexical overlap scoring (4-gram)
- MMR (Maximal Marginal Relevance) reranking for response diversity
- Source-type biasing: upweight CSV facts, tune PDF weight independently

### Agent tools (LLM-callable)
| Tool | What it does |
|------|-------------|
| `ask_to_reserve` | Captures visitor details, validates phone/email, deduplicates within 24 h, persists to SQLite |
| `get_property_specs` | Returns apartment/villa/lot attributes from catalog JSON with fuzzy matching |
| `get_project_info` | Document-grounded Q&A via retrieval pipeline (top-4 snippets) |
| `get_project_facts` | Curated facts lookup (location, architects, delivery dates, security) |
| `end_call` | Gracefully terminates the call |

All tools have per-tool rate limits, input validation (regex phone/email, SQL injection guard), and response size clamping (16 000 chars max).

### Monitoring & compliance
- SQLite call ledger: 40+ metrics per call (duration, tool usage counts, error events)
- Full transcript history saved per call with source attribution
- PII redaction in logs (phone numbers, email addresses masked)
- Circuit breaker on external API calls

### Observability
- Structured JSON logs with `trace_id` / `span_id` correlation
- Optional OpenTelemetry export (OTLP gRPC) — compatible with Datadog, Jaeger, etc.
- Health check endpoint for readiness probes

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| LLM / voice | Google Gemini 2.0 Flash Live (`gemini-2.0-flash-live-001`) |
| Telephony | Twilio media streams |
| WhatsApp | pyVoIP SIP/RTP |
| Vector DB | LanceDB |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Web framework | FastAPI + Uvicorn |
| TTS (optional) | ElevenLabs multilingual v2 |
| Persistence | SQLite (call ledger + reservations) |
| Observability | OpenTelemetry SDK + OTLP |
| Container | Docker / Docker Compose (Python 3.11-slim) |

---

## Setup

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in credentials:

```bash
GEMINI_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
PUBLIC_HTTP_BASE_URL=https://your-public-url   # Twilio webhook base

# Optional
ELEVENLABS_API_KEY=...
ENABLE_OTEL=false
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Ingest documents

Place CSV, PDF, or TXT files in `data/docs/`, then run:

```bash
python src/ingest_docs.py
```

Builds (or updates) the LanceDB table `kb_docs` at `./data/lancedb`. Missing sample files are auto-created.

Verify the index:

```bash
python src/query_lancedb.py "What is the delivery date?" --top_k 4 --verify
```

---

## Run

```bash
python src/main.py
```

- Twilio: configure `POST /incoming-call` as your webhook; media stream upgrades on `GET /media-stream`
- WhatsApp SIP: listens on `META_SIP_LISTEN_PORT` (default 5060)

---

## Docker

Local development:

```bash
docker compose up --build
```

VPS deployment (offline tarball):

```bash
docker build -t alma-ai-voice-agent:vps .
docker save alma-ai-voice-agent:vps | gzip > alma-ai-voice-agent-vps.tar.gz
# Copy archive + .env + docker-compose.yml + config/ + data/ to server
gunzip -c alma-ai-voice-agent-vps.tar.gz | docker load
docker compose up -d
```

The container reads config from `ALMA_CONFIG_FILE` (default `/app/config/settings.yaml`).

---

## Key configuration (`config/settings.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `realtime.voice` | `Kore` | Gemini voice name |
| `realtime.temperature` | `0.65` | Sampling temperature |
| `realtime.vad_threshold` | `0.65` | Voice activity detection sensitivity |
| `realtime.connect_attempts` | `3` | WebSocket retry attempts |
| `realtime.playback_guard_ms` | `1500` | Anti-overlap guard (ms) |
| `ingestion.chunk_chars` | `800` | Chunk size for document splitting |
| `retrieval.top_k` | `4` | Context snippets returned per query |
| `retrieval.use_mmr` | `true` | MMR diversity reranking |
| `retrieval.mmr_lambda` | `0.5` | MMR diversity weight |
| `retrieval.use_lexical_boost` | `true` | Hybrid vector + lexical search |
| `guardrails.reservations.rate_limit` | `5 / 60 s` | Reservation tool rate limit |
| `guardrails.knowledge.rate_limit` | `30 / 60 s` | Knowledge query rate limit |
| `monitoring.enabled` | `true` | Call ledger + transcripts |

All settings can be overridden via environment variables.

---

## Tests

```bash
pytest -q
```

Seven test files covering: end-to-end smoke, MMR retrieval, monitoring ledger, transcript history, and conversation flow. Tests do not hit the network.

---

## Project structure

```
src/
├── main.py                    # Entrypoint — starts FastAPI + WhatsApp SIP server
├── ingest_docs.py             # Document ingestion CLI
├── query_lancedb.py           # Manual index query tool
├── ingestion/                 # CSV / PDF / TXT loaders, chunking, embedder, LanceDB indexing
├── retrieval/                 # LanceDB client, MMR + lexical pipeline
└── utils/
    ├── realtime.py            # Gemini Live streaming handler (FastAPI WebSocket)
    ├── whatsapp_calling.py    # WhatsApp SIP bridge (pyVoIP)
    ├── tools.py               # Agent tool implementations
    ├── call_ledger.py         # SQLite call metrics store
    ├── transcript_history.py  # Per-call conversation recorder
    ├── pii_redaction.py       # Log-level PII masking
    ├── circuit_breaker.py     # External API resilience
    ├── telemetry.py           # OpenTelemetry integration
    ├── monitoring.py          # Pilot call metrics
    ├── config.py              # YAML config + env override loader
    └── prompts/
        └── alma_system_prompt.txt  # French commercial advisor system prompt
config/
└── settings.yaml              # Full configuration file
data/
├── docs/                      # Source documents (CSV, PDF, TXT)
├── lancedb/                   # Vector database storage
├── catalog/                   # Property catalog JSON files
└── monitoring/                # SQLite ledger + transcript files
```
