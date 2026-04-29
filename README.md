# Alma AI Voice Agent

A production-grade multilingual voice AI assistant for real estate customer engagement. Handles incoming calls via Twilio and direct SIP (MicroSIP, Linphone, etc.), streams audio through Google Gemini 2.0 Live for real-time conversation, and uses a local RAG pipeline (LanceDB + SentenceTransformers) for property knowledge retrieval.

Built for Alma Resort (Tangier, Morocco) — qualifies leads, answers property questions, and books reservations in French, with support for Arabic, Darija, Spanish, and English.

---

## What's implemented

### Voice conversation engine
- Real-time audio streaming: Twilio / SIP (8 kHz µ-law) ↔ Gemini 2.0 Live (16 kHz input / 24 kHz output)
- WebSocket reconnect with configurable retry/backoff
- Playback guard to prevent audio overlap (default 1.5 s)
- Configurable voice activity detection (VAD threshold, silence padding)
- Optional ElevenLabs TTS fallback (multilingual v2)

### Generic SIP server (MicroSIP / local testing)
- pyVoIP-based SIP server listening on UDP port 5060
- Accepts calls from any SIP client directly — no registration required
- Acts as its own SIP registrar (responds 200 OK to REGISTER, skips self-registration)
- Auto-answers every incoming INVITE and bridges to Gemini Live
- Same Gemini engine and tool set as the Twilio path
- Python 3.13 compatible via `audioop-lts`

### WhatsApp Business SIP bridge
- pyVoIP-based SIP/RTP bridge (TLS terminated upstream by nginx/HAProxy on port 5061)
- Requires Meta WhatsApp Business Calling API credentials
- Shares the same Gemini Live engine and tool set

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
| `get_property_specs` | Returns apartment/villa/lot attributes from `data/catalog/property_specs.yaml` |
| `get_project_info` | Document-grounded Q&A via RAG retrieval pipeline (top-4 snippets) |
| `get_project_facts` | Curated facts from `data/catalog/project_facts.yaml` (location, architects, amenities, contacts) |
| `end_call` | Gracefully terminates the call |

All tools have per-tool rate limits, input validation, and response size clamping (16 000 chars max).

### Knowledge base (data files)
| File | Content |
|------|---------|
| `data/docs/appartements.csv` | RIPT and private apartment specs, surfaces, prices, payment terms |
| `data/docs/villa.csv` | Villa Type 1/2/3 specs, surfaces, prices |
| `data/docs/lot_de_terrains.csv` | Land lots (bande/jumelée), surfaces, constructibility rules |
| `data/docs/general_information.csv` | Project description, location, amenities, hotel, legal |
| `data/docs/info.txt` | Summary of all product lines and resort features |
| `data/catalog/property_specs.yaml` | Structured product catalog used by `get_property_specs` |
| `data/catalog/project_facts.yaml` | Curated facts used by `get_project_facts` |
| `data/catalog/alma_resort_facts.yaml` | General resort reference data |
| `data/catalog/property_descriptions.yaml` | Marketing descriptions per product type |

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
| SIP (local) | pyVoIP 2.x + audioop-lts (Python 3.13 compatible) |
| WhatsApp | pyVoIP SIP/RTP (Meta Business Calling API) |
| Vector DB | LanceDB |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Web framework | FastAPI + Uvicorn |
| TTS (optional) | ElevenLabs multilingual v2 |
| Persistence | SQLite (call ledger + reservations) |
| Observability | OpenTelemetry SDK + OTLP |
| Container | Docker / Docker Compose (Python 3.11-slim) |

---

## Setup

**Requirements:** Python 3.11+ (3.13 supported)

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in credentials:

```bash
GEMINI_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
PUBLIC_HTTP_BASE_URL=https://your-ngrok-url   # Twilio webhook base

# Generic SIP server (MicroSIP local testing)
SIP_ENABLED=true
SIP_LISTEN_PORT=5060
SIP_MY_IP=192.168.x.x                         # your local LAN IP

# Optional
ELEVENLABS_API_KEY=...
ENABLE_OTEL=false
```

---

## Ingest documents

Place CSV, PDF, or TXT files in `data/docs/`, then run:

```bash
python src/ingest_docs.py
```

Builds (or updates) the LanceDB table `kb_docs` at `./data/lancedb`. Re-run after editing any file in `data/docs/` or `data/catalog/`.

Verify the index:

```bash
python src/query_lancedb.py "Quel est le prix des appartements ?" --top_k 4 --verify
```

---

## Run

```bash
python src/main.py
```

On startup you should see:
```
SIP server started on UDP port 5060
INFO: Uvicorn running on http://0.0.0.0:3001
```

---

## Testing

### Option A — MicroSIP (local, no internet needed)

> **Note:** Requires being on the same WiFi as the server. Does not work on corporate networks with SIP/UDP blocked.

1. Install [MicroSIP](https://www.microsip.org/)
2. Add account:
   | Field | Value |
   |-------|-------|
   | SIP Server | `192.168.x.x` (your machine's LAN IP from `SIP_MY_IP`) |
   | SIP Port | `5060` |
   | Username | `alma` |
   | Domain | `192.168.x.x` |
   | Password | *(leave blank)* |
   | Transport | UDP |
3. Dial `alma` and press call
4. Alma greets you in ~1 second: *"Bonjour, Alma Resort à l'appareil. Je vous écoute."*

### Option B — Twilio (any phone, anywhere)

1. Start ngrok in a second terminal:
   ```bash
   ngrok http 3001
   ```
2. Copy the ngrok URL (e.g. `https://xxxx.ngrok-free.app`) into `.env`:
   ```
   PUBLIC_HTTP_BASE_URL=https://xxxx.ngrok-free.app
   ```
3. Restart the server
4. In Twilio Console → Phone Numbers → your number → Voice Configuration:
   - Webhook URL: `https://xxxx.ngrok-free.app/incoming-call`
   - Method: `HTTP POST`
5. Call your Twilio number from any phone

> **Tips:**
> - Free ngrok URLs change every restart — use `ngrok http 3001 --domain=your-name.ngrok-free.app` for a fixed URL
> - Enable Geo Permissions in Twilio Console (Voice → Settings → Geo Permissions) for international callers
> - Corporate firewalls may block ngrok — use a personal hotspot or VPS instead

### Option C — Share with colleagues

Send them your **Twilio phone number**. They call it like any normal number — nothing to install. You need ngrok (or a VPS) running for the duration of the test.

### Test scenarios
Try these during a call to verify all systems:

| Scenario | What to say | Tests |
|----------|------------|-------|
| Location | "Où se trouve Alma Resort ?" | `get_project_facts` |
| Price | "Quel est le prix des appartements ?" | `get_property_specs` |
| Product detail | "Parlez-moi des villas" | `get_property_specs` |
| General info | "Quels sont les équipements du resort ?" | `get_project_info` |
| Booking | "Je voudrais prendre rendez-vous" | `ask_to_reserve` |

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

## Known limitations

- **Corporate networks:** SIP/UDP (port 5060) and RTP (10000–20000) are frequently blocked by enterprise firewalls (FortiGuard, etc.). Use a personal hotspot or VPS for testing in this environment.
- **WhatsApp SIP:** Requires Meta WhatsApp Business Calling API access (invite-only), a public VPS with TLS on port 5061, and a Cloud API phone number.
- **ngrok free tier:** URL changes on every restart. Use `--domain` flag with a free ngrok account for a stable URL.
- **Python 3.13:** Supported via `audioop-lts`. Docker image uses Python 3.11-slim.

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
├── main.py                    # Entrypoint — starts FastAPI + SIP servers
├── ingest_docs.py             # Document ingestion CLI
├── query_lancedb.py           # Manual index query tool
├── ingestion/                 # CSV / PDF / TXT loaders, chunking, embedder, LanceDB indexing
├── retrieval/                 # LanceDB client, MMR + lexical pipeline
└── utils/
    ├── realtime.py            # Gemini Live streaming handler (FastAPI WebSocket — Twilio path)
    ├── sip_server.py          # Generic SIP server (MicroSIP / any SIP client)
    ├── whatsapp_calling.py    # WhatsApp Business SIP bridge (Meta credentials required)
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
├── docs/                      # Source documents (CSV, PDF, TXT) — edit and re-ingest
├── lancedb/                   # Vector database storage (auto-generated)
├── catalog/                   # Structured product catalogs (YAML)
└── monitoring/                # SQLite ledger + transcript files (auto-generated)
```
