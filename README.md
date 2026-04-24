Refactor & Retrieval Changeover — Voice Agent (Twilio + OpenAI Realtime)

Overview
- Local retrieval using LanceDB and SentenceTransformers (always on).
- Twilio call handling and OpenAI Realtime streaming preserved.

Setup
- Python 3.10+ recommended.
- Install deps: `pip install -r requirements.txt`
- Set `OPENAI_API_KEY` in `.env` or environment.

Docker Deploy
- Copy `.env.example` to `.env` and fill in the required credentials (OpenAI, Twilio, ElevenLabs, etc.).
- For a local container run, use `docker compose up --build` on first run or `docker compose up -d` afterward.
- The container reads `ALMA_CONFIG_FILE` (default `/app/config/settings.yaml`); override it via `.env` if you mount a different config.

Linux VPS Deploy
- Build a transportable image locally: `docker build -t alma-ai-voice-agent:vps .`
- Either push that image to a registry, or export it as a tarball:
  - `docker save alma-ai-voice-agent:vps | gzip > alma-ai-voice-agent-vps.tar.gz`
- Copy the image archive, `.env`, `docker-compose.yml`, `config/settings.yaml`, and the `data/` directory you want on the server.
- On the VPS, load the image if you used a tarball:
  - `gunzip -c alma-ai-voice-agent-vps.tar.gz | docker load`
- Then start it on the VPS with `docker compose up -d`.
- Edit `.env` on the VPS to change voices, models, or API keys, then restart with `docker compose up -d`.

Ingestion
- Run: `python scripts/ingest_docs.py`
- Creates `./data/lancedb` and builds/updates table `kb_docs`.
- If `./data/docs/source.csv` or `./data/docs/source.pdf` are missing, sample docs are auto-created.

Query LanceDB (manual check)
- Search the index with a text query and inspect matches:
  - `python scripts/query_lancedb.py "What is the check-in time?" --top_k 4 --verify`
  - Prints distance, similarity, snippet, and optional vector info.

Config (defaults in `config/settings.yaml`, overrides via `src/utils/config.py`)
- `EMBED_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `LANCEDB_PATH` (default: `./data/lancedb`)
- `DOC_CSV_PATH` (default: `./data/docs/source.csv`)
- `DOC_PDF_PATH` (default: `./data/docs/source.pdf`)
 - `CHUNK_CHARS` (default: `800`)
 - `CHUNK_OVERLAP` (default: `120`)
- `TOP_K` (default: `4`)
- `DISTANCE` (default: `cosine`)
- `SESSION_SYSTEM_PROMPT` (default provided)
- `ENABLE_OTEL` (default: `false`) — turn on OpenTelemetry tracing/log correlation
- `OTEL_SERVICE_NAME` / `OTEL_ENVIRONMENT` — service metadata when telemetry is enabled
- `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP gRPC endpoint (e.g. `http://localhost:4317`)
- `REALTIME_CONNECT_ATTEMPTS`, `REALTIME_CONNECT_BACKOFF`, `REALTIME_CONNECT_BACKOFF_MAX` — retry tuning for the realtime websocket

Run
- Start server: `python main.py`
- Twilio points to `POST /incoming-call`; media stream upgrades to `/media-stream`.

Tests
- `pytest -q` (tests are lightweight and do not hit network).

Telemetry
- Enable OpenTelemetry export by setting `ENABLE_OTEL=true` and pointing `OTEL_EXPORTER_OTLP_ENDPOINT` at your collector.
- Logs now include `trace_id` / `span_id` fields so you can correlate application logs with spans in your APM.
