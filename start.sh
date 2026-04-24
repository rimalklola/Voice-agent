#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log() {
  printf '[start] %s\n' "$*"
}

die() {
  printf '[start] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "$2"
  fi
}

usage() {
  cat <<'EOS'
Usage: ./start.sh [--docker|--local] [--skip-ingest]

Options:
  --docker       Force Docker Compose deployment.
  --local        Force local Python deployment (uses .venv).
  --skip-ingest  Skip the document ingestion warmup step.
  --help         Show this message.
EOS
}

MODE="auto"
SKIP_INGEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)
      MODE="docker"
      shift
      ;;
    --local)
      MODE="local"
      shift
      ;;
    --skip-ingest)
      SKIP_INGEST=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

try_configure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    return 0
  fi
  return 1
}

ensure_docker_mode() {
  if ! try_configure_docker; then
    die "Docker Compose not found. Install Docker Desktop from https://www.docker.com/get-started."
  fi
  RUN_MODE="docker"
  log "Using Docker Compose (${COMPOSE_CMD[*]})"
}

ensure_python_version() {
  if ! python3 - <<'PY'
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    die "Python 3.10+ is required. Install it from https://www.python.org/downloads/"
  fi
}

activate_local_env() {
  require_cmd python3 "Python 3.10+ is required. Install it from https://www.python.org/downloads/"
  ensure_python_version
  if [ ! -d ".venv" ]; then
    log "Creating virtual environment (.venv)"
    python3 -m venv .venv || die "Failed to create .venv"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate || die "Failed to activate .venv"
  if ! python -m pip install --upgrade pip; then
    log "pip upgrade failed; continuing with existing version."
  fi
  log "Installing Python dependencies (requirements.txt)"
  python -m pip install -r requirements.txt
  RUN_MODE="local"
  log "Using local virtual environment at .venv"
}

configure_mode() {
  case "$MODE" in
    docker)
      ensure_docker_mode
      ;;
    local)
      activate_local_env
      ;;
    auto)
      if try_configure_docker; then
        RUN_MODE="docker"
        log "Docker Compose detected; running containerized deployment."
      else
        log "Docker not detected; falling back to local Python."
        activate_local_env
      fi
      ;;
  esac
}

validate_env() {
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == YOUR_* ]]; then
    die "OPENAI_API_KEY is missing. Update .env with your OpenAI key."
  fi
  if [[ -z "${TWILIO_ACCOUNT_SID:-}" || "${TWILIO_ACCOUNT_SID}" == YOUR_* ]]; then
    die "TWILIO_ACCOUNT_SID is missing. Update .env with your Twilio Account SID."
  fi
  if [[ -z "${TWILIO_AUTH_TOKEN:-}" || "${TWILIO_AUTH_TOKEN}" == YOUR_* ]]; then
    die "TWILIO_AUTH_TOKEN is missing. Update .env with your Twilio Auth Token."
  fi
  if [[ "${USE_ELEVENLABS_TTS:-false}" == "true" ]]; then
    if [[ -z "${ELEVENLABS_API_KEY:-}" || "${ELEVENLABS_API_KEY}" == YOUR_* ]]; then
      die "ElevenLabs TTS enabled but ELEVENLABS_API_KEY is missing."
    fi
    if [[ -z "${ELEVENLABS_VOICE_ID:-}" || "${ELEVENLABS_VOICE_ID}" == YOUR_* ]]; then
      die "ElevenLabs TTS enabled but ELEVENLABS_VOICE_ID is missing."
    fi
  fi
}

ensure_env_file() {
  if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
      cp .env.example .env
      die "Created .env from .env.example. Edit it with your credentials and re-run ./start.sh."
    else
      die ".env file missing (and .env.example not found)."
    fi
  fi
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
  validate_env
}

needs_ingestion() {
  local target="${LANCEDB_PATH:-./data/lancedb}"
  if [ ! -d "$target" ]; then
    return 0
  fi
  if find "$target" -mindepth 1 -print -quit | read -r _; then
    return 1
  fi
  return 0
}

run_ingestion() {
  local target="${LANCEDB_PATH:-./data/lancedb}"
  mkdir -p "$target"
  if [ "$SKIP_INGEST" -eq 1 ]; then
    log "Skipping ingestion (requested)."
    return
  fi
  if ! needs_ingestion; then
    log "Existing LanceDB found at ${target}; skipping ingestion."
    return
  fi
  log "Running ingestion to populate ${target}"
  if [ "${RUN_MODE}" = "docker" ]; then
    "${COMPOSE_CMD[@]}" run --rm voice-agent python src/ingest_docs.py
  else
    python src/ingest_docs.py
  fi
}

start_docker() {
  if [ "${COMPOSE_CMD[0]}" = "docker" ]; then
    "${COMPOSE_CMD[@]}" up --pull always -d
  else
    "${COMPOSE_CMD[@]}" up -d
  fi
  log "Voice agent is starting. Follow logs with '${COMPOSE_CMD[*]} logs -f voice-agent'."
}

start_local() {
  log "Launching python src/main.py (Ctrl+C to stop)."
  python src/main.py
}

main() {
  ensure_env_file
  configure_mode
  run_ingestion
  if [ "${RUN_MODE}" = "docker" ]; then
    start_docker
  else
    start_local
  fi
}

main
