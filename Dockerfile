# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/monitoring /app/data/lancedb \
    && chown -R appuser:appuser /app

USER appuser

# HTTP/WebSocket for Twilio
EXPOSE 3001
# SIP signalling
EXPOSE 5060/udp
# RTP media
EXPOSE 10000-20000/udp

CMD ["python", "src/main.py"]
