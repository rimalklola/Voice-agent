from __future__ import annotations

import time
import uuid
from typing import Dict, List

import lancedb
from lancedb.pydantic import LanceModel, Vector

from src.utils.config import config
from src.ingestion.chunking import Chunk


TABLE_NAME = "kb_docs"


DIM = 384


class KBDoc(LanceModel):
    id: str
    source_type: str
    source_path: str
    page: int | None
    row_idx: int | None
    text: str
    vector: Vector(DIM) # type: ignore
    meta_json: str | None
    created_at: int


def connect_db():
    return lancedb.connect(config.ingestion.lancedb_path)


def ensure_table(db) -> "lancedb.table":
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    # Create table from schema (no dummy row to avoid Null type inference)
    return db.create_table(TABLE_NAME, schema=KBDoc, mode="overwrite")


def upsert_chunks(chunks: List[Chunk], vectors: List[List[float]]):
    db = connect_db()
    table = ensure_table(db)

    # Delete prior rows for the given sources to keep idempotency
    source_paths = {c.source_path for c in chunks}
    for sp in source_paths:
        try:
            escaped = sp.replace('"', '\\"')
            table.delete(f'source_path == "{escaped}"')
        except Exception:
            pass

    now_ms = int(time.time() * 1000)
    to_write: List[Dict] = []
    for c, v in zip(chunks, vectors):
        # Deterministic id per source + page/row + chunk_id
        tag = f"{c.source_path}|{c.source_type}|{c.page}|{c.row_idx}|{c.chunk_id}"
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, tag))
        # Serialize meta to JSON string if present
        meta_json: str | None = None
        try:
            import json as _json
            if c.meta:
                meta_json = _json.dumps(c.meta)
        except Exception:
            meta_json = None
        to_write.append({
            "id": doc_id,
            "source_type": c.source_type,
            "source_path": c.source_path,
            "page": c.page,
            "row_idx": c.row_idx,
            "text": c.text,
            "vector": v,
            "meta_json": meta_json,
            "created_at": now_ms,
        })

    if to_write:
        table.add(to_write)

    # Ensure HNSW index exists
    try:
        table.create_index(
            column="vector",
            index_type="HNSW",
            metric="cosine",
            num_partitions=1,
            **{"ef_construction": 400, "m": 32},
        )
    except Exception:
        # Index may already exist
        pass

    return len(to_write)
