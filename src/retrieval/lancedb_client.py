from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import lancedb
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    lancedb = None  # type: ignore

from src.utils.config import config


TABLE_NAME = "kb_docs"
_TABLE_CACHE = None  # type: ignore


def connect_table() -> Optional["lancedb.table"]:
    if lancedb is None:
        return None
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    try:
        db = lancedb.connect(config.ingestion.lancedb_path)
        if TABLE_NAME not in db.table_names():
            return None
        _TABLE_CACHE = db.open_table(TABLE_NAME)
        return _TABLE_CACHE
    except Exception:
        return None


def search(query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
    if lancedb is None:
        return []
    table = connect_table()
    if table is None:
        return []
    qb = table.search(query_vector).metric("cosine").limit(top_k)
    # Try to increase recall by raising ef_search if supported
    try:
        qb = qb.with_params({"ef_search": int(config.retrieval.ef_search)})  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        res = qb.to_list()
    except Exception as exc:
        logger.warning("LanceDB search failed; returning empty results", extra={"error": str(exc)})
        global _TABLE_CACHE
        _TABLE_CACHE = None  # force reconnect on next call
        return []
    # res entries contain dict with fields + _distance
    return res
