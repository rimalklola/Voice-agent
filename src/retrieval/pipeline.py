from __future__ import annotations

import time
from typing import Dict, List
import unicodedata
import re
import logging

from src.utils.config import config
from src.ingestion.embedder import get_embedder
from src.retrieval.lancedb_client import search

logger = logging.getLogger(__name__)


def _normalize_text(t: str) -> str:
    t = t.lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _char_ngrams(t: str, n: int = 4) -> List[str]:
    t = _normalize_text(t).replace(" ", "")
    if len(t) < n:
        return [t]
    return [t[i:i+n] for i in range(len(t) - n + 1)]


def _lexical_overlap(query: str, text: str) -> float:
    # Accent-folded, lowercase, alnum-only char-4grams overlap ratio
    qg = set(_char_ngrams(query))
    if not qg:
        return 0.0
    dg = set(_char_ngrams(text))
    if not dg:
        return 0.0
    inter = len(qg & dg)
    return inter / max(1, len(qg))


def _mmr_select(query_vec: List[float], candidates: List[Dict], top_k: int, lambda_mult: float = 0.5, *, query_text: str = "", lex_w: float = 0.0) -> List[Dict]:
    # Compute cosine similarity (dot) assuming normalized vectors
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    selected: List[int] = []
    cand_vecs = [c.get("vector") for c in candidates]
    sims = [dot(query_vec, v) if v else 0.0 for v in cand_vecs]
    if lex_w > 0 and query_text:
        lex_scores = [_lexical_overlap(query_text, c.get("text") or "") for c in candidates]
        sims = [s + lex_w * l for s, l in zip(sims, lex_scores)]
    # Source-type bias to bring precise CSV facts up slightly
    csv_bias = config.retrieval.csv_bias
    pdf_bias = config.retrieval.pdf_bias
    biases = []
    for c in candidates:
        st = (c.get("source_type") or "").lower()
        b = 0.0
        if st == "csv":
            b += csv_bias
        elif st == "pdf":
            b += pdf_bias
        biases.append(b)
    sims = [s + b for s, b in zip(sims, biases)]

    if not candidates:
        return []
    # pick best first
    selected.append(max(range(len(candidates)), key=lambda i: sims[i]))
    while len(selected) < min(top_k, len(candidates)):
        best_i = -1
        best_score = -1e9
        for i in range(len(candidates)):
            if i in selected:
                continue
            # diversity term: max sim to any selected doc
            diversity = max(
                (sum(x * y for x, y in zip(cand_vecs[i], cand_vecs[j])) if cand_vecs[i] and cand_vecs[j] else 0.0)
                for j in selected
            )
            score = lambda_mult * sims[i] - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best_i = i
        if best_i == -1:
            break
        selected.append(best_i)
    return [candidates[i] for i in selected]


def retrieve_context(query_text: str, top_k: int | None = None) -> List[Dict]:
    start = time.time()
    top_k = top_k or config.retrieval.top_k

    if not (query_text or "").strip():
        return []

    t0 = time.perf_counter()
    embedder = get_embedder(config.ingestion.embed_model_name)
    qv = embedder.embed([query_text])[0]
    t_embed = (time.perf_counter() - t0) * 1000.0
    # Retrieve a larger candidate set, then re-rank to the final K
    init_k = max(top_k, config.retrieval.init_k)
    t1 = time.perf_counter()
    candidates = search(qv, init_k)
    t_search = (time.perf_counter() - t1) * 1000.0

    # Convert distances to similarities for readability
    for h in candidates:
        d = h.get("_distance")
        if isinstance(d, (int, float)):
            h["_similarity"] = 1.0 - d

    hits: List[Dict]
    # Optional cross-encoder rerank (if configured)
    ce_model = config.retrieval.cross_encoder_model
    if ce_model:
        try:
            t2 = time.perf_counter()
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(ce_model)
            pairs = [(query_text, (c.get("text") or "")) for c in candidates]
            scores = ce.predict(pairs).tolist()
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            hits = [c for c, _ in ranked[:top_k]]
            t_rerank = (time.perf_counter() - t2) * 1000.0
            logger.debug(
                "retrieval timings with rerank",
                extra={
                    "embed_ms": round(t_embed, 1),
                    "search_ms": round(t_search, 1),
                    "rerank_ms": round(t_rerank, 1),
                    "candidate_count": len(candidates),
                },
            )
        except Exception:
            hits = candidates[:top_k]
    elif config.retrieval.use_mmr:
        lex_w = config.retrieval.lexical_weight if config.retrieval.use_lexical_boost else 0.0
        t2 = time.perf_counter()
        hits = _mmr_select(qv, candidates, top_k, config.retrieval.mmr_lambda, query_text=query_text, lex_w=lex_w)
        t_mmr = (time.perf_counter() - t2) * 1000.0
        logger.debug(
            "retrieval timings with mmr",
            extra={
                "embed_ms": round(t_embed, 1),
                "search_ms": round(t_search, 1),
                "mmr_ms": round(t_mmr, 1),
                "candidate_count": len(candidates),
            },
        )
    else:
        hits = candidates[:top_k]

    results: List[Dict] = []
    for h in hits:
        results.append({
            "text": h.get("text", ""),
            "source_path": h.get("source_path", ""),
            "source_type": h.get("source_type", ""),
            "page": h.get("page"),
            "row_idx": h.get("row_idx"),
            "distance": h.get("_distance"),
            "similarity": h.get("_similarity"),
        })

    latency_ms = int((time.time() - start) * 1000)
    # Minimal inline logging without importing logging module here
    logger.debug(
        "retrieval summary",
        extra={
            "query_length": len(query_text),
            "top_k": top_k,
            "candidate_count": len(candidates),
            "hit_count": len(results),
            "latency_ms": latency_ms,
        },
    )
    return results
