#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dotenv import load_dotenv
from pathlib import Path


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Query LanceDB kb_docs with a text query.")
    parser.add_argument("query", help="Query text to search for")
    parser.add_argument("--top_k", type=int, default=4, help="Number of results to return")
    parser.add_argument("--show_vectors", action="store_true", help="Print vector dim and first few values")
    parser.add_argument("--verify", action="store_true", help="Compute cosine similarity manually to verify")
    parser.add_argument("--pipeline", action="store_true", help="Use retrieval pipeline (MMR/rerank) instead of raw vector search")
    parser.add_argument("--warm", action="store_true", help="Warm-load the embedder before timing (excludes first-load latency)")
    parser.add_argument("--only-context", "--just-context", dest="only_context", action="store_true", help="Print only context text blocks (no headers/metrics)")
    args = parser.parse_args()

    # Ensure repo root on sys.path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import json
    import math
    from src.utils.config import config
    from src.ingestion.embedder import get_embedder

    if args.pipeline:
        from src.retrieval.pipeline import retrieve_context
        if args.warm:
            # Warm the embedder in this process so we don't include model load time
            from src.ingestion.embedder import get_embedder
            get_embedder(config.ingestion.embed_model_name).embed(["warmup"])
        t0 = time.perf_counter()
        rows = retrieve_context(args.query, args.top_k)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if args.only_context:
            for r in rows:
                t = (r.get("text") or "").strip()
                if t:
                    print(t)
            # keep timing separate from context-only stdout
            print(f"[time] retrieval_ms={dt_ms:.1f}", file=sys.stderr)
            return
        else:
            print(f"[time] retrieval_ms={dt_ms:.1f}")
            print(f"Query: {args.query}")
            print(f"Top {args.top_k} by pipeline re-ranking (MMR/CrossEncoder if enabled):\n")
            for i, r in enumerate(rows, 1):
                dist = r.get("distance")
                sim = r.get("similarity")
                tag = (
                    f"{Path(r.get('source_path','')).name} "
                    f"{'p.'+str(r['page']) if r.get('page') else ('r.'+str(r['row_idx']) if r.get('row_idx') is not None else '')}"
                ).strip()
                text = (r.get("text") or "").strip().replace("\n", " ")
                snippet = text[:160] + ("…" if len(text) > 160 else "")
                if dist is not None and sim is not None:
                    print(f"{i:>2}. dist={dist:.4f} sim={sim:.4f} {tag}")
                else:
                    print(f"{i:>2}. {tag}")
                print(f"    {snippet}")
            return

    import lancedb
    db = lancedb.connect(config.ingestion.lancedb_path)
    table = db.open_table("kb_docs")

    embedder = get_embedder(config.ingestion.embed_model_name)
    if args.warm:
        embedder.embed(["warmup"])  # load model first
    t0 = time.perf_counter()
    qv = embedder.embed([args.query])[0]
    rows = table.search(qv).metric("cosine").limit(args.top_k).to_list()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if args.only_context:
        for r in rows:
            t = (r.get("text") or "").strip()
            if t:
                print(t)
        print(f"[time] retrieval_ms={dt_ms:.1f}", file=sys.stderr)
        return

    print(f"[time] retrieval_ms={dt_ms:.1f}")
    print(f"Query: {args.query}")
    print(f"Top {args.top_k} by cosine distance (lower is better):\n")
    for i, r in enumerate(rows, 1):
        dist = r.get("_distance")
        sim = (1.0 - dist) if isinstance(dist, (int, float)) else None
        tag = (
            f"{Path(r.get('source_path','')).name} "
            f"{'p.'+str(r['page']) if r.get('page') else ('r.'+str(r['row_idx']) if r.get('row_idx') is not None else '')}"
        ).strip()
        text = (r.get("text") or "").strip().replace("\n", " ")
        snippet = text[:160] + ("…" if len(text) > 160 else "")
        print(f"{i:>2}. dist={dist:.4f} sim={sim:.4f} {tag}")
        print(f"    {snippet}")

        if args.show_vectors:
            vec = r.get("vector") or []
            print(f"    vector dim={len(vec)} head={json.dumps(vec[:8])}")

    if args.verify and rows:
        # Manual similarity check: dot(q, v) if both normalized.
        # We assume ingestion normalized vectors; embedder returns normalized vectors as well.
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        sims = []
        for r in rows:
            v = r.get("vector") or []
            sims.append(dot(qv, v))
        print("\nManual cosine similarities (dot products) for returned rows:")
        for i, s in enumerate(sims, 1):
            print(f"  {i:>2}. cos_sim={s:.4f}")


if __name__ == "__main__":
    main()
