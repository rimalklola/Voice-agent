import os
import shutil
import pytest


def test_retrieval_returns_results(tmp_path, monkeypatch):
    from src.ingestion import embedder as embedder_module

    embedder_module.enable_test_mode()
    db_path = tmp_path / "lancedb"
    monkeypatch.setenv("LANCEDB_PATH", str(db_path))

    # Now import modules that read config
    pytest.importorskip("lancedb")
    from src.ingestion.chunking import Chunk
    from src.ingestion.embedder import get_embedder
    from src.ingestion.build_index import upsert_chunks
    from src.retrieval.pipeline import retrieve_context

    # Build a tiny in-memory corpus
    texts = [
        ("csv", "source.csv", None, 0, "Check-in time is 15:00 at Alma Resort."),
        ("pdf", "source.pdf", 1, None, "The main pool is open from 07:00 to 20:00 daily."),
    ]
    chunks = [
        Chunk(source_type=t[0], source_path=t[1], page=t[2], row_idx=t[3], text=t[4], meta={}, chunk_id=i)
        for i, t in enumerate(texts)
    ]
    try:
        embedder = get_embedder("dummy")
        vecs = embedder.embed([c.text for c in chunks])
        written = upsert_chunks(chunks, vecs)
        assert written >= 2

        res = retrieve_context("What is the check-in time?", top_k=4)
        assert isinstance(res, list)
        assert len(res) >= 1
        fields = {"text", "source_path", "page", "row_idx"}
        assert fields.issubset(set(res[0].keys()))
        assert any("Check-in" in r["text"] for r in res)
    finally:
        embedder_module.disable_test_mode()
