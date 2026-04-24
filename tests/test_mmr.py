def test_mmr_select_prefers_diverse_relevant():
    import sys, types
    # Stub lancedb client import so pipeline can load without lancedb
    fake_mod = types.ModuleType("app.retrieval.lancedb_client")
    setattr(fake_mod, "search", lambda qv, k: [])
    sys.modules["app.retrieval.lancedb_client"] = fake_mod
    from src.retrieval.pipeline import _mmr_select

    # Build synthetic normalized vectors in 3D for simplicity
    import math
    def norm(v):
        s = math.sqrt(sum(x*x for x in v))
        return [x/s for x in v]

    q = norm([1.0, 0.0, 0.0])
    # Two very similar docs to each other near query
    d1 = norm([0.9, 0.1, 0.0])
    d2 = norm([0.9, 0.09, 0.0])
    # One slightly less similar but orthogonal to avoid redundancy
    d3 = norm([0.7, 0.0, 0.7])

    cands = [
        {"vector": d1, "text": "A"},
        {"vector": d2, "text": "B"},
        {"vector": d3, "text": "C"},
    ]

    sel = _mmr_select(q, cands, top_k=2, lambda_mult=0.5)
    assert len(sel) == 2
    texts = {s["text"] for s in sel}
    # Expect it to include one of (A,B) and also C for diversity
    assert "C" in texts
    assert ("A" in texts) or ("B" in texts)
