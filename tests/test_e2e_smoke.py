import os


def test_prompt_assembly_with_mock_asr(monkeypatch):
    from src.utils.prompt import build_instructions

    ctx = [
        {"text": "Pool hours are 07:00–20:00.", "source_type": "csv", "source_path": "source.csv", "row_idx": 5, "page": None}
    ]
    user = "What are the pool hours?"
    instr = build_instructions(ctx, user)
    assert isinstance(instr, str)
    assert "CONTEXT:" in instr and "USER:" in instr

    # Simulate well-formed response.create payload
    payload = {
        "type": "response.create",
        "response": {"instructions": instr}
    }
    assert payload["response"]["instructions"].endswith(user)
