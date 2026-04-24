from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Chunk:
    source_type: str
    source_path: str
    page: Optional[int]
    row_idx: Optional[int]
    text: str
    meta: Dict
    chunk_id: int


def chunk_text_records(records: List[Dict], chunk_chars: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for rec in records:
        text = rec.get("text", "") or ""
        if not text.strip():
            continue
        start = 0
        part_id = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            piece = text[start:end]
            chunks.append(Chunk(
                source_type=rec.get("source_type", "unknown"),
                source_path=rec.get("source_path", ""),
                page=rec.get("page"),
                row_idx=rec.get("row_idx"),
                text=piece,
                meta=rec.get("meta", {}),
                chunk_id=part_id,
            ))
            if end >= len(text):
                break
            start = max(0, end - overlap)
            part_id += 1
    return chunks

