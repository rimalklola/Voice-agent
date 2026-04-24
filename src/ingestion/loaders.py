from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except Exception:
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            escapechar = None
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _D()


def load_csv_rows(path: str | Path) -> List[Dict]:
    p = Path(path)
    rows: List[Dict] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", newline="") as f:
        head = f.read(4096)
        f.seek(0)
        dialect = _detect_dialect(head)
        reader = csv.DictReader(f, dialect=dialect)
        for idx, row in enumerate(reader):
            # Stringify as "col: value" lines
            parts = []
            extras_text = ""
            for k, v in row.items():
                if k is None:
                    # extra columns beyond header; merge into free text
                    if isinstance(v, list):
                        extras_text = " ".join([str(x) for x in v if x is not None])
                    elif v is not None:
                        extras_text = str(v)
                    continue
                v_str = "" if v is None else str(v)
                parts.append(f"{k}: {v_str}")
            text = "\n".join(parts)
            if extras_text:
                text = text + "\n" + extras_text
            rows.append({
                "source_type": "csv",
                "source_path": str(p),
                "row_idx": idx,
                "page": None,
                "text": text,
                "meta": {},
            })
    return rows


def load_pdf_pages(path: str | Path) -> List[Dict]:
    p = Path(path)
    pages: List[Dict] = []
    if not p.exists():
        return pages
    reader = PdfReader(str(p))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({
            "source_type": "pdf",
            "source_path": str(p),
            "page": i,
            "row_idx": None,
            "text": text,
            "meta": {},
        })
    return pages


def load_txt_documents(path: str | Path) -> List[Dict]:
    p = Path(path)
    docs: List[Dict] = []
    if not p.exists():
        return docs
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        text = ""
    docs.append({
        "source_type": "txt",
        "source_path": str(p),
        "page": None,
        "row_idx": None,
        "text": text,
        "meta": {},
    })
    return docs
