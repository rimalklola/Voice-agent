#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

import sys
from dotenv import load_dotenv

# Ensure repo root is importable when run from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import config
from src.ingestion.loaders import load_csv_rows, load_pdf_pages, load_txt_documents
from src.ingestion.chunking import chunk_text_records
from src.ingestion.embedder import get_embedder
from src.ingestion.build_index import upsert_chunks


def _csv_targets() -> list[Path]:
    base = Path(config.ingestion.doc_csv_path)
    if base.suffix.lower() == ".csv" or base.name.lower().endswith(".csv"):
        base.parent.mkdir(parents=True, exist_ok=True)
        return [base]
    base.mkdir(parents=True, exist_ok=True)
    return sorted(base.glob("*.csv"))


def _txt_targets() -> list[Path]:
    base = Path(config.ingestion.doc_txt_path)
    if base.suffix.lower() == ".txt" or base.name.lower().endswith(".txt"):
        base.parent.mkdir(parents=True, exist_ok=True)
        return [base]
    base.mkdir(parents=True, exist_ok=True)
    return sorted(base.glob("*.txt"))


def ensure_sample_docs():
    # Create sample CSV/PDF if missing to ease onboarding
    csv_targets = _csv_targets()
    if not csv_targets:
        sample_csv = Path(config.ingestion.doc_csv_path)
        if sample_csv.suffix.lower() != ".csv":
            sample_csv = sample_csv / "sample.csv"
        sample_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_csv.write_text(
            "title,info\n"
            "Check-in,Check-in time is 15:00 at Alma Resort.\n"
            "Pool Hours,The main pool is open from 07:00 to 20:00 daily.\n",
            encoding="utf-8",
        )
        csv_targets = [sample_csv]
        print(f"[ingest] Created sample CSV at {sample_csv}")

    txt_targets = _txt_targets()
    if not txt_targets:
        sample_txt = Path(config.ingestion.doc_txt_path)
        if sample_txt.suffix.lower() != ".txt":
            sample_txt = sample_txt / "sample.txt"
        sample_txt.parent.mkdir(parents=True, exist_ok=True)
        sample_txt.write_text(
            "Alma Resort notes:\n- Villas disponibles\n- Spa ouvert de 10h à 18h\n",
            encoding="utf-8",
        )
        txt_targets = [sample_txt]
        print(f"[ingest] Created sample TXT at {sample_txt}")

    pdf_p = Path(config.ingestion.doc_pdf_path)
    pdf_p.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_p.exists():
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(pdf_p), pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(72, 720, "Alma Resort and Spa – Amenities Guide")
            c.drawString(72, 700, "Spa hours: 10:00–18:00. Contact front desk to book.")
            c.showPage()
            c.setFont("Helvetica", 12)
            c.drawString(72, 720, "Dining – The Ocean Restaurant")
            c.drawString(72, 700, "Breakfast served 07:00–10:30. Reservations recommended on weekends.")
            c.save()
            print(f"[ingest] Created sample PDF at {pdf_p}")
        except Exception as e:
            print(f"[ingest] Could not create sample PDF at {pdf_p}: {e}")
    return csv_targets, txt_targets


def main():
    load_dotenv()
    start = time.time()
    csv_files, txt_files = ensure_sample_docs()

    csv_rows = []
    total_rows = 0
    for csv_file in csv_files:
        rows = load_csv_rows(csv_file)
        total_rows += len(rows)
        csv_rows.extend(rows)
        print(f"[ingest] Loaded {len(rows)} rows from {csv_file}")

    txt_docs = []
    for txt_file in txt_files:
        docs = load_txt_documents(txt_file)
        txt_docs.extend(docs)
        print(f"[ingest] Loaded text document from {txt_file}")

    pdf_pages = load_pdf_pages(config.ingestion.doc_pdf_path)
    print(f"[ingest] Loaded CSV rows: {total_rows}, TXT docs: {len(txt_docs)}, PDF pages: {len(pdf_pages)}")

    chunks = chunk_text_records(
        csv_rows + txt_docs + pdf_pages,
        config.ingestion.chunk_chars,
        config.ingestion.chunk_overlap,
    )
    print(f"[ingest] Total chunks: {len(chunks)}")

    embedder = get_embedder(config.ingestion.embed_model_name)
    vectors = embedder.embed([c.text for c in chunks])
    written = upsert_chunks(chunks, vectors)
    print(f"[ingest] Wrote {written} rows to LanceDB at {config.ingestion.lancedb_path}")

    print(f"[ingest] Done in {int((time.time() - start)*1000)} ms")


if __name__ == "__main__":
    main()
