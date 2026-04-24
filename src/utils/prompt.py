from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from src.utils.config import config


_FRENCH_MONTHS = {
    1: "janvier",
    2: "fevrier",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "aout",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "decembre",
}


def _tag(snippet: Dict) -> str:
    st = snippet.get("source_type") or ""
    sp = snippet.get("source_path") or ""
    page = snippet.get("page")
    row = snippet.get("row_idx")
    if st == "pdf" and page:
        return f"[{_fname(sp)} p.{page}]"
    if st == "csv" and row is not None:
        return f"[{_fname(sp)} r.{row}]"
    return f"[{_fname(sp)}]"


def _fname(path: str) -> str:
    return path.split("/")[-1]


def build_temporal_guidance(now: datetime | None = None) -> str:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    month_name = _FRENCH_MONTHS[current.month]
    return (
        f"Date de reference: nous sommes en {month_name} {current.year}. "
        "Si une date de livraison mentionnee est anterieure a cette date, parle au passe et dis explicitement que le projet est deja livre "
        "(ex: 'deja livre en decembre 2025'). Si la date est future, parle au futur."
    )


def build_instructions(context_snippets: List[Dict], user_text: str, clamp_chars: int = 3000) -> str:
    sys = config.retrieval.session_system_prompt.strip()
    temporal_guidance = build_temporal_guidance()
    if not context_snippets:
        return (
            f"{sys}\n\n{temporal_guidance}\n\nCONTEXT:\n(none)\n\nUSER:\n{user_text.strip()}"
        )

    # Compact snippets, truncate long pieces
    parts: List[str] = []
    budget = max(1000, clamp_chars - len(sys) - len(user_text) - 128)
    for s in context_snippets:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        tag = _tag(s)
        # soft truncation per snippet
        t = t[: min(len(t), 800)]
        piece = f"{tag} {t}"
        if len("\n\n".join(parts + [piece])) > budget:
            break
        parts.append(piece)

    ctx = "\n\n".join(parts)
    return (
        f"{sys}\n\n{temporal_guidance}\n\nCONTEXT:\n{ctx}\n\nUSER:\n{user_text.strip()}"
    )
