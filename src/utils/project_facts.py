from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from src.utils.catalog import normalize_token


FACTS_CATALOG_PATH = Path("data/catalog/project_facts.yaml")


def _aliases(values: Iterable[str] | None) -> Tuple[str, ...]:
    if not values:
        return tuple()
    return tuple(normalize_token(v) for v in values if v)


@dataclass(slots=True)
class FactEntry:
    id: str
    label: str
    aliases: Tuple[str, ...]
    text: str
    bullets: Tuple[str, ...] = field(default_factory=tuple)

    def matches(self, token: str) -> bool:
        norm = normalize_token(token)
        return norm == self.id or norm in self.aliases


@dataclass(slots=True)
class FactSection:
    id: str
    label: str
    aliases: Tuple[str, ...]
    entries: Dict[str, FactEntry] = field(default_factory=dict)

    def matches(self, token: str) -> bool:
        norm = normalize_token(token)
        if norm == self.id or norm in self.aliases:
            return True
        return any(alias and alias in norm for alias in self.aliases)


@dataclass(slots=True)
class ProjectFactsCatalog:
    version: int
    disclaimer: Optional[str]
    sections: Dict[str, FactSection]

    def find_section(self, key: Optional[str]) -> Optional[FactSection]:
        if not key:
            return None
        norm = normalize_token(key)
        for section in self.sections.values():
            if section.id == norm or norm in section.aliases:
                return section
        return None


def _build_entry(raw: Dict) -> FactEntry:
    bullets = tuple(raw.get("bullets") or [])
    return FactEntry(
        id=normalize_token(raw["id"]),
        label=raw.get("label", raw["id"]),
        aliases=_aliases(raw.get("aliases")),
        text=str(raw.get("text", "")).strip(),
        bullets=tuple(bullets),
    )


def _build_section(raw: Dict) -> FactSection:
    entries = {
        entry.id: entry
        for entry in (_build_entry(e) for e in raw.get("entries", []))
    }
    return FactSection(
        id=normalize_token(raw["id"]),
        label=raw.get("label", raw["id"]),
        aliases=_aliases(raw.get("aliases")),
        entries=entries,
    )


def _load_catalog(path: Path) -> ProjectFactsCatalog:
    if not path.exists():
        raise FileNotFoundError(f"Project facts catalog missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    sections = {
        section.id: section
        for section in (_build_section(s) for s in data.get("sections", []))
    }
    version = int(data.get("version", 1))
    disclaimer = (data.get("metadata") or {}).get("disclaimer")
    return ProjectFactsCatalog(version=version, disclaimer=disclaimer, sections=sections)


@lru_cache(maxsize=1)
def get_project_facts_catalog(path: str | Path | None = None) -> ProjectFactsCatalog:
    catalog_path = Path(path) if path else FACTS_CATALOG_PATH
    return _load_catalog(catalog_path)


def resolve_fact_entry(section: FactSection, entry_key: Optional[str], question: Optional[str] = None) -> Optional[FactEntry]:
    if not section.entries:
        return None
    if entry_key:
        target = section.entries.get(normalize_token(entry_key))
        if target:
            return target
        for entry in section.entries.values():
            if entry.matches(entry_key):
                return entry
    if question:
        normalized_question = normalize_token(question)
        for entry in section.entries.values():
            if entry.id in normalized_question:
                return entry
            if any(alias and alias in normalized_question for alias in entry.aliases):
                return entry
    if len(section.entries) == 1:
        return next(iter(section.entries.values()))
    return None


def fuzzy_find_entry(catalog: ProjectFactsCatalog, question: str) -> Optional[Tuple[FactSection, FactEntry]]:
    normalized = normalize_token(question)
    for section in catalog.sections.values():
        if section.matches(question) or section.id in normalized:
            entry = resolve_fact_entry(section, None, question)
            if entry:
                return section, entry
        for entry in section.entries.values():
            if entry.id in normalized or any(alias and alias in normalized for alias in entry.aliases):
                return section, entry
    return None


__all__ = [
    "FactEntry",
    "FactSection",
    "ProjectFactsCatalog",
    "get_project_facts_catalog",
    "resolve_fact_entry",
    "fuzzy_find_entry",
]
