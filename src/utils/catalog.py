from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


CATALOG_PATH = Path("data/catalog/property_specs.yaml")


def _norm(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", " ", value.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.replace(" ", "_")


def normalize_token(value: str) -> str:
    return _norm(value)


@dataclass(slots=True)
class AttributeSpec:
    name: str
    label: str
    unit: Optional[str] = None
    value: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    note: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
        }
        if self.unit:
            payload["unit"] = self.unit
        if self.value is not None:
            payload["value"] = self.value
        if self.min is not None:
            payload["min"] = self.min
        if self.max is not None:
            payload["max"] = self.max
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(slots=True)
class VariantSpec:
    id: str
    label: str
    aliases: Tuple[str, ...]
    bedrooms: Optional[int] = None
    suites: Optional[int] = None
    attributes: Dict[str, AttributeSpec] = field(default_factory=dict)

    def matches(self, token: str) -> bool:
        token_norm = _norm(token)
        return token_norm in self.aliases or token_norm == self.id


@dataclass(slots=True)
class CategorySpec:
    id: str
    label: str
    type: str
    aliases: Tuple[str, ...]
    description: Optional[str] = None
    notes: Optional[str] = None
    variants: Dict[str, VariantSpec] = field(default_factory=dict)

    def matches(self, token: str) -> bool:
        token_norm = _norm(token)
        return token_norm == self.id or token_norm in self.aliases


@dataclass(slots=True)
class PropertyCatalog:
    version: int
    disclaimer: Optional[str]
    categories: Dict[str, CategorySpec]

    def get_category(self, key: str | None) -> Optional[CategorySpec]:
        if not key:
            return None
        key_norm = _norm(key)
        for category in self.categories.values():
            if category.matches(key_norm):
                return category
        return None


def _tuple_aliases(values: Iterable[str] | None) -> Tuple[str, ...]:
    if not values:
        return tuple()
    return tuple(_norm(v) for v in values if v)


def _build_attribute(name: str, payload: Dict[str, Any]) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        label=payload.get("label", name.replace("_", " ").title()),
        unit=payload.get("unit"),
        value=payload.get("value"),
        min=payload.get("min"),
        max=payload.get("max"),
        note=payload.get("note"),
    )


def _build_variant(raw: Dict[str, Any]) -> VariantSpec:
    attr_map = {
        name: _build_attribute(name, data)
        for name, data in (raw.get("attributes") or {}).items()
    }
    return VariantSpec(
        id=_norm(raw["id"]),
        label=raw.get("label", raw["id"]),
        aliases=_tuple_aliases(raw.get("aliases")),
        bedrooms=raw.get("bedrooms"),
        suites=raw.get("suites"),
        attributes=attr_map,
    )


def _build_category(raw: Dict[str, Any]) -> CategorySpec:
    variants = {
        variant.id: variant
        for variant in (_build_variant(v) for v in raw.get("variants", []))
    }
    return CategorySpec(
        id=_norm(raw["id"]),
        label=raw.get("label", raw["id"]),
        type=raw.get("type", "other"),
        aliases=_tuple_aliases(raw.get("aliases")),
        description=raw.get("description"),
        notes=raw.get("notes"),
        variants=variants,
    )


def _load_catalog_from_disk(path: Path) -> PropertyCatalog:
    if not path.exists():
        raise FileNotFoundError(f"Catalog file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    categories = {
        cat.id: cat
        for cat in (_build_category(c) for c in data.get("categories", []))
    }
    return PropertyCatalog(
        version=int(data.get("version", 1)),
        disclaimer=data.get("metadata", {}).get("disclaimer"),
        categories=categories,
    )


@lru_cache(maxsize=1)
def get_catalog(path: str | Path | None = None) -> PropertyCatalog:
    catalog_path = Path(path) if path else CATALOG_PATH
    return _load_catalog_from_disk(catalog_path)


def resolve_variant(category: CategorySpec, variant_key: Optional[str], question: Optional[str] = None) -> Optional[VariantSpec]:
    if not category.variants:
        return None
    if variant_key:
        candidate = category.variants.get(_norm(variant_key))
        if candidate:
            return candidate
        # alias lookup
        for variant in category.variants.values():
            if variant.matches(variant_key):
                return variant
    if len(category.variants) == 1:
        return next(iter(category.variants.values()))
    if question:
        question_norm = _norm(question)
        for variant in category.variants.values():
            for alias in (variant.aliases or ()):  # type: ignore[assignment]
                if alias and alias in question_norm:
                    return variant
        # fallback: check literal ids in text (e.g., "f3")
        for variant in category.variants.values():
            if variant.id in question_norm:
                return variant
    return None


__all__ = [
    "AttributeSpec",
    "VariantSpec",
    "CategorySpec",
    "PropertyCatalog",
    "get_catalog",
    "resolve_variant",
    "normalize_token",
]
