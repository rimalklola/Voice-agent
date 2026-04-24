from __future__ import annotations

import asyncio
import re
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import unicodedata

from src.retrieval.pipeline import retrieve_context
from src.utils.call_ledger import insert_reservation_record, reservation_exists
from src.utils.config import config
from src.utils.catalog import (
    get_catalog,
    resolve_variant,
    normalize_token,
    VariantSpec,
    AttributeSpec,
)
from src.utils.project_facts import (
    get_project_facts_catalog,
    resolve_fact_entry,
    fuzzy_find_entry,
)

_reservations_cfg = config.guardrails.reservations
_monitoring_cfg = config.get("monitoring", {})
_knowledge_cfg = config.guardrails.knowledge

CALL_LEDGER_SQLITE_PATH = str((_monitoring_cfg or {}).get("sqlite_path", "./data/monitoring/call_ledger.sqlite3"))
RESERVATION_RATE_LIMIT = _reservations_cfg.rate_limit
RESERVATION_RATE_WINDOW = _reservations_cfg.rate_window
RESERVATION_DUPLICATE_WINDOW = _reservations_cfg.duplicate_window
RESERVATION_MIN_PHONE_DIGITS = _reservations_cfg.min_phone_digits
RESERVATION_MAX_PHONE_DIGITS = _reservations_cfg.max_phone_digits
RESERVATION_MAX_NOTES_LENGTH = _reservations_cfg.max_notes_length

PROJECT_INFO_MIN_LENGTH = _knowledge_cfg.min_length
PROJECT_INFO_MAX_QUERY_LENGTH = _knowledge_cfg.max_query_length
PROJECT_INFO_RATE_LIMIT = _knowledge_cfg.rate_limit
PROJECT_INFO_RATE_WINDOW = _knowledge_cfg.rate_window
PROJECT_INFO_DUPLICATE_WINDOW = _knowledge_cfg.duplicate_window
PROJECT_INFO_BLOCKLIST = tuple(_knowledge_cfg.blocklist)

_RESERVATION_LOCK = threading.Lock()
_PROJECT_INFO_LOCK = threading.Lock()
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_ALLOWED_RE = re.compile(r"^[+0-9\s().-]{6,}$")
_SUSPICIOUS_INPUT_RE = re.compile(
    r"[{}<>$]|--|\b(?:drop|delete|insert|update|select|http|https|ftp)\b",
    re.IGNORECASE,
)

_RESERVATION_ATTEMPTS: deque[float] = deque()
_RESERVATION_RECENT: Dict[str, float] = {}
_PROJECT_INFO_ATTEMPTS: deque[float] = deque()
_PROJECT_INFO_RECENT: Dict[str, float] = {}
_RESERVATION_CONFIRM_FAILS: Dict[str, int] = {}
MAX_CONFIRM_RETRIES = 2
PROPERTY_SPECS_MAX_ATTR = 6
_MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "janvier": 1,
    "feb": 2,
    "fev": 2,
    "fevr": 2,
    "fevrier": 2,
    "mar": 3,
    "mars": 3,
    "apr": 4,
    "avr": 4,
    "avril": 4,
    "may": 5,
    "mai": 5,
    "jun": 6,
    "juin": 6,
    "jul": 7,
    "juil": 7,
    "juillet": 7,
    "aug": 8,
    "aou": 8,
    "aout": 8,
    "sept": 9,
    "sep": 9,
    "septembre": 9,
    "oct": 10,
    "octobre": 10,
    "nov": 11,
    "novembre": 11,
    "dec": 12,
    "decembre": 12,
    "december": 12,
}
_DELIVERY_DATE_RE = re.compile(r"(?P<qualifier>fin\s+)?(?P<month>[A-Za-zÀ-ÿ]{3,12})[-\s]+(?P<year>\d{2,4})", re.IGNORECASE)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _parse_delivery_month_year(text: str) -> Optional[tuple[int, int, str]]:
    for match in _DELIVERY_DATE_RE.finditer(text or ""):
        raw_month = match.group("month")
        raw_year = match.group("year")
        month_key = _strip_accents(raw_month).lower().rstrip(".")
        month = _MONTH_NAME_TO_NUMBER.get(month_key)
        if month is None:
            continue
        year = int(raw_year)
        if len(raw_year) == 2:
            year += 2000
        return year, month, match.group(0)
    return None


def _is_past_delivery_date(text: str, now: Optional[datetime] = None) -> bool:
    parsed = _parse_delivery_month_year(text)
    if not parsed:
        return False
    year, month, _ = parsed
    ref = (now or _utcnow()).astimezone(timezone.utc)
    return (year, month) < (ref.year, ref.month)


def _replace_if_past(match: re.Match[str], replacement_factory, now: datetime) -> str:
    original = match.group(0)
    date_text = match.group("date")
    if not _is_past_delivery_date(date_text, now=now):
        return original
    return replacement_factory(match)


def _normalize_delivery_timeline_text(text: str, now: Optional[datetime] = None) -> str:
    if not text:
        return text

    ref_now = (now or _utcnow()).astimezone(timezone.utc)
    date_pattern = r"(?P<date>(?:fin\s+)?[A-Za-zÀ-ÿ]{3,12}[-\s]+\d{2,4})"

    def replace_delivery_date_sentence(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or "la livraison"
        date_text = match.group("date")
        return f"{prefix} a déjà eu lieu en {date_text}"

    text = re.sub(
        rf"(?P<prefix>la\s+date\s+de\s+livraison(?:\s+des\s+[^,.;:]+)?|la\s+livraison(?:\s+des\s+[^,.;:]+)?)\s+(?:est|sera)\s+pr[ée]vu(?:e|s|es)?\s+(?:en\s+)?{date_pattern}",
        lambda match: _replace_if_past(match, replace_delivery_date_sentence, ref_now),
        text,
        flags=re.IGNORECASE,
    )

    def replace_future_delivered(match: re.Match[str]) -> str:
        aux = (match.group("aux") or "").lower()
        participle = match.group("participle")
        date_text = match.group("date")
        qualifier = match.group("qualifier") or ""
        past_aux = "ont" if aux == "seront" else "a"
        return f"{past_aux} déjà été {participle} {qualifier}{date_text}".strip()

    text = re.sub(
        rf"(?P<aux>sera|seront)\s+(?P<participle>livr(?:é|ée|és|ées))\s+(?P<qualifier>fin\s+)?{date_pattern}",
        lambda match: _replace_if_past(match, replace_future_delivered, ref_now),
        text,
        flags=re.IGNORECASE,
    )

    def replace_planned(match: re.Match[str]) -> str:
        planned = _strip_accents(match.group("planned")).lower()
        date_text = match.group("date")
        delivered = {
            "prevu": "déjà livré en",
            "prevue": "déjà livrée en",
            "prevus": "déjà livrés en",
            "prevues": "déjà livrées en",
        }.get(planned, "déjà livré en")
        return f"{delivered} {date_text}"

    text = re.sub(
        rf"(?P<planned>pr[ée]vu(?:e|s|es)?)\s+(?:en\s+)?{date_pattern}",
        lambda match: _replace_if_past(match, replace_planned, ref_now),
        text,
        flags=re.IGNORECASE,
    )

    def replace_bare_delivered(match: re.Match[str]) -> str:
        participle = match.group("participle")
        qualifier = match.group("qualifier") or ""
        date_text = match.group("date")
        return f"déjà {participle} {qualifier}{date_text}".strip()

    text = re.sub(
        rf"(?<!déjà\s)(?<!ete\s)(?<!été\s)(?P<participle>livr(?:é|ée|és|ées))\s+(?P<qualifier>fin\s+)?{date_pattern}",
        lambda match: _replace_if_past(match, replace_bare_delivered, ref_now),
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", text).strip()


def _normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D+", "", phone)
    if not digits:
        return ""
    return ("+" if phone.strip().startswith("+") else "") + digits


def _append_reservation_row(row: Dict[str, Any]) -> None:
    insert_reservation_record(CALL_LEDGER_SQLITE_PATH, row)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "oui"}
    return False


def _register_reservation_attempt() -> Optional[float]:
    if RESERVATION_RATE_LIMIT <= 0:
        return None
    now = time.monotonic()
    with _RESERVATION_LOCK:
        while _RESERVATION_ATTEMPTS and (now - _RESERVATION_ATTEMPTS[0]) > RESERVATION_RATE_WINDOW:
            _RESERVATION_ATTEMPTS.popleft()
        if len(_RESERVATION_ATTEMPTS) >= RESERVATION_RATE_LIMIT:
            return None
        _RESERVATION_ATTEMPTS.append(now)
        return now


def _rollback_reservation_attempt(token: Optional[float]) -> None:
    if token is None or RESERVATION_RATE_LIMIT <= 0:
        return
    with _RESERVATION_LOCK:
        try:
            _RESERVATION_ATTEMPTS.remove(token)
        except ValueError:
            pass


def _normalize_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_request_type(value: Any) -> str:
    normalized = _normalize_lower(value)
    aliases = {
        "visite": "visite",
        "visit": "visite",
        "rappel": "rappel",
        "callback": "rappel",
        "brochure": "brochure",
    }
    return aliases.get(normalized, "rappel")


def _reservation_key(phone: str, date: str, time_str: str, request_type: str) -> str:
    return "|".join([
        _normalize_lower(_normalize_phone(phone)),
        _normalize_lower(date),
        _normalize_lower(time_str),
        _normalize_request_type(request_type),
    ])


def _purge_stale_reservations(now: float) -> None:
    if RESERVATION_DUPLICATE_WINDOW <= 0:
        _RESERVATION_RECENT.clear()
        return
    stale = [key for key, ts in _RESERVATION_RECENT.items() if (now - ts) > RESERVATION_DUPLICATE_WINDOW]
    for key in stale:
        _RESERVATION_RECENT.pop(key, None)


def _reservation_exists_in_csv(phone: str, date: str, time_str: str, request_type: str) -> bool:
    return reservation_exists(
        CALL_LEDGER_SQLITE_PATH,
        phone=_normalize_phone(phone),
        requested_date=_normalize_lower(date),
        requested_time=_normalize_lower(time_str),
        request_type=_normalize_request_type(request_type),
    )


async def _has_recent_reservation(phone: str, date: str, time_str: str, request_type: str) -> bool:
    if RESERVATION_DUPLICATE_WINDOW <= 0:
        return False

    key = _reservation_key(phone, date, time_str, request_type)
    now = time.monotonic()

    with _RESERVATION_LOCK:
        _purge_stale_reservations(now)
        in_memory = key in _RESERVATION_RECENT
    if in_memory:
        return True

    exists_in_csv = await asyncio.to_thread(_reservation_exists_in_csv, phone, date, time_str, request_type)
    if exists_in_csv:
        with _RESERVATION_LOCK:
            _RESERVATION_RECENT[key] = now
        return True

    return False


def _record_reservation(phone: str, date: str, time_str: str, request_type: str) -> None:
    if RESERVATION_DUPLICATE_WINDOW <= 0:
        return
    key = _reservation_key(phone, date, time_str, request_type)
    now = time.monotonic()
    with _RESERVATION_LOCK:
        _purge_stale_reservations(now)
        _RESERVATION_RECENT[key] = now


def _confirmation_key(full_name: str, date: str, time_str: str) -> str:
    parts = [
        _normalize_lower(full_name),
        _normalize_lower(date),
        _normalize_lower(time_str),
    ]
    return "|".join(parts)


def _register_confirm_failure(key: str) -> int:
    if not key:
        return MAX_CONFIRM_RETRIES
    with _RESERVATION_LOCK:
        count = _RESERVATION_CONFIRM_FAILS.get(key, 0) + 1
        _RESERVATION_CONFIRM_FAILS[key] = count
        return count


def _clear_confirm_failure(key: str) -> None:
    if not key:
        return
    with _RESERVATION_LOCK:
        _RESERVATION_CONFIRM_FAILS.pop(key, None)


def _fallback_reservation_message(full_name: str, human_when: str, phone_display: str, request_type: str) -> str:
    request_labels = {
        "visite": "demande de visite",
        "rappel": "demande de rappel",
        "brochure": "demande de brochure",
    }
    request_label = request_labels.get(_normalize_request_type(request_type), "demande")
    when_clause = f" {human_when}" if human_when else ""
    base = f"Merci {full_name}. J’ai bien noté votre {request_label}{when_clause}."
    if phone_display:
        return base + f" Nous vous confirmerons par téléphone au {phone_display}."
    return base + " Une conseillère vous recontactera rapidement pour confirmer vos coordonnées."


async def _finalize_with_fallback(
    call_id: str,
    full_name: str,
    email: str,
    phone: str,
    phone_display: str,
    lead_source: str,
    request_type: str,
    date_str: str,
    time_str: str,
    guests: int,
    notes: str,
    human_when: str,
    fallback_key: str,
) -> Dict[str, Any]:
    fallback_note = "Auto: coordonnées à confirmer (validation échouée)"
    merged_notes = notes.strip()
    if merged_notes:
        merged_notes = merged_notes + " | " + fallback_note
    else:
        merged_notes = fallback_note

    row = {
        "call_id": str(call_id or ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "phone_display": phone_display,
        "date": date_str,
        "time": time_str,
        "guests": guests,
        "notes": merged_notes,
        "source": "twilio",
        "lead_source": lead_source,
        "request_type": _normalize_request_type(request_type),
        "status": "ok",
        "fallback_used": 1,
        "requested_date": date_str,
        "requested_time": time_str,
    }

    try:
        await asyncio.to_thread(_append_reservation_row, row)
    except Exception:
        return {
            "status": "error",
            "answer": "Désolée, j’ai eu un problème pour enregistrer la réservation. Voulez-vous réessayer ?"
        }

    _record_reservation(phone_display or phone, date_str, time_str, request_type)
    _clear_confirm_failure(fallback_key)

    message = _fallback_reservation_message(full_name, human_when, phone_display or phone, request_type)
    message += " Merci pour votre patience, une collègue vérifiera vos coordonnées si besoin."

    return {
        "status": "ok",
        "answer": message,
        "fallback": True,
    }


async def ask_to_reserve(args: Dict[str, Any]) -> Dict[str, Any]:
    call_id = str(args.get("call_id") or "").strip()
    full_name = str(args.get("full_name") or "").strip()
    email = str(args.get("email") or "").strip()
    phone_input = str(args.get("phone") or "").strip()
    lead_source = str(args.get("lead_source") or "").strip()
    request_type = _normalize_request_type(args.get("request_type"))
    date_str = str(args.get("date") or "").strip()
    time_str = str(args.get("time") or "").strip()
    notes = str(args.get("notes") or "").strip()

    try:
        guests = int(args.get("guests") or 0)
    except (TypeError, ValueError):
        guests = 0

    confirmed_raw = args.get("confirmed")
    override_duplicate_raw = args.get("override_duplicate")
    confirmed = _parse_bool(confirmed_raw)
    override_duplicate = _parse_bool(override_duplicate_raw)

    if date_str and time_str:
        human_when = f"le {date_str} à {time_str}"
    elif date_str:
        human_when = f"le {date_str}"
    else:
        human_when = ""

    guests_text = f" pour {guests} personne(s)" if guests else ""
    action_label = {
        "visite": "visite",
        "rappel": "rappel",
        "brochure": "brochure",
    }.get(request_type, "rappel")

    required_fields = ["full_name", "phone", "lead_source"]
    missing = [name for name, value in {
        "full_name": full_name,
        "phone": phone_input,
        "lead_source": lead_source,
    }.items() if not value]

    if confirmed_raw is None:
        missing.append("confirmed")

    fallback_key = _confirmation_key(full_name, date_str, time_str)

    if missing:
        if confirmed:
            if _register_confirm_failure(fallback_key) >= MAX_CONFIRM_RETRIES:
                return await _finalize_with_fallback(
                    call_id,
                    full_name,
                    email,
                    "",
                    phone_input,
                    lead_source,
                    request_type,
                    date_str,
                    time_str,
                    guests,
                    notes,
                    human_when,
                    fallback_key,
                )
        msg = (
            "Pour finaliser la réservation, il me manque: "
            + ", ".join(missing)
            + ". Pouvez-vous me confirmer clairement ces informations ?"
        )
        return {"status": "missing_fields", "missing": missing, "answer": msg}

    if not confirmed:
        when_clause = f", {human_when}" if human_when else ""
        preview = (
            f"Récapitulatif: {full_name}, téléphone {phone_input}, source {lead_source}, "
            f"demande de {action_label}{when_clause}{guests_text}. Confirmez-vous que tout est correct ?"
        )
        return {"status": "needs_confirmation", "answer": preview}

    invalid: List[str] = []
    if email and not _EMAIL_RE.match(email):
        invalid.append("email")
    phone = _normalize_phone(phone_input)
    digits = re.sub(r"\D+", "", phone_input)
    if (
        not _PHONE_ALLOWED_RE.match(phone_input)
        or len(digits) < RESERVATION_MIN_PHONE_DIGITS
        or len(digits) > RESERVATION_MAX_PHONE_DIGITS
    ):
        invalid.append("phone")

    if notes and len(notes) > RESERVATION_MAX_NOTES_LENGTH:
        invalid.append("notes_length")

    suspicious_fields = [
        name
        for name, value in {
            "full_name": full_name,
            "email": email,
            "phone": phone_input,
            "lead_source": lead_source,
            "request_type": request_type,
            "date": date_str,
            "time": time_str,
            "notes": notes,
        }.items()
        if _is_suspicious_text(value)
    ]
    if suspicious_fields:
        invalid.extend(f"suspicious_{field}" for field in suspicious_fields if f"suspicious_{field}" not in invalid)

    if invalid:
        if confirmed and _register_confirm_failure(fallback_key) >= MAX_CONFIRM_RETRIES:
            return await _finalize_with_fallback(
                call_id,
                full_name,
                email,
                phone,
                phone_input,
                lead_source,
                request_type,
                date_str,
                time_str,
                guests,
                notes,
                human_when,
                fallback_key,
            )
        msg = (
            "Certaines informations semblent incorrectes ou non autorisées: "
            + ", ".join(sorted(invalid))
            + ". Pouvez-vous me les redonner de manière claire et simple ?"
        )
        return {"status": "invalid_fields", "invalid": sorted(set(invalid)), "answer": msg}

    duplicate_allowed = override_duplicate or config.guardrails.reservations.allow_duplicate_without_prompt
    if not duplicate_allowed:
        duplicate = await _has_recent_reservation(phone, date_str, time_str, request_type)
        if duplicate:
            when_clause = f" {human_when}" if human_when else ""
            duplicate_msg = (
                f"Il existe déjà une demande similaire pour le numéro {phone}{when_clause}. "
                "Si vous souhaitez la modifier, dites-le clairement et je mettrai la fiche à jour."
            )
            return {"status": "duplicate", "answer": duplicate_msg}

    rate_token = _register_reservation_attempt()
    if rate_token is None:
        return {
            "status": "rate_limited",
            "answer": "Je viens de traiter plusieurs réservations. Pour votre sécurité, merci de patienter quelques instants avant de réessayer.",
        }

    row = {
        "call_id": call_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "phone_display": phone_input,
        "date": date_str,
        "time": time_str,
        "guests": guests,
        "notes": notes,
        "source": "twilio",
        "lead_source": lead_source,
        "request_type": request_type,
        "status": "ok",
        "fallback_used": 0,
        "requested_date": date_str,
        "requested_time": time_str,
    }

    try:
        await asyncio.to_thread(_append_reservation_row, row)
    except Exception:
        _rollback_reservation_attempt(rate_token)
        return {
            "status": "error",
            "answer": "Désolée, j’ai eu un problème pour enregistrer la réservation. Voulez-vous réessayer ?"
        }

    _record_reservation(phone, date_str, time_str, request_type)
    _clear_confirm_failure(fallback_key)

    when_clause = f" {human_when}" if human_when else ""
    guests_text = f" pour {guests} personne(s)" if guests else ""
    msg = (
        f"Parfait, {full_name}. J’ai bien noté votre demande de {action_label}{when_clause}{guests_text}. "
        f"Nous vous recontacterons au {phone}. Source notée: {lead_source}."
    )
    if email:
        msg += f" J’ai aussi noté votre email {email}."
    msg += " Est-ce correct ?"
    return {"status": "ok", "answer": msg}


def _register_project_query_attempt() -> Optional[float]:
    if PROJECT_INFO_RATE_LIMIT <= 0:
        return None
    now = time.monotonic()
    with _PROJECT_INFO_LOCK:
        while _PROJECT_INFO_ATTEMPTS and (now - _PROJECT_INFO_ATTEMPTS[0]) > PROJECT_INFO_RATE_WINDOW:
            _PROJECT_INFO_ATTEMPTS.popleft()
        if len(_PROJECT_INFO_ATTEMPTS) >= PROJECT_INFO_RATE_LIMIT:
            return None
        _PROJECT_INFO_ATTEMPTS.append(now)
        return now


def _rollback_project_query_attempt(token: Optional[float]) -> None:
    if token is None or PROJECT_INFO_RATE_LIMIT <= 0:
        return
    with _PROJECT_INFO_LOCK:
        try:
            _PROJECT_INFO_ATTEMPTS.remove(token)
        except ValueError:
            pass


def _project_query_key(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    return normalized[:PROJECT_INFO_MAX_QUERY_LENGTH]


def _purge_project_queries(now: float) -> None:
    if PROJECT_INFO_DUPLICATE_WINDOW <= 0:
        _PROJECT_INFO_RECENT.clear()
        return
    stale = [
        key for key, ts in _PROJECT_INFO_RECENT.items()
        if (now - ts) > PROJECT_INFO_DUPLICATE_WINDOW
    ]
    for key in stale:
        _PROJECT_INFO_RECENT.pop(key, None)


def _is_recent_project_query(query: str) -> bool:
    if PROJECT_INFO_DUPLICATE_WINDOW <= 0:
        return False
    key = _project_query_key(query)
    now = time.monotonic()
    with _PROJECT_INFO_LOCK:
        _purge_project_queries(now)
        return key in _PROJECT_INFO_RECENT


def _record_project_query(query: str) -> None:
    if PROJECT_INFO_DUPLICATE_WINDOW <= 0:
        return
    key = _project_query_key(query)
    now = time.monotonic()
    with _PROJECT_INFO_LOCK:
        _purge_project_queries(now)
        _PROJECT_INFO_RECENT[key] = now


def _is_suspicious_text(value: str) -> bool:
    if not value:
        return False
    if any(ord(ch) < 32 for ch in value):
        return True
    return bool(_SUSPICIOUS_INPUT_RE.search(value))


def _format_attribute_value(attr: AttributeSpec) -> str:
    unit = f" {attr.unit}" if attr.unit else ""
    if attr.value is not None:
        value_text = f"{attr.value}{unit}".strip()
        return _normalize_delivery_timeline_text(value_text)
    if attr.min is not None and attr.max is not None:
        return f"{attr.min}–{attr.max}{unit}"
    if attr.min is not None:
        return f"≥ {attr.min}{unit}"
    if attr.max is not None:
        return f"≤ {attr.max}{unit}"
    return attr.label


def _select_attributes(variant: VariantSpec, requested: Optional[List[str]]) -> List[AttributeSpec]:
    if requested:
        selected: List[AttributeSpec] = []
        for raw_name in requested:
            key = raw_name or ""
            attr = variant.attributes.get(key)
            if not attr:
                norm_key = normalize_token(key)
                attr = variant.attributes.get(norm_key)
            if attr and attr not in selected:
                selected.append(attr)
        if selected:
            return selected[:PROPERTY_SPECS_MAX_ATTR]
    return list(variant.attributes.values())[:PROPERTY_SPECS_MAX_ATTR]


async def get_property_specs(category: str, variant: Optional[str] = None, attributes: Optional[List[str]] = None, question: Optional[str] = None) -> Dict[str, Any]:
    catalog = get_catalog()
    payload: Dict[str, Any] = {
        "category": category,
        "variant": variant,
        "attributes": [],
    }

    category_spec = catalog.get_category(category)
    if not category_spec:
        payload["status"] = "unknown_category"
        payload["message"] = (
            "Je n'ai pas trouvé cette catégorie dans la fiche produits. Dis-moi si tu parles des appartements RIPT, résidences privées, villas ou lots."  # noqa: E501
        )
        return payload

    variant_spec = resolve_variant(category_spec, variant, question)
    if not variant_spec:
        if variant:
            payload["status"] = "unknown_variant"
            payload["message"] = (
                "Cette catégorie contient plusieurs variantes. Précise le type exact (ex: F2, Type 1, lot en bande)."
            )
            return payload

        # Aggregate all variants when no specific type identified
        variant_payloads: List[Dict[str, Any]] = []
        summary_parts: List[str] = []
        for var in category_spec.variants.values():
            attrs = _select_attributes(var, attributes)
            if not attrs:
                continue
            attr_payloads: List[Dict[str, Any]] = []
            attr_clauses: List[str] = []
            for attr in attrs:
                formatted = _format_attribute_value(attr)
                clause = f"{attr.label}: {formatted}"
                if attr.note:
                    clause += f" ({attr.note})"
                attr_payload = attr.to_payload()
                if isinstance(attr_payload.get("value"), str):
                    attr_payload["value"] = _normalize_delivery_timeline_text(str(attr_payload["value"]))
                attr_payload["text"] = clause
                attr_payloads.append(attr_payload)
                attr_clauses.append(clause)
            variant_payloads.append({
                "variant": var.id,
                "variant_label": var.label,
                "attributes": attr_payloads,
            })
            summary_parts.append(f"{var.label} — " + "; ".join(attr_clauses))

        if not variant_payloads:
            payload["status"] = "unknown_variant"
            payload["message"] = (
                "Je n'ai trouvé aucun détail pour ce type. Peux-tu préciser la typologie demandée ?"
            )
            return payload

        disclaimer = catalog.disclaimer or "Surfaces indicatives; confirmer avec le commercial."
        answer = ". ".join(summary_parts) + ". " + disclaimer if summary_parts else disclaimer
        answer = _normalize_delivery_timeline_text(answer)

        payload.update({
            "status": "ok",
            "category": category_spec.id,
            "category_label": category_spec.label,
            "variant": None,
            "variant_label": None,
            "attributes": [],
            "variants": variant_payloads,
            "answer": answer.strip(),
        })
        return payload

    selected_attrs = _select_attributes(variant_spec, attributes)
    attr_payloads: List[Dict[str, Any]] = []
    attr_clauses: List[str] = []
    for attr in selected_attrs:
        formatted = _format_attribute_value(attr)
        clause = f"{attr.label}: {formatted}"
        if attr.note:
            clause += f" ({attr.note})"
        attr_payload = attr.to_payload()
        if isinstance(attr_payload.get("value"), str):
            attr_payload["value"] = _normalize_delivery_timeline_text(str(attr_payload["value"]))
        attr_payload["text"] = clause
        attr_payloads.append(attr_payload)
        attr_clauses.append(clause)

    disclaimer = catalog.disclaimer or "Surfaces indicatives; confirmer avec le commercial."
    if attr_clauses:
        answer = f"{variant_spec.label} — " + "; ".join(attr_clauses) + ". " + disclaimer
    else:
        answer = (
            f"{variant_spec.label} : contactez un conseiller pour plus de précisions. "
            + disclaimer
        )
    answer = _normalize_delivery_timeline_text(answer)

    payload.update({
        "status": "ok",
        "category": category_spec.id,
        "category_label": category_spec.label,
        "variant": variant_spec.id,
        "variant_label": variant_spec.label,
        "attributes": attr_payloads,
        "answer": answer.strip(),
    })
    return payload


async def get_project_facts(section: Optional[str] = None, topic: Optional[str] = None, question: Optional[str] = None) -> Dict[str, Any]:
    catalog = get_project_facts_catalog()
    payload: Dict[str, Any] = {
        "section": section,
        "topic": topic,
    }

    selected_section = catalog.find_section(section) if section else None
    if section and not selected_section:
        payload["status"] = "unknown_section"
        payload["message"] = (
            "Je ne retrouve pas cette rubrique. Parle-moi de localisation, projet, cadre légal, vie sur site ou commercialisation."
        )
        return payload

    selected_entry = None
    if selected_section:
        selected_entry = resolve_fact_entry(selected_section, topic, question)
        if not selected_entry and not question:
            topics = ", ".join(e.label for e in selected_section.entries.values())
            payload["status"] = "unknown_topic"
            payload["message"] = f"Cette rubrique contient: {topics}. Lequel t'intéresse ?"
            return payload

    if not selected_entry and not selected_section and question:
        match = fuzzy_find_entry(catalog, question)
        if match:
            selected_section, selected_entry = match

    if not selected_entry:
        payload["status"] = "not_found"
        payload["message"] = "Précise la rubrique (ex: localisation, promoteurs, sécurité) pour que je te réponde."
        return payload

    bullets = selected_entry.bullets
    extra = ""
    if bullets:
        extra = " " + "; ".join(bullets)
    disclaimer = catalog.disclaimer or ""
    answer = selected_entry.text + extra
    if disclaimer:
        answer = f"{answer} {disclaimer}".strip()
    answer = _normalize_delivery_timeline_text(answer)

    payload.update({
        "status": "ok",
        "section": selected_section.id if selected_section else section,
        "section_label": selected_section.label if selected_section else None,
        "entry": selected_entry.id,
        "entry_label": selected_entry.label,
        "answer": answer,
    })
    return payload


async def get_project_info(query: str, top_k: int | None = None) -> Dict[str, Any]:
    """Retrieve contextual snippets with guardrails applied."""
    clean_query = (query or "").strip()
    payload: Dict[str, Any] = {"query": clean_query, "snippets": []}

    if not clean_query:
        payload["status"] = "empty"
        payload["message"] = "Je n'ai pas compris la question. Pouvez-vous la reformuler ?"
        return payload

    if len(clean_query) < PROJECT_INFO_MIN_LENGTH:
        payload["status"] = "too_short"
        payload["message"] = (
            f"Merci de préciser votre question en au moins {PROJECT_INFO_MIN_LENGTH} caractères."
        )
        return payload

    if len(clean_query) > PROJECT_INFO_MAX_QUERY_LENGTH:
        payload["status"] = "too_long"
        payload["query"] = clean_query[:PROJECT_INFO_MAX_QUERY_LENGTH]
        payload["message"] = "La question est trop longue; pouvez-vous la raccourcir ?"
        return payload

    lowered = clean_query.lower()
    if PROJECT_INFO_BLOCKLIST and any(blocked in lowered for blocked in PROJECT_INFO_BLOCKLIST):
        payload["status"] = "blocked_keyword"
        payload["message"] = "Je ne peux pas répondre à cette question. Merci de rester sur des sujets liés à Alma Resort."
        return payload

    if _is_suspicious_text(clean_query):
        payload["status"] = "suspicious"
        payload["message"] = "La question semble contenir des caractères inattendus. Pouvez-vous la reformuler ?"
        return payload

    if _is_recent_project_query(clean_query):
        payload["status"] = "duplicate"
        payload["message"] = (
            "Je viens déjà de répondre à cette question. Avez-vous un autre point à préciser ?"
        )
        return payload

    rate_token = _register_project_query_attempt()
    if rate_token is None:
        payload["status"] = "rate_limited"
        payload["message"] = "Je reçois beaucoup de demandes. Réessayez dans quelques instants, s'il vous plaît."
        return payload

    k = top_k or config.retrieval.top_k
    k = max(1, min(k, 3))

    try:
        snippets: List[Dict[str, Any]] = await asyncio.to_thread(retrieve_context, clean_query, k)
    except Exception:
        _rollback_project_query_attempt(rate_token)
        raise

    _record_project_query(clean_query)
    payload["snippets"] = snippets
    payload["status"] = "ok"
    return payload


__all__ = [
    "ask_to_reserve",
    "get_project_info",
    "get_property_specs",
    "get_project_facts",
]
