"""PII redaction for logging to protect sensitive data."""

from __future__ import annotations

import logging
import re
from typing import Pattern


class PIIRedactingFormatter(logging.Formatter):
    """Custom formatter that redacts PII (Personally Identifiable Information) from logs."""

    # Compiled regex patterns for PII detection
    PII_PATTERNS: list[tuple[Pattern, str]] = [
        # Email addresses
        (re.compile(r"[\w\.\-\+]+@[\w\.\-]+\.\w+", re.IGNORECASE), "[EMAIL_REDACTED]"),
        # Phone numbers (various formats)
        (re.compile(r"\+?[0-9]{1,3}[\s\-\(\)]?[0-9]{3,4}[\s\-\(\)]?[0-9]{3,4}[\s\-\(\)]?[0-9]{0,4}"), "[PHONE_REDACTED]"),
        # Credit card patterns (basic)
        (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD_REDACTED]"),
        # API keys and tokens (common patterns)
        (re.compile(r"(?:api_key|token|secret|password)['\"]?\s*[:=]\s*['\"]?([^'\"\s,}]+)", re.IGNORECASE), r"\g<0>[REDACTED]"),
        # Full name patterns in logs (e.g., full_name: "John Doe")
        (re.compile(r'(?:full_name)["\']?\s*[:=]\s*["\']?([^"\'}{,\s]+\s+[^"\'}{,\s]+)', re.IGNORECASE), r"full_name: [REDACTED]"),
        # Email in JSON/dict format
        (re.compile(r'(?:email|email_address)["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), r"email: [REDACTED]"),
        # Phone in JSON/dict format
        (re.compile(r'(?:phone|phone_number)["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.IGNORECASE), r"phone: [REDACTED]"),
    ]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with PII redaction."""
        # Format the message first
        msg = super().format(record)

        # Apply redaction patterns
        for pattern, replacement in self.PII_PATTERNS:
            msg = pattern.sub(replacement, msg)

        return msg
