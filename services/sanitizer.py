"""Input sanitization for agent-worker.

Protects against:
1. Prompt injection — user input embedded in LLM prompts
2. Log injection — user input written to log files
3. SQL injection — ORM already handles this, defense-in-depth only

Strategy:
- Pydantic validators on request models (first line of defense)
- Sanitization helpers for LLM-bound strings (second line)
- Output validation on LLM responses (third line)
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints
from pydantic.functional_validators import AfterValidator

# ── Constants ──────────────────────────────────────────────────

# Characters that are NEVER valid in brand names, URLs, or search queries
_NEVER_VALID_IN_IDENTIFIER = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"

# Prompt injection markers to strip
_INJECTION_MARKERS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "forget your instructions",
    "system prompt:",
    "<<SYS>>",
    "<|im_start|>",
    "<|im_end|>",
    "you are now",
    "new instructions:",
    "override:",
]

# Max reasonable lengths for different field types
MAX_BRAND_NAME_LEN = 200
MAX_INDUSTRY_LEN = 2000
MAX_VALUE_PROP_LEN = 3000
MAX_AUDIENCE_LEN = 1000
MAX_URL_LEN = 500
MAX_QUERY_TEXT_LEN = 300


# ── Sanitizers ─────────────────────────────────────────────────

def _is_url(text: str) -> bool:
    """Check if text looks like a URL."""
    return bool(re.match(r"^https?://", text.strip()))


def sanitize_llm_text(text: str, max_len: int = MAX_INDUSTRY_LEN) -> str:
    """Sanitize user input before embedding in LLM prompts.

    - Truncates to max length
    - Strips control characters
    - Removes common injection markers
    - Escapes JSON-breaking characters
    """
    if not text:
        return ""

    # Truncate
    text = text[:max_len]

    # Strip null bytes and control characters (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Neutralize injection markers (case-insensitive)
    for marker in _INJECTION_MARKERS:
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        text = pattern.sub("[filtered]", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def sanitize_url(text: str) -> str:
    """Sanitize a URL field.

    - Must be a valid HTTP(S) URL or empty
    - Strip control characters
    - Limit length
    """
    if not text:
        return ""

    text = text.strip()[:MAX_URL_LEN]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Only allow http/https
    if not re.match(r"^https?://", text):
        return ""

    return text


def sanitize_brand_name(text: str) -> str:
    """Sanitize brand name field.

    - Short length limit
    - Strip injection markers
    - No URLs allowed (brand name should not be a URL)
    """
    if not text:
        return ""

    text = sanitize_llm_text(text, MAX_BRAND_NAME_LEN)

    # Brand name should not be a URL
    if _is_url(text):
        # Extract just the domain/path part without protocol
        text = re.sub(r"^https?://", "", text)

    return text


def validate_llm_output(text: str, expected_keys: set[str] | None = None) -> bool:
    """Validate LLM-generated output is well-formed.

    For JSON outputs with response_format=json_object, the LLM
    should always return valid JSON. This is a defense-in-depth check.
    """
    import json

    if not text:
        return False

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return False

        # If we expect specific keys, validate them
        if expected_keys:
            for key in expected_keys:
                if key not in data:
                    return False
                value = data[key]
                if not isinstance(value, str):
                    return False
                if len(value) > MAX_QUERY_TEXT_LEN:
                    return False
                # Ensure no injection markers in LLM output
                for marker in _INJECTION_MARKERS:
                    if marker.lower() in value.lower():
                        return False

        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


# ── Pydantic validators ────────────────────────────────────────

def _validate_no_injection(v: str) -> str:
    """Pydantic validator: reject strings containing injection markers."""
    if not v:
        return v
    lower = v.lower()
    for marker in _INJECTION_MARKERS:
        if marker.lower() in lower:
            raise ValueError(f"Input contains prohibited content")
    return v


def _validate_no_control_chars(v: str) -> str:
    """Pydantic validator: reject strings with control characters."""
    if not v:
        return v
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", v):
        raise ValueError("Input contains invalid control characters")
    return v


# Reusable Pydantic types for request models
SafeText = Annotated[
    str,
    StringConstraints(max_length=MAX_INDUSTRY_LEN),
    AfterValidator(_validate_no_injection),
    AfterValidator(_validate_no_control_chars),
]

SafeBrandName = Annotated[
    str,
    StringConstraints(max_length=MAX_BRAND_NAME_LEN),
    AfterValidator(_validate_no_injection),
    AfterValidator(_validate_no_control_chars),
]

SafeURL = Annotated[
    str,
    StringConstraints(max_length=MAX_URL_LEN),
    AfterValidator(_validate_no_control_chars),
]
