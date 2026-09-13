"""Conservative text cleaning without changing clinical meaning."""

from __future__ import annotations

import html
import re
import unicodedata


CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Normalize Unicode, HTML entities, controls, and repeated whitespace."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = CONTROL_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_identifier(value: object) -> str:
    """Normalize an identifier without case-folding or semantic rewriting."""

    return normalize_text(value)

