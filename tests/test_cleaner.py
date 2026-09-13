from __future__ import annotations

import pytest

from clinical_assistant.preprocessing.cleaner import normalize_text


@pytest.mark.unit
def test_normalize_text_decodes_html_unicode_and_whitespace() -> None:
    assert normalize_text("  Asma&nbsp;\tgrave\x00  ") == "Asma grave"


@pytest.mark.unit
def test_normalize_text_handles_none() -> None:
    assert normalize_text(None) == ""

