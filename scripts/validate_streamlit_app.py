"""Validate the Stage 11 Streamlit MVP without opening a browser."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.validation.interface import validate_streamlit_stage


if __name__ == "__main__":
    result = validate_streamlit_stage(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
