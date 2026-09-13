"""Validate generated QLoRA splits without importing the GPU stack."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.finetuning.dataset import validate_finetuning_dataset


if __name__ == "__main__":
    report = validate_finetuning_dataset(project_root())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
