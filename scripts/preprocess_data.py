"""Execute the complete, deterministic stage 2 preprocessing pipeline."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.preprocessing.dataset_builder import build_processed_datasets


if __name__ == "__main__":
    result = build_processed_datasets(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))

