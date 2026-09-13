"""Prepare the deterministic stage 5 QLoRA dataset."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.finetuning.dataset import prepare_finetuning_dataset


if __name__ == "__main__":
    print(json.dumps(prepare_finetuning_dataset(project_root()), ensure_ascii=False, indent=2))
