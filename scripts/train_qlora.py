"""Run official stage 5 QLoRA training in Colab or Kaggle."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.finetuning.trainer import run_qlora_training


if __name__ == "__main__":
    print(json.dumps(run_qlora_training(project_root()), ensure_ascii=False, indent=2))
