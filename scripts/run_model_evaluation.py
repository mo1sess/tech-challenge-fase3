"""Run the stage 10 adapter and adapter+RAG evaluation on a remote CUDA GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_assistant.config import project_root
from clinical_assistant.evaluation.evaluator import run_model_evaluation


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_directory", type=Path)
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for official run")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    result = run_model_evaluation(
        project_root(), adapter_directory=arguments.adapter_directory, limit=arguments.limit
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
