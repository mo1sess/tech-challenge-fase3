"""Validate stage 10 preflight or an imported official evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_assistant.config import project_root
from clinical_assistant.evaluation.validation import (
    validate_evaluation_run,
    validate_stage10_preflight,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Diretório importado em outputs/evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    root = project_root()
    result = (
        validate_evaluation_run(root, arguments.run)
        if arguments.run
        else validate_stage10_preflight(root)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
