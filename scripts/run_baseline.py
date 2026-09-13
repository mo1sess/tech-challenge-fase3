"""Run the official Qwen3-8B baseline in a remote Linux CUDA environment."""

from __future__ import annotations

import argparse
import json

from clinical_assistant.config import project_root
from clinical_assistant.evaluation.baseline import run_baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; omit for the official 24-case run")
    args = parser.parse_args()
    result = run_baseline(project_root(), limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
