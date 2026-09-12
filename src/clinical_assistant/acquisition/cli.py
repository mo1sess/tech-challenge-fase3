"""Command-line entry point for stage 1 acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_assistant.acquisition.datasets import acquire_archive
from clinical_assistant.acquisition.protocols import import_protocols
from clinical_assistant.config import load_yaml, project_root, resolve_project_path


DOWNLOADABLE = ("medquad", "pubmedqa", "synthea")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire stage 1 datasets.")
    parser.add_argument(
        "--dataset",
        choices=("all", *DOWNLOADABLE, "protocols"),
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing raw data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    config = load_yaml(root / "configs" / "datasets.yaml")["datasets"]
    names = (*DOWNLOADABLE, "protocols") if args.dataset == "all" else (args.dataset,)
    results = []
    for name in names:
        if name == "protocols":
            item = config[name]
            result = import_protocols(
                resolve_project_path(item["inbox"], root=root),
                resolve_project_path(item["destination"], root=root),
                allowed_extensions={value.lower() for value in item["allowed_extensions"]},
                manifest_path=root / "data" / "raw" / "_manifests" / "protocols.json",
            )
        else:
            result = acquire_archive(name, config[name], project_root=root, force=args.force)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

