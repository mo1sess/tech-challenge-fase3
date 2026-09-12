"""Basic structural validations for acquired datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinical_assistant.acquisition.datasets import COUNTERS
from clinical_assistant.acquisition.common import utc_now, write_json


def validate_all(root: Path) -> dict[str, Any]:
    """Validate presence, parsability, and non-zero primary record counts."""

    checks: list[dict[str, Any]] = []
    primary = {"medquad": "qa_pairs", "pubmedqa": "pqa_l_records", "synthea": "patients"}
    for name, counter in COUNTERS.items():
        path = root / "data" / "raw" / name
        try:
            counts = counter(path)
            ok = counts.get(primary[name], 0) > 0
            checks.append({"dataset": name, "ok": ok, "counts": counts})
        except Exception as exc:  # reported as validation data, not hidden
            checks.append({"dataset": name, "ok": False, "error": str(exc)})

    protocols_manifest = root / "data" / "raw" / "_manifests" / "protocols.json"
    if protocols_manifest.exists():
        with protocols_manifest.open("r", encoding="utf-8") as stream:
            protocol_data = json.load(stream)
        checks.append(
            {
                "dataset": "protocols",
                "ok": True,
                "counts": protocol_data.get("counts", {}),
                "note": "Zero is valid when no clinical protocol was supplied.",
            }
        )
    else:
        checks.append({"dataset": "protocols", "ok": False, "error": "manifest missing"})

    report = {
        "validated_at_utc": utc_now(),
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }
    write_json(root / "outputs" / "data_validation.json", report)
    return report

