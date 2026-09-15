"""Preflight validation for the stage-11.1 remote Qwen integration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_assistant.config import load_yaml, resolve_project_path


REQUIRED_FILES = (
    "src/clinical_assistant/finetuning/remote.py",
    "src/clinical_assistant/finetuning/service.py",
    "scripts/serve_qwen_agent.py",
    "scripts/validate_full_agent.py",
    "scripts/run_remote_agent_validation.py",
    "requirements/full-agent-gpu.txt",
    "notebooks/07_full_agent_colab.ipynb",
)


def validate_full_agent_preflight(
    root: Path, *, write_report: bool = True
) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "full_agent.yaml")
    errors: list[str] = []
    if str(config.get("stage")) != "11.1":
        errors.append("full-agent configuration must declare stage 11.1")

    model = config.get("model", {})
    if model.get("id") != "Qwen/Qwen3-8B" or model.get("allow_substitution") is not False:
        errors.append("official model must remain the pinned Qwen/Qwen3-8B")
    if config.get("execution", {}).get("local_gpu_allowed") is not False:
        errors.append("the official model cannot be enabled on the local GTX 1650")

    files = {relative: (root / relative).is_file() for relative in REQUIRED_FILES}
    errors.extend(f"missing required file: {path}" for path, ready in files.items() if not ready)

    manifest_path = resolve_project_path(str(model.get("adapter_manifest", "")), root=root)
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append(f"missing stage-5 training manifest: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_key = str(model.get("adapter_artifact", ""))
    adapter_hash = str(manifest.get("artifact_sha256", {}).get(artifact_key, ""))
    if len(adapter_hash) != 64:
        errors.append("official adapter SHA-256 is missing from the training manifest")
    manifest_model = manifest.get("model", {})
    if manifest and (
        manifest_model.get("id") != model.get("id")
        or manifest_model.get("revision") != model.get("revision")
    ):
        errors.append("full-agent model identity differs from the stage-5 manifest")

    execution = config.get("execution", {})
    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "stage": "11.1",
        "implementation": {
            "local_mode": execution.get("local_mode"),
            "official_mode": execution.get("official_mode"),
            "langchain_remote_generator": files[
                "src/clinical_assistant/finetuning/remote.py"
            ],
            "gpu_service": files["src/clinical_assistant/finetuning/service.py"],
            "colab_notebook": files["notebooks/07_full_agent_colab.ipynb"],
            "silent_fallback_allowed": False,
        },
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
            "adapter_sha256": adapter_hash,
        },
        "local_environment": {
            "official_model_loaded": False,
            "gpu_used": False,
        },
        "official_gpu_execution": {
            "status": "pending_remote_execution",
            "minimum_vram_gb": execution.get("minimum_vram_gb"),
        },
        "required_files": files,
        "errors": errors,
    }
    if write_report:
        report_path = resolve_project_path(config["outputs"]["preflight_report"], root=root)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
