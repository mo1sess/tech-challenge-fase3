from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from clinical_assistant.config import project_root
from clinical_assistant.finetuning import remote
from clinical_assistant.finetuning.remote import RemoteQwenResponseGenerator
from clinical_assistant.finetuning.service import inference_gpu_environment
from clinical_assistant.validation.full_agent import validate_full_agent_preflight


MODEL = {
    "id": "Qwen/Qwen3-8B",
    "revision": "official-revision",
    "adapter_sha256": "a" * 64,
}


@pytest.mark.unit
def test_remote_generator_uses_langchain_and_validates_model(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, url: str, **kwargs):
        calls.append((method, url))
        assert kwargs["token"] == "x" * 32
        if method == "GET":
            return {"ok": True, "model": MODEL, "environment": {"gpu_name": "Tesla T4"}}
        assert kwargs["payload"] == {"prompt": "contexto sintético"}
        return {"response": "Resposta contextualizada.", "model": MODEL}

    monkeypatch.setattr(remote, "_request_json", fake_request)
    generator = RemoteQwenResponseGenerator(
        endpoint_url="https://agent.example.test",
        token="x" * 32,
        expected_model_id=MODEL["id"],
        expected_revision=MODEL["revision"],
        expected_adapter_sha256=MODEL["adapter_sha256"],
    )
    assert generator.health()["environment"]["gpu_name"] == "Tesla T4"
    assert generator.generate("contexto sintético", {}) == "Resposta contextualizada."
    assert calls == [
        ("GET", "https://agent.example.test/health"),
        ("POST", "https://agent.example.test/v1/generate"),
    ]


@pytest.mark.unit
def test_remote_generator_rejects_insecure_public_url() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        RemoteQwenResponseGenerator(
            endpoint_url="http://agent.example.test",
            token="x" * 32,
        )


@pytest.mark.unit
def test_remote_generator_rejects_model_substitution(monkeypatch) -> None:
    monkeypatch.setattr(
        remote,
        "_request_json",
        lambda *args, **kwargs: {
            "response": "Não deve ser aceita.",
            "model": {**MODEL, "id": "another/model"},
        },
    )
    generator = RemoteQwenResponseGenerator(
        endpoint_url="https://agent.example.test",
        token="x" * 32,
        expected_model_id=MODEL["id"],
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        generator.generate("teste", {})


@pytest.mark.unit
def test_official_inference_rejects_cpu_and_small_local_gpu() -> None:
    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="remote CUDA GPU"):
        inference_gpu_environment(no_cuda, 14)

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda index: SimpleNamespace(
            name="GTX 1650", total_memory=4 * 1024**3
        ),
        get_device_capability=lambda index: (7, 5),
    )
    small_gpu = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="11.1"))
    with pytest.raises(RuntimeError, match="at least 14.0 GB"):
        inference_gpu_environment(small_gpu, 14)


@pytest.mark.integration
def test_full_agent_preflight_is_reproducible() -> None:
    result = validate_full_agent_preflight(project_root(), write_report=False)
    assert result["ok"] is True
    assert result["stage"] == "11.1"
    assert result["model"]["adapter_sha256"] == (
        "d744bf09a8d7bbe1018ce48091429d82361f72f7c9e34e2a6f8f89d45e1855e3"
    )
    assert result["implementation"]["silent_fallback_allowed"] is False
    assert result["implementation"]["structured_evidence_lock"] is True
    assert result["official_gpu_execution"]["status"] == "pending_remote_execution"


@pytest.mark.unit
def test_colab_notebook_is_valid_and_never_targets_local_gpu() -> None:
    path = project_root() / "notebooks" / "07_full_agent_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    assert notebook["nbformat"] == 4
    assert "serve_qwen_agent.py" in source
    assert "TECHCARE_REMOTE_TOKEN" in source
    assert "GTX 1650" in source
    assert "Não execute na GTX 1650" in source
