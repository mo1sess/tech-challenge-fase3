"""LangChain-backed client for the official remote Qwen3-8B adapter service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.runnables import RunnableLambda


OFFICIAL_GENERATOR_MODE = "qwen3_8b_qlora_remote"


def _validate_endpoint_url(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    parsed = urlparse(endpoint)
    is_local = parsed.hostname in {"127.0.0.1", "localhost"}
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("TECHCARE_REMOTE_URL must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and not is_local:
        raise ValueError("Remote Qwen endpoint must use HTTPS")
    return endpoint


def _request_json(
    method: str,
    url: str,
    *,
    token: str,
    timeout_seconds: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Remote Qwen service returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Remote Qwen service is unavailable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Remote Qwen service returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Remote Qwen service response must be a JSON object")
    return result


@dataclass
class RemoteQwenResponseGenerator:
    """Invoke the fine-tuned Qwen service through a LangChain Runnable."""

    endpoint_url: str
    token: str = field(repr=False)
    timeout_seconds: float = 180.0
    expected_model_id: str = "Qwen/Qwen3-8B"
    expected_revision: str = ""
    expected_adapter_sha256: str = ""
    mode: str = field(default=OFFICIAL_GENERATOR_MODE, init=False)

    def __post_init__(self) -> None:
        self.endpoint_url = _validate_endpoint_url(self.endpoint_url)
        self.token = self.token.strip()
        if not self.token:
            raise ValueError("TECHCARE_REMOTE_TOKEN is required in remote mode")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.runnable = RunnableLambda(self._invoke_remote)

    def _invoke_remote(self, prompt: str) -> str:
        result = _request_json(
            "POST",
            f"{self.endpoint_url}/v1/generate",
            token=self.token,
            timeout_seconds=self.timeout_seconds,
            payload={"prompt": prompt},
        )
        response = str(result.get("response", "")).strip()
        if not response:
            raise RuntimeError("Remote Qwen service returned an empty response")
        metadata = result.get("model", {})
        self._validate_model_metadata(metadata)
        return response

    def _validate_model_metadata(self, metadata: Any) -> None:
        if not isinstance(metadata, dict):
            raise RuntimeError("Remote Qwen service omitted model metadata")
        expected = {
            "id": self.expected_model_id,
            "revision": self.expected_revision,
            "adapter_sha256": self.expected_adapter_sha256,
        }
        mismatches = {
            key: {"expected": value, "received": metadata.get(key)}
            for key, value in expected.items()
            if value and metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Remote model identity mismatch: {mismatches}")

    def health(self) -> dict[str, Any]:
        result = _request_json(
            "GET",
            f"{self.endpoint_url}/health",
            token=self.token,
            timeout_seconds=min(self.timeout_seconds, 15.0),
        )
        if result.get("ok") is not True:
            raise RuntimeError("Remote Qwen service health check failed")
        self._validate_model_metadata(result.get("model", {}))
        return result

    def generate(self, prompt: str, state: dict[str, Any]) -> str:
        del state
        return str(self.runnable.invoke(prompt))
