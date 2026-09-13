from __future__ import annotations

from types import SimpleNamespace

import pytest

from clinical_assistant.evaluation.baseline import _gpu_environment


class _NoCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _SmallCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_properties(index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name="Test GPU", total_memory=4 * 1024**3)

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        return (7, 5)

    @staticmethod
    def is_bf16_supported() -> bool:
        return False


@pytest.mark.unit
def test_baseline_blocks_cpu_execution() -> None:
    torch = SimpleNamespace(cuda=_NoCuda(), version=SimpleNamespace(cuda=None))
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        _gpu_environment(torch, 12)


@pytest.mark.unit
def test_baseline_blocks_local_4gb_gpu() -> None:
    torch = SimpleNamespace(cuda=_SmallCuda(), version=SimpleNamespace(cuda="11.1"))
    with pytest.raises(RuntimeError, match="at least 12.0 GB"):
        _gpu_environment(torch, 12)
