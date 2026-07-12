"""Tests for training device / precision resolution."""

from __future__ import annotations

import pytest

from chant_omr.training.lightning_module import (
    _effective_accelerator,
    format_training_device_message,
    resolve_precision,
    resolve_trainer_devices,
)
from chant_omr.training.xpu_strategy import SingleXPUStrategy


def test_effective_accelerator_cpu_when_gpus_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chant_omr.training.lightning_module.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("chant_omr.training.lightning_module.xpu_is_available", lambda: True)
    assert _effective_accelerator("auto", 0) == "cpu"


def test_effective_accelerator_prefers_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chant_omr.training.lightning_module.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("chant_omr.training.lightning_module.xpu_is_available", lambda: True)
    assert _effective_accelerator("auto", 1) == "cuda"


def test_effective_accelerator_falls_back_to_xpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "chant_omr.training.lightning_module.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr("chant_omr.training.lightning_module.xpu_is_available", lambda: True)
    assert _effective_accelerator("auto", 1) == "xpu"


def test_resolve_precision_cpu_mixed_to_full():
    assert resolve_precision("bf16-mixed", accelerator="cpu") == "32-true"


def test_resolve_precision_xpu_mixed_to_bf16_true():
    assert resolve_precision("bf16-mixed", accelerator="xpu") == "bf16-true"
    assert resolve_precision("16-mixed", accelerator="xpu") == "bf16-true"


def test_format_training_device_message_xpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chant_omr.training.lightning_module.xpu_is_available", lambda: True)
    monkeypatch.setattr(
        "chant_omr.training.lightning_module.torch.xpu.get_device_name",
        lambda _index: "Intel(R) Arc(TM) Graphics",
    )
    msg = format_training_device_message(effective="xpu", xpu_index=0)
    assert msg.startswith("Using Intel XPU: xpu:0")
    assert "Intel(R) Arc(TM) Graphics" in msg
    assert "CUDA only" in msg


def test_format_training_device_message_cpu():
    assert format_training_device_message(effective="cpu") == "Using CPU"


def test_resolve_trainer_devices_xpu_strategy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chant_omr.training.lightning_module.xpu_is_available", lambda: True)
    accel, devices, strategy = resolve_trainer_devices(accelerator="xpu", gpus=1, xpu_index=0)
    assert accel is None
    assert devices == 1
    assert isinstance(strategy, SingleXPUStrategy)
    assert str(strategy.root_device) == "xpu:0"
