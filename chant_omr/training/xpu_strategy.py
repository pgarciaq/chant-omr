"""Native Intel XPU support for PyTorch Lightning (no IPEX).

Lightning has no built-in ``accelerator="xpu"`` yet; pass :class:`SingleXPUStrategy`
to :class:`~lightning.pytorch.Trainer` instead.

Adapted from https://github.com/MarekOzana/lightning-neuralforecast-xpu (MIT).
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def xpu_is_available() -> bool:
    """Return True when ``torch.xpu`` exists and a device is usable."""
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


class NativeXPUAccelerator(Accelerator):
    """Minimal native ``torch.xpu`` accelerator for Lightning."""

    @staticmethod
    def name() -> str:
        return "xpu"

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be XPU, got {device}.")
        if not self.is_available():
            raise MisconfigurationException("XPU is not available on this system.")
        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        return devices

    @staticmethod
    def get_parallel_devices(devices: int) -> list[torch.device]:
        return [torch.device("xpu", index) for index in range(int(devices))]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count() if NativeXPUAccelerator.is_available() else 0

    @staticmethod
    def is_available() -> bool:
        return xpu_is_available()

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:
        # Avoid Lightning metrics calls that can destabilize some XPU stacks.
        return {}

    def teardown(self) -> None:
        if self.is_available():
            torch.xpu.empty_cache()

    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description="Native Intel XPU accelerator via torch.xpu",
        )


class SingleXPUStrategy(SingleDeviceStrategy):
    """Single Intel GPU via native ``torch.xpu`` (bypasses Lightning string checks)."""

    strategy_name = "xpu_single"

    def __init__(self, device_index: int = 0, **kwargs: Any) -> None:
        super().__init__(
            device=torch.device("xpu", device_index),
            accelerator=NativeXPUAccelerator(),
            **kwargs,
        )

    @classmethod
    def register_strategies(cls, strategy_registry: Any) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description="Single-device Intel XPU strategy",
        )
