"""Hypertensor Field Regressor package."""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType
from typing import Any


def _ensure_torch_hypertensor_stub() -> None:
    """Install a shim for `torch.distributed.hypertensor` when torch lacks it."""

    try:
        import torch
        import torch.distributed  # type: ignore
    except ModuleNotFoundError:
        return

    module_name = "torch.distributed.hypertensor"
    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    warning = (
        "torch.distributed.hypertensor is unavailable; tensor parallel features will be disabled. "
        "Install PyTorch >= 2.5 with hypertensor support for full functionality."
    )
    warnings.warn(warning)

    class Placement:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class Replicate(Placement):
        pass

    class Shard(Placement):
        pass

    class DTensor:
        def __init__(self, tensor: Any) -> None:
            self._tensor = tensor

        @classmethod
        def from_local(cls, tensor: Any, *args: Any, **kwargs: Any) -> DTensor:
            return cls(tensor)

        def to_local(self) -> Any:
            return self._tensor

        def __getattr__(self, name: str) -> Any:
            return getattr(self._tensor, name)

    stub = ModuleType(module_name)
    stub.DTensor = DTensor
    stub.Placement = Placement
    stub.Replicate = Replicate
    stub.Shard = Shard
    sys.modules[module_name] = stub
    setattr(torch.distributed, "hypertensor", stub)


_ensure_torch_hypertensor_stub()

from .hypertensor import Hypertensor
from .interpolation import available_interpolations
from .model import HTFRModel, locality_weights
from .hypertensor_field_transformer import HypertensorFieldTransformer, StageRuntime

__all__ = [
    "Hypertensor",
    "HTFRModel",
    "locality_weights",
    "available_interpolations",
    "HypertensorFieldTransformer",
    "StageRuntime",
]
