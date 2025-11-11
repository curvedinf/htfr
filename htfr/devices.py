"""Torch device helpers supporting CUDA, ROCm (HIP), and MPS."""
from __future__ import annotations

import os
from typing import Optional

import torch


def _has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _has_hip() -> bool:
    return bool(getattr(torch.version, "hip", None))


def get_preferred_device(device: Optional[str] = None) -> torch.device:
    """Return a torch.device honoring HIP/MPS backends when available."""

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        # On ROCm builds torch.cuda reports HIP devices, so treat them uniformly.
        return torch.device("cuda", torch.cuda.current_device())
    if _has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device identifier used in logs/metadata."""

    if device.type == "cuda" and _has_hip():
        index = device.index if device.index is not None else 0
        gfx = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        suffix = f":{index}"
        if gfx:
            suffix += f" (gfx{gfx})"
        return f"hip{suffix}"
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type
