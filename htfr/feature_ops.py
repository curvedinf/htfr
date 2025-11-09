"""Feature transformations used by HTFR training and inference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _fwht_inplace(x: np.ndarray) -> None:
    """In-place normalized fast Walsh-Hadamard transform."""

    h = 1
    n = x.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            j = i
            k = i + h
            for _ in range(h):
                x_j = x[j]
                x_k = x[k]
                x[j] = x_j + x_k
                x[k] = x_j - x_k
                j += 1
                k += 1
        h *= 2
    x /= math.sqrt(float(n))


def _next_power_of_two(value: int) -> int:
    """Return the next power-of-two greater than or equal to ``value``."""

    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class SRHTParameters:
    """Parameters describing a block-structured SRHT projection."""

    signs: np.ndarray
    permutation: np.ndarray
    input_dim: int
    padded_dim: int
    target_dim: int
    block_size: int
    block_eps: float
    scale: float

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "srht_signs": self.signs,
            "srht_permutation": self.permutation,
            "srht_input_dim": np.array([self.input_dim], dtype=np.int32),
            "srht_padded_dim": np.array([self.padded_dim], dtype=np.int32),
            "srht_target_dim": np.array([self.target_dim], dtype=np.int32),
            "srht_block_size": np.array([self.block_size], dtype=np.int32),
            "srht_block_eps": np.array([self.block_eps], dtype=np.float32),
            "srht_scale": np.array([self.scale], dtype=np.float32),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "SRHTParameters":
        return cls(
            signs=np.asarray(arrays["srht_signs"], dtype=np.float32),
            permutation=np.asarray(arrays["srht_permutation"], dtype=np.int64),
            input_dim=int(np.asarray(arrays["srht_input_dim"]).item()),
            padded_dim=int(np.asarray(arrays["srht_padded_dim"]).item()),
            target_dim=int(np.asarray(arrays["srht_target_dim"]).item()),
            block_size=int(np.asarray(arrays["srht_block_size"]).item()),
            block_eps=float(np.asarray(arrays["srht_block_eps"]).item()),
            scale=float(np.asarray(arrays["srht_scale"]).item()),
        )


def make_block_srht(
    input_dim: int,
    target_dim: int = 4096,
    block_size: int = 256,
    block_eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> SRHTParameters:
    """Construct parameters for a blockwise SRHT projection."""

    if rng is None:
        rng = np.random.default_rng()
    padded_dim = _next_power_of_two(input_dim)
    signs = rng.choice([-1.0, 1.0], size=padded_dim).astype(np.float32)
    permutation = rng.permutation(padded_dim)
    scale = math.sqrt(padded_dim / float(target_dim))
    return SRHTParameters(
        signs=signs,
        permutation=permutation,
        input_dim=input_dim,
        padded_dim=padded_dim,
        target_dim=target_dim,
        block_size=block_size,
        block_eps=block_eps,
        scale=scale,
    )


def apply_block_srht(data: np.ndarray, params: SRHTParameters) -> np.ndarray:
    """Apply the SRHT projection to ``data`` and return the transformed features."""

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != params.input_dim:
        raise ValueError(
            f"Expected data with shape (N, {params.input_dim}), got {data.shape}"
        )
    n = data.shape[0]
    padded = np.zeros((n, params.padded_dim), dtype=np.float32)
    padded[:, : params.input_dim] = data
    padded *= params.signs
    transformed = np.empty((n, params.target_dim), dtype=np.float32)
    for idx in range(n):
        vec = padded[idx]
        _fwht_inplace(vec)
        projected = vec[params.permutation][: params.target_dim] * params.scale
        transformed[idx] = projected
    return block_rmsnorm(transformed, params.block_size, params.block_eps)


def block_rmsnorm(
    data: np.ndarray, block_size: int, eps: float = 1e-6
) -> np.ndarray:
    """Apply block-wise RMS normalization to ``data``."""

    data = np.asarray(data, dtype=np.float32)
    if block_size <= 0:
        return data
    n, dim = data.shape
    output = data.copy()
    for start in range(0, dim, block_size):
        stop = min(start + block_size, dim)
        block = output[:, start:stop]
        rms = np.sqrt(np.mean(block**2, axis=1, keepdims=True) + eps)
        output[:, start:stop] = block / rms
    return output


def srht_feature_tuple(
    hidden: np.ndarray, params: SRHTParameters
) -> Tuple[np.ndarray, SRHTParameters]:
    """Convenience wrapper returning SRHT features and parameters."""

    return apply_block_srht(hidden, params), params

