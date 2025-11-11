"""Feature transformations used by HTFR training and inference."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

_DISABLE_NUMBA = os.environ.get("HTFR_DISABLE_NUMBA", "").lower() in {"1", "true", "yes"}
if not _DISABLE_NUMBA:  # pragma: no cover - optional dependency
    try:
        from numba import njit, prange
    except Exception:  # pragma: no cover - fallback path
        njit = None
        prange = range
else:  # pragma: no cover - explicit fallback
    njit = None
    prange = range


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

    signs: np.ndarray  # shape (tiles, padded_dim)
    permutation: np.ndarray  # shape (tiles, padded_dim)
    input_dim: int
    padded_dim: int
    target_dim: int
    block_size: int
    block_eps: float
    scale: float
    tile_count: int = 1

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
            "srht_tile_count": np.array([self.tile_count], dtype=np.int32),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "SRHTParameters":
        signs = np.asarray(arrays["srht_signs"], dtype=np.float32)
        permutation = np.asarray(arrays["srht_permutation"], dtype=np.int64)
        if signs.ndim == 1:
            signs = signs[None, :]
        if permutation.ndim == 1:
            permutation = permutation[None, :]
        tile_count = int(
            np.asarray(arrays.get("srht_tile_count", np.array([signs.shape[0]]))).item()
        )
        return cls(
            signs=signs,
            permutation=permutation,
            input_dim=int(np.asarray(arrays["srht_input_dim"]).item()),
            padded_dim=int(np.asarray(arrays["srht_padded_dim"]).item()),
            target_dim=int(np.asarray(arrays["srht_target_dim"]).item()),
            block_size=int(np.asarray(arrays["srht_block_size"]).item()),
            block_eps=float(np.asarray(arrays["srht_block_eps"]).item()),
            scale=float(np.asarray(arrays["srht_scale"]).item()),
            tile_count=tile_count,
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
    tile_count = int(math.ceil(target_dim / padded_dim))
    signs = np.empty((tile_count, padded_dim), dtype=np.float32)
    permutation = np.empty((tile_count, padded_dim), dtype=np.int64)
    for idx in range(tile_count):
        signs[idx] = rng.choice([-1.0, 1.0], size=padded_dim).astype(np.float32)
        permutation[idx] = rng.permutation(padded_dim)
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
        tile_count=tile_count,
    )


def apply_block_srht(data: np.ndarray, params: SRHTParameters) -> np.ndarray:
    """Apply the SRHT projection to ``data`` and return the transformed features."""

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != params.input_dim:
        raise ValueError(
            f"Expected data with shape (N, {params.input_dim}), got {data.shape}"
        )
    signs = np.ascontiguousarray(params.signs, dtype=np.float32)
    permutation = np.ascontiguousarray(params.permutation, dtype=np.int64)
    if njit is not None:
        transformed = _apply_srht_numba(
            data,
            signs,
            permutation,
            params.input_dim,
            params.padded_dim,
            params.target_dim,
            params.tile_count,
            params.scale,
        )
    else:
        transformed = _apply_srht_numpy(
            data,
            signs,
            permutation,
            params.input_dim,
            params.padded_dim,
            params.target_dim,
            params.tile_count,
            params.scale,
        )
    return block_rmsnorm(transformed, params.block_size, params.block_eps)


def _apply_srht_numpy(
    data: np.ndarray,
    signs: np.ndarray,
    permutation: np.ndarray,
    input_dim: int,
    padded_dim: int,
    target_dim: int,
    tile_count: int,
    scale: float,
) -> np.ndarray:
    """Reference SRHT implementation for environments without numba."""

    n = data.shape[0]
    transformed = np.empty((n, target_dim), dtype=np.float32)
    for idx in range(n):
        base = np.zeros(padded_dim, dtype=np.float32)
        base[:input_dim] = data[idx, :input_dim]
        offset = 0
        for tile in range(tile_count):
            vec = base.copy()
            vec *= signs[tile]
            _fwht_inplace(vec)
            perm = permutation[tile]
            take = min(padded_dim, target_dim - offset)
            transformed[idx, offset : offset + take] = vec[perm[:take]] * scale
            offset += take
            if offset >= target_dim:
                break
    return transformed


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


@dataclass(frozen=True)
class CountSketchParameters:
    """Parameters describing a CountSketch projection."""

    bucket_indices: np.ndarray  # shape (input_dim,)
    signs: np.ndarray  # shape (input_dim,)
    input_dim: int
    output_dim: int

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "cs_bucket_indices": self.bucket_indices,
            "cs_signs": self.signs,
            "cs_input_dim": np.array([self.input_dim], dtype=np.int32),
            "cs_output_dim": np.array([self.output_dim], dtype=np.int32),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "CountSketchParameters":
        return cls(
            bucket_indices=np.asarray(arrays["cs_bucket_indices"], dtype=np.int64),
            signs=np.asarray(arrays["cs_signs"], dtype=np.float32),
            input_dim=int(np.asarray(arrays["cs_input_dim"]).item()),
            output_dim=int(np.asarray(arrays["cs_output_dim"]).item()),
        )


def make_count_sketch(
    input_dim: int,
    output_dim: int,
    rng: np.random.Generator | None = None,
) -> CountSketchParameters:
    """Construct parameters for a CountSketch projection."""

    if rng is None:
        rng = np.random.default_rng()
    bucket_indices = rng.integers(low=0, high=output_dim, size=input_dim, dtype=np.int64)
    signs = rng.choice([-1.0, 1.0], size=input_dim).astype(np.float32)
    return CountSketchParameters(bucket_indices=bucket_indices, signs=signs, input_dim=input_dim, output_dim=output_dim)


def apply_count_sketch(data: np.ndarray, params: CountSketchParameters) -> np.ndarray:
    """Apply CountSketch to ``data`` with shape (N, input_dim)."""

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != params.input_dim:
        raise ValueError(
            f"Expected data with shape (N, {params.input_dim}), got {data.shape}"
        )
    scaled = data * params.signs
    output = np.zeros((data.shape[0], params.output_dim), dtype=np.float32)
    for idx in range(params.input_dim):
        output[:, params.bucket_indices[idx]] += scaled[:, idx]
    return output


def _stable_hash(values: Sequence[int], seed: int) -> int:
    acc = 0x9E3779B185EBCA87 ^ seed
    for value in values:
        acc ^= (value + 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
        acc = (acc * 0xC2B2AE3D27D4EB4F + 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF
    return acc


def hashed_ngram_features(
    tokens: Sequence[int],
    dim: int,
    ngram: int = 2,
    num_hashes: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Return hashed n-gram indicator features for ``tokens``."""

    vec = np.zeros(dim, dtype=np.float32)
    if ngram <= 0 or len(tokens) < ngram:
        return vec
    limit = len(tokens) - ngram + 1
    for start in range(limit):
        window = tokens[start : start + ngram]
        for h in range(num_hashes):
            bucket = _stable_hash(window, seed + h) % dim
            vec[bucket] += 1.0
    return vec


class ProjectionStack:
    """Implements CountSketch → SRHT → optional PCA projections."""

    def __init__(
        self,
        raw_dim: int,
        srht_params: Sequence[SRHTParameters],
        countsketch: CountSketchParameters | None = None,
        post_linear: np.ndarray | None = None,
        output_dtype: np.dtype = np.float16,
    ) -> None:
        if not srht_params:
            raise ValueError("ProjectionStack requires at least one SRHT parameter set")
        self.raw_dim = int(raw_dim)
        self.srht_params = tuple(srht_params)
        self.countsketch = countsketch
        self.post_linear = None if post_linear is None else np.asarray(post_linear, dtype=np.float32)
        self.output_dtype = output_dtype
        self.output_dim = int(sum(params.target_dim for params in self.srht_params))

    def _prepare_input(self, vector: np.ndarray, extra: np.ndarray | None) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.raw_dim:
            raise ValueError(f"Expected raw feature dimension {self.raw_dim}, got {vec.shape}")
        if extra is not None:
            if self.countsketch is None:
                raise ValueError("Extra features require a CountSketch stage")
            extra_vec = np.asarray(extra, dtype=np.float32)
            vec = np.concatenate([vec, extra_vec], axis=0)
        return vec

    def project(self, vector: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        """Project a single feature vector through the configured stack."""

        prepared = self._prepare_input(vector, extra)
        if self.countsketch is not None and prepared.shape[0] != self.countsketch.input_dim:
            raise ValueError(
                f"CountSketch expects input dim {self.countsketch.input_dim}, got {prepared.shape[0]}"
            )
        stage = prepared[None, :]
        if self.countsketch is not None:
            stage = apply_count_sketch(stage, self.countsketch)
        srht_views = [apply_block_srht(stage, params) for params in self.srht_params]
        merged = np.concatenate(srht_views, axis=-1)
        if self.post_linear is not None:
            merged = merged @ self.post_linear
        return merged[0].astype(self.output_dtype)

    def project_batch(
        self, vectors: np.ndarray, extras: np.ndarray | None = None
    ) -> np.ndarray:
        """Project a batch of feature vectors."""

        batch = np.asarray(vectors, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != self.raw_dim:
            raise ValueError(
                f"Expected batch with shape (N, {self.raw_dim}), got {batch.shape}"
            )
        if extras is not None:
            extras_arr = np.asarray(extras, dtype=np.float32)
            if extras_arr.ndim != 2 or extras_arr.shape[0] != batch.shape[0]:
                raise ValueError(
                    "Extras must be a 2-D array with matching batch size"
                )
            batch = np.concatenate([batch, extras_arr], axis=-1)
        if self.countsketch is not None and batch.shape[1] != self.countsketch.input_dim:
            raise ValueError(
                f"CountSketch expects input dim {self.countsketch.input_dim}, got {batch.shape[1]}"
            )
        stage = batch
        if self.countsketch is not None:
            stage = apply_count_sketch(stage, self.countsketch)
        srht_views = [apply_block_srht(stage, params) for params in self.srht_params]
        merged = np.concatenate(srht_views, axis=-1)
        if self.post_linear is not None:
            merged = merged @ self.post_linear
        return merged.astype(self.output_dtype)

    def update_post_linear(self, matrix: np.ndarray) -> None:
        """Install/replace the optional PCA/whitening linear transform."""

        self.post_linear = np.asarray(matrix, dtype=np.float32)


def srht_feature_tuple(
    hidden: np.ndarray, params: SRHTParameters, output_dtype: np.dtype = np.float16
) -> Tuple[np.ndarray, SRHTParameters]:
    """Convenience wrapper returning SRHT features and parameters."""

    projected = apply_block_srht(hidden, params)
    return projected.astype(output_dtype), params


if njit is not None:
    from math import sqrt

    @njit(cache=True)
    def _fwht_inplace_numba(vec: np.ndarray) -> None:
        """In-place FWHT with numba acceleration."""

        n = vec.shape[0]
        h = 1
        while h < n:
            step = h * 2
            for start in range(0, n, step):
                for offset in range(h):
                    i = start + offset
                    j = i + h
                    x = vec[i]
                    y = vec[j]
                    vec[i] = x + y
                    vec[j] = x - y
            h *= 2
        norm = sqrt(float(n))
        for idx in range(n):
            vec[idx] = vec[idx] / norm

    @njit(parallel=True, cache=True)
    def _apply_srht_numba(
        data: np.ndarray,
        signs: np.ndarray,
        permutation: np.ndarray,
        input_dim: int,
        padded_dim: int,
        target_dim: int,
        tile_count: int,
        scale: float,
    ) -> np.ndarray:
        n = data.shape[0]
        transformed = np.empty((n, target_dim), dtype=np.float32)
        for row in prange(n):
            base = np.zeros(padded_dim, dtype=np.float32)
            for i in range(input_dim):
                base[i] = data[row, i]
            offset = 0
            for tile in range(tile_count):
                vec = base.copy()
                tile_signs = signs[tile]
                for i in range(padded_dim):
                    vec[i] = vec[i] * tile_signs[i]
                _fwht_inplace_numba(vec)
                perm = permutation[tile]
                take = padded_dim
                remaining = target_dim - offset
                if remaining < take:
                    take = remaining
                for i in range(take):
                    transformed[row, offset + i] = vec[perm[i]] * scale
                offset += take
                if offset >= target_dim:
                    break
        return transformed
else:  # pragma: no cover - executed only when numba is absent

    def _apply_srht_numba(*_args, **_kwargs) -> np.ndarray:
        raise RuntimeError("Numba acceleration requested but numba is not installed.")
