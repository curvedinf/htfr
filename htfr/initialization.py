"""Initialization helpers for the Hypertensor Field Regressor (HTFR)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .tensor import HyperTensor


@dataclass
class ClusterResult:
    centers: np.ndarray
    assignments: np.ndarray


def kmeans(
    data: np.ndarray,
    k: int,
    iters: int = 20,
    rng: np.random.Generator | None = None,
) -> ClusterResult:
    """Simple k-means clustering used for Hypertensor Field Regressor initialization."""

    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=np.float32)
    n, _ = data.shape
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and number of samples")
    indices = rng.choice(n, size=k, replace=False)
    centers = data[indices]
    assignments = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=-1)
        assignments = distances.argmin(axis=1)
        for i in range(k):
            mask = assignments == i
            if not np.any(mask):
                centers[i] = data[rng.integers(n)]
            else:
                centers[i] = data[mask].mean(axis=0)
    return ClusterResult(centers=centers, assignments=assignments)


def principal_direction(data: np.ndarray) -> np.ndarray:
    """Return the leading principal component of ``data``."""

    data = np.asarray(data, dtype=np.float32)
    data = data - data.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(data, full_matrices=False)
    return vh[0]


def initialize_hypertensors(
    data: np.ndarray,
    outputs: np.ndarray,
    k: int,
    tau: float = 1.0,
    reference_radius: float | None = None,
    rng: np.random.Generator | None = None,
) -> List[HyperTensor]:
    """Construct an initial set of HyperTensors following the whitepaper."""

    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=np.float32)
    outputs = np.asarray(outputs, dtype=np.float32)
    if data.shape[0] != outputs.shape[0]:
        raise ValueError("Mismatched data and output lengths")

    clusters = kmeans(data, k=k, rng=rng)
    tensors: List[HyperTensor] = []
    for idx in range(k):
        mask = clusters.assignments == idx
        if not np.any(mask):
            continue
        cluster_points = data[mask]
        cluster_outputs = outputs[mask]
        direction = principal_direction(cluster_points)
        offset = -float(direction @ cluster_points.mean(axis=0))
        projections = cluster_points @ direction + offset
        std = projections.std() + 1e-6
        dpos = float(std)
        dneg = -float(std)
        local_mean = cluster_outputs.mean(axis=0)
        C = np.stack(
            [local_mean, local_mean, local_mean], axis=-1
        ).astype(np.float32)
        tensors.append(
            HyperTensor(
                direction,
                offset,
                dneg,
                dpos,
                C,
                tau=tau,
                reference_radius=reference_radius if reference_radius is not None else 5.0,
            )
        )
    return tensors

