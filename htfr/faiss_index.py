"""FAISS index helpers for Hypertensor neighbor lookup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _import_faiss() -> object | None:
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError:  # pragma: no cover - exercised when faiss absent
        return None


@dataclass
class FaissIndex:
    """Thin wrapper that prefers FAISS but falls back to NumPy search."""

    dim: int | None = None
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64

    def __post_init__(self) -> None:
        self._faiss = _import_faiss()
        self._index: object | None = None
        self._data = np.zeros((0, self.dim or 0), dtype=np.float16)

    @property
    def size(self) -> int:
        return int(self._data.shape[0])

    def rebuild(self, data: np.ndarray) -> None:
        """Recreate the underlying index from ``data``."""

        arr = np.asarray(data, dtype=np.float16)
        if arr.ndim != 2:
            raise ValueError("FAISS data must be 2-D")
        if self.dim is None:
            self.dim = arr.shape[1]
        if arr.shape[1] != (self.dim or 0):
            raise ValueError("Dimension mismatch when rebuilding FAISS index")
        self._data = arr.copy()
        if self._faiss is not None:
            index = self._faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
            index.hnsw.efConstruction = int(self.ef_construction)
            index.hnsw.efSearch = int(self.ef_search)
            if arr.size:
                index.add(self._data.astype(np.float32))
            self._index = index
        else:  # pragma: no cover - deterministic but redundant with numpy fallback tests
            self._index = None

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return squared distances and indices of the nearest neighbors."""

        if self.dim is None:
            raise ValueError("FAISS index dimension is undefined")
        if k <= 0 or self.size == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )
        q = np.asarray(query, dtype=np.float16).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError("Query dimension does not match FAISS index")
        limit = min(k, self.size)
        if self._index is not None:
            self._index.hnsw.efSearch = int(max(self.ef_search, limit))
            distances, indices = self._index.search(q.astype(np.float32), limit)
            return distances[0], indices[0]
        # NumPy fallback for environments without faiss
        diffs = self._data.astype(np.float32) - q.astype(np.float32)
        dists = np.sum(diffs * diffs, axis=1).astype(np.float32)
        if limit == self.size:
            order = np.argsort(dists)
        else:
            topk = np.argpartition(dists, kth=limit - 1)[:limit]
            order = topk[np.argsort(dists[topk])]
        return dists[order], order.astype(np.int64)
