"""HyperTensor primitive implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np


@dataclass
class HyperTensor:
    """Piecewise-linear interpolant anchored to an oriented hyperplane.

    Parameters
    ----------
    n:
        Normal vector describing the oriented hyperplane. It will be normalized
        during construction to ensure unit length.
    delta:
        Signed offset of the hyperplane in the direction of ``n``.
    dneg, dpos:
        Negative/positive band radii that delimit the interpolation region. The
        values must satisfy ``dneg < 0 < dpos``.
    C:
        Control matrix with three columns ``[V_neg, V_0, V_pos]`` describing the
        responses at the negative band edge, center, and positive band edge.
    tau:
        Locality temperature that controls softmax-based weighting.
    """

    n: np.ndarray
    delta: float
    dneg: float
    dpos: float
    C: np.ndarray
    tau: float = 1.0
    dtype: np.dtype = field(default=np.float32, repr=False)

    def __post_init__(self) -> None:
        self.n = np.asarray(self.n, dtype=self.dtype)
        norm = np.linalg.norm(self.n)
        if norm == 0.0:
            raise ValueError("Normal vector must be non-zero")
        self.n = self.n / norm
        self.delta = float(self.delta)
        self.dneg = float(self.dneg)
        self.dpos = float(self.dpos)
        if not (self.dneg < 0.0 < self.dpos):
            raise ValueError("dneg must be < 0 and dpos must be > 0")
        self.C = np.asarray(self.C, dtype=self.dtype)
        if self.C.ndim != 2 or self.C.shape[1] != 3:
            raise ValueError("C must have shape (output_dim, 3)")
        self.tau = float(self.tau)
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

    @property
    def output_dim(self) -> int:
        return self.C.shape[0]

    def distance(self, x: np.ndarray) -> float:
        """Return the signed distance of ``x`` from the hyperplane."""

        x = np.asarray(x, dtype=self.dtype)
        return float(self.n @ x + self.delta)

    def _alpha(self, d: float) -> Tuple[np.ndarray, float]:
        """Return barycentric weights for a given signed distance.

        The returned distance is clipped to the interpolation band, mirroring the
        formulation in the whitepaper.
        """

        dcl = float(np.clip(d, self.dneg, self.dpos))
        if dcl >= 0.0:
            a_pos = dcl / (self.dpos + 1e-12)
            alpha = np.array([0.0, 1.0 - a_pos, a_pos], dtype=self.dtype)
        else:
            a_neg = -dcl / (self.dneg - 1e-12)
            alpha = np.array([a_neg, 1.0 - a_neg, 0.0], dtype=self.dtype)
        return alpha, dcl

    def local(self, x: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """Evaluate the local interpolant at ``x``.

        Returns
        -------
        L : ndarray
            Local interpolated value.
        d : float
            Signed distance of ``x``.
        alpha : ndarray
            Barycentric coefficients used for interpolation.
        d_clipped : float
            Distance clipped to the interpolation band.
        """

        x = np.asarray(x, dtype=self.dtype)
        d = self.distance(x)
        alpha, dcl = self._alpha(d)
        L = self.C @ alpha
        return L, d, alpha, dcl

    def to_tuple(self) -> Tuple[np.ndarray, float, float, float, np.ndarray, float]:
        """Return a serializable tuple of HyperTensor parameters."""

        return self.n.copy(), self.delta, self.dneg, self.dpos, self.C.copy(), self.tau

    @classmethod
    def from_tuple(
        cls, params: Iterable[np.ndarray | float], dtype: np.dtype = np.float32
    ) -> "HyperTensor":
        n, delta, dneg, dpos, C, tau = params
        return cls(np.array(n, dtype=dtype), delta, dneg, dpos, np.array(C, dtype=dtype), tau)

    def clone(self) -> "HyperTensor":
        """Deep copy the HyperTensor."""

        return HyperTensor(*self.to_tuple(), dtype=self.dtype)

    def renormalize(self) -> None:
        """Ensure that the normal vector remains unit length."""

        norm = np.linalg.norm(self.n)
        if norm == 0.0:
            raise ValueError("Normal became zero during updates")
        self.n = self.n / norm

