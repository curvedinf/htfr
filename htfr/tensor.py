"""HyperTensor primitive implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np

from .interpolation import InterpolationResult, available_interpolations, get_interpolation_module


@dataclass
class LocalResult:
    """Container returned by :meth:`HyperTensor.local`."""

    value: np.ndarray
    distance: float
    weights: np.ndarray
    clipped_distance: float
    distance_derivative: np.ndarray

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
    interpolation: str = "lerp"
    reference_radius: float = 5.0
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
        if self.interpolation not in available_interpolations():
            raise ValueError(f"Unknown interpolation module '{self.interpolation}'")
        self.reference_radius = float(self.reference_radius)
        if self.reference_radius <= 0.0:
            raise ValueError("reference_radius must be positive")

    @property
    def output_dim(self) -> int:
        return self.C.shape[0]

    def distance(self, x: np.ndarray) -> float:
        """Return the signed distance of ``x`` from the hyperplane."""

        x = np.asarray(x, dtype=self.dtype)
        return float(self.n @ x + self.delta)

    def local(self, x: np.ndarray) -> LocalResult:
        """Evaluate the local interpolant at ``x`` using the configured module."""

        x = np.asarray(x, dtype=self.dtype)
        d = self.distance(x)
        module = get_interpolation_module(self.interpolation)
        result: InterpolationResult = module.evaluate(
            self.C, d, self.dneg, self.dpos, self.reference_radius
        )
        return LocalResult(
            value=result.value.astype(self.dtype, copy=False),
            distance=d,
            weights=result.weights.astype(self.dtype, copy=False),
            clipped_distance=result.clipped_distance,
            distance_derivative=result.distance_derivative.astype(self.dtype, copy=False),
        )

    def to_tuple(
        self,
    ) -> Tuple[np.ndarray, float, float, float, np.ndarray, float, str, float]:
        """Return a serializable tuple of HyperTensor parameters."""

        return (
            self.n.copy(),
            self.delta,
            self.dneg,
            self.dpos,
            self.C.copy(),
            self.tau,
            self.interpolation,
            self.reference_radius,
        )

    @classmethod
    def from_tuple(
        cls, params: Iterable[np.ndarray | float], dtype: np.dtype = np.float32
    ) -> "HyperTensor":
        items = list(params)
        if len(items) == 6:
            n, delta, dneg, dpos, C, tau = items
            interpolation = "lerp"
            reference_radius = 5.0
        elif len(items) >= 8:
            n, delta, dneg, dpos, C, tau, interpolation, reference_radius = items[:8]
        else:
            raise ValueError("Invalid HyperTensor tuple")
        return cls(
            np.array(n, dtype=dtype),
            delta,
            dneg,
            dpos,
            np.array(C, dtype=dtype),
            tau,
            interpolation=str(interpolation),
            reference_radius=float(reference_radius),
            dtype=dtype,
        )

    def clone(self) -> "HyperTensor":
        """Deep copy the HyperTensor."""

        return HyperTensor(*self.to_tuple(), dtype=self.dtype)

    def renormalize(self) -> None:
        """Ensure that the normal vector remains unit length."""

        norm = np.linalg.norm(self.n)
        if norm == 0.0:
            raise ValueError("Normal became zero during updates")
        self.n = self.n / norm

