"""Interpolation modules for Hypertensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np


@dataclass(frozen=True)
class InterpolationResult:
    """Container describing interpolation evaluation results."""

    value: np.ndarray
    weights: np.ndarray
    clipped_distance: float
    distance_derivative: np.ndarray


class InterpolationModule(Protocol):
    """Protocol implemented by Hypertensor interpolation modules."""

    name: str

    def evaluate(
        self,
        controls: np.ndarray,
        distance: float,
        dneg: float,
        dpos: float,
        reference_radius: float,
    ) -> InterpolationResult:
        """Evaluate the module for ``distance`` using ``controls``."""


def _clip_distance(distance: float, dneg: float, dpos: float) -> float:
    return float(np.clip(distance, dneg, dpos))


def _within_band(distance: float, dneg: float, dpos: float) -> bool:
    return dneg < distance < dpos


class _LerpModule:
    name = "lerp"

    def evaluate(
        self,
        controls: np.ndarray,
        distance: float,
        dneg: float,
        dpos: float,
        reference_radius: float,
    ) -> InterpolationResult:
        dcl = _clip_distance(distance, dneg, dpos)
        weights = np.zeros(3, dtype=np.float32)
        if dcl >= 0.0:
            denom = dpos if dpos != 0.0 else 1e-12
            t = float(np.clip(dcl / denom, 0.0, 1.0))
            weights[1] = 1.0 - t
            weights[2] = t
            dt_dd = 1.0 / denom if _within_band(distance, dneg, dpos) else 0.0
            deriv = (controls[:, 2] - controls[:, 1]) * dt_dd
        else:
            denom = -dneg if dneg != 0.0 else 1e-12
            s = (dcl - dneg) / denom
            s = float(np.clip(s, 0.0, 1.0))
            weights[0] = 1.0 - s
            weights[1] = s
            ds_dd = 1.0 / denom if _within_band(distance, dneg, dpos) else 0.0
            deriv = (controls[:, 1] - controls[:, 0]) * ds_dd
        value = controls @ weights
        return InterpolationResult(
            value=value,
            weights=weights,
            clipped_distance=dcl,
            distance_derivative=deriv.astype(np.float32),
        )


class _HermiteModule:
    name = "hermite"

    def _positive_weights(self, t: float, dneg: float, dpos: float) -> np.ndarray:
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        ratio = dpos / (dneg if dneg != 0.0 else -1e-12)
        w_neg = 0.5 * h10 * ratio
        w_zero = h00 - 0.5 * h10 * ratio - 0.5 * h10 - h11
        w_pos = 0.5 * h10 + h01 + h11
        return np.array([w_neg, w_zero, w_pos], dtype=np.float32)

    def _positive_derivative(
        self,
        controls: np.ndarray,
        t: float,
        dneg: float,
        dpos: float,
        active: bool,
    ) -> np.ndarray:
        if not active:
            return np.zeros(controls.shape[0], dtype=np.float32)
        h00p = 6 * t**2 - 6 * t
        h10p = 3 * t**2 - 4 * t + 1
        h01p = -6 * t**2 + 6 * t
        h11p = 3 * t**2 - 2 * t
        ratio = dpos / (dneg if dneg != 0.0 else -1e-12)
        dw = np.array(
            [
                0.5 * h10p * ratio,
                h00p - 0.5 * h10p * ratio - 0.5 * h10p - h11p,
                0.5 * h10p + h01p + h11p,
            ],
            dtype=np.float64,
        )
        dt_dd = 1.0 / dpos if dpos != 0.0 else 0.0
        return (controls @ (dw * dt_dd)).astype(np.float32)

    def _negative_weights(self, t: float, dneg: float, dpos: float) -> np.ndarray:
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        ratio = (-dneg) / (dpos if dpos != 0.0 else 1e-12)
        w_neg = h00 - h10 - 0.5 * h11
        w_zero = h10 + h01 + 0.5 * h11 + 0.5 * h11 * ratio
        w_pos = 0.5 * h11 * ratio
        return np.array([w_neg, w_zero, w_pos], dtype=np.float32)

    def _negative_derivative(
        self,
        controls: np.ndarray,
        t: float,
        dneg: float,
        dpos: float,
        active: bool,
    ) -> np.ndarray:
        if not active:
            return np.zeros(controls.shape[0], dtype=np.float32)
        h00p = 6 * t**2 - 6 * t
        h10p = 3 * t**2 - 4 * t + 1
        h01p = -6 * t**2 + 6 * t
        h11p = 3 * t**2 - 2 * t
        ratio = (-dneg) / (dpos if dpos != 0.0 else 1e-12)
        dw = np.array(
            [
                h00p - h10p - 0.5 * h11p,
                h10p + h01p + 0.5 * h11p + 0.5 * h11p * ratio,
                0.5 * h11p * ratio,
            ],
            dtype=np.float64,
        )
        dt_dd = 1.0 / (-dneg) if dneg != 0.0 else 0.0
        return (controls @ (dw * dt_dd)).astype(np.float32)

    def evaluate(
        self,
        controls: np.ndarray,
        distance: float,
        dneg: float,
        dpos: float,
        reference_radius: float,
    ) -> InterpolationResult:
        dcl = _clip_distance(distance, dneg, dpos)
        if dcl >= 0.0:
            denom = dpos if dpos != 0.0 else 1e-12
            t = float(np.clip(dcl / denom, 0.0, 1.0))
            weights = self._positive_weights(t, dneg, dpos)
            deriv = self._positive_derivative(controls, t, dneg, dpos, _within_band(distance, dneg, dpos))
        else:
            denom = -dneg if dneg != 0.0 else 1e-12
            s = (dcl - dneg) / denom
            t = float(np.clip(s, 0.0, 1.0))
            weights = self._negative_weights(t, dneg, dpos)
            deriv = self._negative_derivative(controls, t, dneg, dpos, _within_band(distance, dneg, dpos))
        value = controls @ weights
        return InterpolationResult(
            value=value,
            weights=weights,
            clipped_distance=dcl,
            distance_derivative=deriv,
        )


class _CatmullRomModule:
    name = "catmull_rom"

    def _positive(self, t: float) -> np.ndarray:
        w_neg = 0.5 * (-t + 2 * t**2 - t**3)
        w_zero = 0.5 * (2 - 4 * t**2 + 2 * t**3)
        w_pos = 0.5 * (t + 2 * t**2 - t**3)
        return np.array([w_neg, w_zero, w_pos], dtype=np.float32)

    def _positive_derivative(self, controls: np.ndarray, t: float, dpos: float, active: bool) -> np.ndarray:
        if not active:
            return np.zeros(controls.shape[0], dtype=np.float32)
        dw = np.array(
            [
                0.5 * (-1 + 4 * t - 3 * t**2),
                0.5 * (-8 * t + 6 * t**2),
                0.5 * (1 + 4 * t - 3 * t**2),
            ],
            dtype=np.float64,
        )
        dt_dd = 1.0 / dpos if dpos != 0.0 else 0.0
        return (controls @ (dw * dt_dd)).astype(np.float32)

    def _negative(self, t: float) -> np.ndarray:
        w_neg = 0.5 * (2 - 2 * t - t**2 + t**3)
        w_zero = 0.5 * (2 * t + 2 * t**2 - 2 * t**3)
        w_pos = 0.5 * (-t**2 + t**3)
        return np.array([w_neg, w_zero, w_pos], dtype=np.float32)

    def _negative_derivative(self, controls: np.ndarray, t: float, dneg: float, active: bool) -> np.ndarray:
        if not active:
            return np.zeros(controls.shape[0], dtype=np.float32)
        dw = np.array(
            [
                0.5 * (-2 - 2 * t + 3 * t**2),
                0.5 * (2 + 4 * t - 6 * t**2),
                0.5 * (-2 * t + 3 * t**2),
            ],
            dtype=np.float64,
        )
        dt_dd = 1.0 / (-dneg) if dneg != 0.0 else 0.0
        return (controls @ (dw * dt_dd)).astype(np.float32)

    def evaluate(
        self,
        controls: np.ndarray,
        distance: float,
        dneg: float,
        dpos: float,
        reference_radius: float,
    ) -> InterpolationResult:
        dcl = _clip_distance(distance, dneg, dpos)
        if dcl >= 0.0:
            denom = dpos if dpos != 0.0 else 1e-12
            t = float(np.clip(dcl / denom, 0.0, 1.0))
            weights = self._positive(t)
            deriv = self._positive_derivative(controls, t, dpos, _within_band(distance, dneg, dpos))
        else:
            denom = -dneg if dneg != 0.0 else 1e-12
            s = (dcl - dneg) / denom
            t = float(np.clip(s, 0.0, 1.0))
            weights = self._negative(t)
            deriv = self._negative_derivative(controls, t, dneg, _within_band(distance, dneg, dpos))
        value = controls @ weights
        return InterpolationResult(
            value=value,
            weights=weights,
            clipped_distance=dcl,
            distance_derivative=deriv,
        )


class _LocalPolynomialModule:
    name = "local_poly"

    def _weights(self, t: float) -> np.ndarray:
        w_neg = -0.5 * t + 0.5 * t**2
        w_zero = 1.0 - t**2
        w_pos = 0.5 * t + 0.5 * t**2
        return np.array([w_neg, w_zero, w_pos], dtype=np.float32)

    def _derivative(self, controls: np.ndarray, t: float, scale: float, active: bool) -> np.ndarray:
        if not active or scale == 0.0:
            return np.zeros(controls.shape[0], dtype=np.float32)
        dw_dt = np.array([-0.5 + t, -2.0 * t, 0.5 + t], dtype=np.float64)
        dt_dd = 1.0 / scale
        return (controls @ (dw_dt * dt_dd)).astype(np.float32)

    def evaluate(
        self,
        controls: np.ndarray,
        distance: float,
        dneg: float,
        dpos: float,
        reference_radius: float,
    ) -> InterpolationResult:
        dcl = _clip_distance(distance, dneg, dpos)
        if dcl >= 0.0:
            scale = dpos if dpos != 0.0 else 1e-12
        else:
            scale = -dneg if dneg != 0.0 else 1e-12
        t = float(np.clip(dcl / scale, -1.0, 1.0))
        weights = self._weights(t)
        deriv = self._derivative(controls, t, scale, _within_band(distance, dneg, dpos))
        value = controls @ weights
        return InterpolationResult(
            value=value,
            weights=weights,
            clipped_distance=dcl,
            distance_derivative=deriv,
        )


class _WaveletModule:
    name = "wavelet"

    def _weights(self, t: float) -> np.ndarray:
        if t < 0.0:
            psi_neg = float(np.clip(1.0 + 2.0 * t, 0.0, 1.0))
            psi_pos = 0.0
        else:
            psi_neg = 0.0
            psi_pos = float(np.clip(1.0 - 2.0 * t, 0.0, 1.0))
        w_neg = psi_neg
        w_pos = psi_pos
        w_zero = 1.0 - w_neg - w_pos
        return np.array([w_neg, w_zero, w_pos], dtype=np.float32)

    def _derivative(self, controls: np.ndarray, t: float, scale: float, active: bool) -> np.ndarray:
        if not active or scale == 0.0:
            return np.zeros(controls.shape[0], dtype=np.float32)
        if t < 0.0:
            dpsi_dt = 2.0 if -0.5 < t < 0.0 else 0.0
            dw_dt = np.array([dpsi_dt, -dpsi_dt, 0.0], dtype=np.float64)
        else:
            dpsi_dt = -2.0 if 0.0 < t < 0.5 else 0.0
            dw_dt = np.array([0.0, -dpsi_dt, dpsi_dt], dtype=np.float64)
        dt_dd = 1.0 / scale
        return (controls @ (dw_dt * dt_dd)).astype(np.float32)

    def evaluate(
        self,
        controls: np.ndarray,
        distance: float,
        dneg: float,
        dpos: float,
        reference_radius: float,
    ) -> InterpolationResult:
        dcl = _clip_distance(distance, dneg, dpos)
        if dcl >= 0.0:
            scale = max(dpos, reference_radius)
        else:
            scale = max(-dneg, reference_radius)
        if scale == 0.0:
            scale = 1e-12
        t = float(np.clip(dcl / scale, -1.0, 1.0))
        weights = self._weights(t)
        deriv = self._derivative(controls, t, scale, _within_band(distance, dneg, dpos))
        value = controls @ weights
        return InterpolationResult(
            value=value,
            weights=weights,
            clipped_distance=dcl,
            distance_derivative=deriv,
        )


_MODULES: Dict[str, InterpolationModule] = {
    module.name: module
    for module in (
        _LerpModule(),
        _HermiteModule(),
        _CatmullRomModule(),
        _LocalPolynomialModule(),
        _WaveletModule(),
    )
}


def get_interpolation_module(name: str) -> InterpolationModule:
    """Return the interpolation module registered under ``name``."""

    try:
        return _MODULES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown interpolation module '{name}'") from exc


def available_interpolations() -> tuple[str, ...]:
    """Return the names of all registered interpolation modules."""

    return tuple(_MODULES.keys())

