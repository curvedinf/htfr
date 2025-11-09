"""Hypertensor Field Regressor model and training utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np

from .tensor import HyperTensor

WeightMode = Literal["softmax", "inverse"]
LossMode = Literal["mse", "logits_ce"]


def locality_weights(
    distances: Sequence[float],
    taus: Sequence[float],
    mode: WeightMode = "softmax",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute locality weights from distances.

    Parameters
    ----------
    distances:
        Signed distances for the active HyperTensors.
    taus:
        Temperature parameters for softmax weighting.
    mode:
        Either ``"softmax"`` or ``"inverse"`` to select the weighting rule.
    epsilon:
        Small constant for numerical stability with inverse weighting.
    """

    absd = np.abs(np.asarray(distances, dtype=np.float64))
    taus = np.asarray(taus, dtype=np.float64)
    if mode == "softmax":
        logits = -absd / (taus + 1e-12)
        logits -= logits.max(initial=0.0)
        weights = np.exp(logits)
    elif mode == "inverse":
        weights = 1.0 / (absd + epsilon)
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")
    weights_sum = weights.sum()
    if weights_sum == 0.0:
        raise ValueError("All locality weights collapsed to zero")
    return (weights / weights_sum).astype(np.float32)


@dataclass
class HTFRModel:
    """Hypertensor Field Regressor model."""

    tensors: List[HyperTensor] = field(default_factory=list)
    top_k: int = 4
    weight_mode: WeightMode = "softmax"
    epsilon: float = 1e-6
    eta: float = 0.05
    eta_g: float = 0.005
    dtype: np.dtype = field(default=np.float32, repr=False)

    def add_tensor(self, tensor: HyperTensor) -> None:
        self.tensors.append(tensor)

    @property
    def output_dim(self) -> int:
        if not self.tensors:
            raise ValueError("Model has no HyperTensors")
        return self.tensors[0].output_dim

    def _select_active(self, x: np.ndarray) -> List[Tuple[int, Tuple]]:
        cache: List[Tuple[int, Tuple]] = []
        for idx, tensor in enumerate(self.tensors):
            local = tensor.local(x)
            cache.append((idx, local))
        if not cache:
            raise ValueError("No HyperTensors available")
        k = min(self.top_k, len(cache))
        # select indices of k smallest absolute distances
        absd = np.array([abs(entry[1][1]) for entry in cache])
        topk_indices = np.argpartition(absd, kth=k - 1)[:k]
        active = [cache[i] for i in topk_indices]
        # sort for determinism
        active.sort(key=lambda item: abs(item[1][1]))
        return active

    def predict(self, x: np.ndarray) -> np.ndarray:
        active = self._select_active(np.asarray(x, dtype=self.dtype))
        distances = [entry[1][1] for entry in active]
        taus = [self.tensors[idx].tau for idx, _ in active]
        weights = locality_weights(distances, taus, self.weight_mode, self.epsilon)
        outputs = np.stack([entry[1][0] for entry in active])
        yhat = weights @ outputs
        return yhat.astype(self.dtype)

    def _loss_gradient(self, yhat: np.ndarray, y: np.ndarray, mode: LossMode) -> np.ndarray:
        if mode == "mse":
            return yhat - y
        if mode == "logits_ce":
            # y is expected to be the target class index or probability vector
            if y.ndim == 0 or (y.ndim == 1 and y.shape[0] == 1):
                # scalar class index
                target = int(np.asarray(y).item())
                probs = np.exp(yhat - yhat.max())
                probs /= probs.sum()
                grad = probs
                grad[target] -= 1.0
                return grad
            y = np.asarray(y, dtype=self.dtype)
            probs = np.exp(yhat - yhat.max())
            probs /= probs.sum()
            return probs - y
        raise ValueError(f"Unsupported loss mode: {mode}")

    def predict_and_update(
        self,
        x: np.ndarray,
        y: np.ndarray | int,
        loss: LossMode = "mse",
        train: bool = True,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype)
        active = self._select_active(x)
        distances = [entry[1][1] for entry in active]
        taus = [self.tensors[idx].tau for idx, _ in active]
        weights = locality_weights(distances, taus, self.weight_mode, self.epsilon)
        outputs = np.stack([entry[1][0] for entry in active])
        yhat = weights @ outputs
        if not train:
            return yhat.astype(self.dtype)

        if isinstance(y, (int, np.integer)):
            y_arr = np.zeros_like(yhat)
            y_arr[int(y)] = 1.0
            grad = self._loss_gradient(yhat, y_arr, loss)
        else:
            y_arr = np.asarray(y, dtype=self.dtype)
            grad = self._loss_gradient(yhat, y_arr, loss)

        for w, (tensor_idx, local) in zip(weights, active):
            tensor = self.tensors[tensor_idx]
            L, d, alpha, _ = local
            controls_before = tensor.C.copy()
            tensor.C -= self.eta * w * np.outer(grad, alpha)
            dLd = 0.0
            if 0.0 < d < tensor.dpos:
                v_pos = controls_before[:, 2]
                v_zero = controls_before[:, 1]
                dLd = (v_pos - v_zero) / (tensor.dpos + 1e-12)
            elif tensor.dneg < d < 0.0:
                v_neg = controls_before[:, 0]
                v_zero = controls_before[:, 1]
                dLd = (v_neg - v_zero) / (tensor.dneg - 1e-12)
            if isinstance(dLd, np.ndarray):
                coeff = float(grad @ dLd)
                tensor.n -= self.eta_g * w * coeff * x
                tensor.renormalize()
                tensor.delta -= self.eta_g * w * coeff
        return yhat.astype(self.dtype)

    @classmethod
    def from_tensors(
        cls,
        tensors: Iterable[HyperTensor],
        top_k: int = 4,
        weight_mode: WeightMode = "softmax",
        epsilon: float = 1e-6,
        eta: float = 0.05,
        eta_g: float = 0.005,
        dtype: np.dtype = np.float32,
    ) -> "HTFRModel":
        model = cls(
            tensors=list(tensors),
            top_k=top_k,
            weight_mode=weight_mode,
            epsilon=epsilon,
            eta=eta,
            eta_g=eta_g,
            dtype=dtype,
        )
        return model

