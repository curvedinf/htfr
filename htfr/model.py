"""Hypertensor Field Regressor model and training utilities."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from .interpolation import available_interpolations
from .initialization import random_hypertensors, reseed_tensor_around_sample
from .tensor import HyperTensor, LocalResult

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
    top_k: int = 32
    weight_mode: WeightMode = "softmax"
    epsilon: float = 1e-6
    eta: float = 0.05
    eta_g: float = 0.005
    max_knn_radius: float = 1.0
    locality_radius: float | None = None
    interpolation_reference: float | None = None
    interpolation_weights: Dict[str, float] = field(default_factory=dict)
    randomize_interpolations: bool = True
    rng: Generator = field(default_factory=default_rng, repr=False)
    dtype: np.dtype = field(default=np.float16, repr=False)
    error_threshold: float = 2.0
    max_error_queue: int = 128
    relocation_interval: int = 32
    usage_decay: float = 0.995

    def __post_init__(self) -> None:
        if self.interpolation_reference is None:
            self.interpolation_reference = 5.0 * self.max_knn_radius
        if self.locality_radius is None:
            self.locality_radius = self.interpolation_reference
        if not self.interpolation_weights:
            self.interpolation_weights = {
                name: 1.0 for name in available_interpolations()
            }
        else:
            for name in available_interpolations():
                self.interpolation_weights.setdefault(name, 0.0)
        for tensor in self.tensors:
            self._configure_tensor(tensor)
        if self.randomize_interpolations and self.tensors:
            self.randomize_interpolations_by_weight()
        else:
            for tensor in self.tensors:
                tensor.reference_radius = self.interpolation_reference
        self._usage_counts = np.zeros(len(self.tensors), dtype=np.float32)
        self._avg_distance = np.zeros(len(self.tensors), dtype=np.float32)
        self._loss_trace = np.zeros(len(self.tensors), dtype=np.float32)
        self._error_queue: Deque[tuple[np.ndarray, float]] = deque(
            maxlen=self.max_error_queue
        )

    def _configure_tensor(self, tensor: HyperTensor) -> None:
        tensor.reference_radius = self.interpolation_reference
        if tensor.dtype != self.dtype:
            tensor.dtype = self.dtype

    def _interpolation_probabilities(self) -> Tuple[List[str], np.ndarray]:
        names = list(self.interpolation_weights.keys())
        weights = np.array(
            [max(float(self.interpolation_weights[name]), 0.0) for name in names],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("Interpolation weights must sum to a positive value")
        return names, (weights / total)

    def randomize_interpolations_by_weight(self) -> None:
        """Randomly assign interpolation modules to each HyperTensor."""

        if not self.tensors:
            return
        names, probs = self._interpolation_probabilities()
        for tensor in self.tensors:
            tensor.reference_radius = self.interpolation_reference
            tensor.interpolation = str(self.rng.choice(names, p=probs))

    def add_tensor(self, tensor: HyperTensor) -> None:
        self._configure_tensor(tensor)
        if self.randomize_interpolations:
            names, probs = self._interpolation_probabilities()
            tensor.interpolation = str(self.rng.choice(names, p=probs))
        else:
            tensor.reference_radius = self.interpolation_reference
        self.tensors.append(tensor)
        self._append_stat_slots(1)

    def seed_random_tensors(
        self,
        count: int,
        input_dim: int,
        output_dim: int,
        tau: float = 1.0,
        reference_radius: float | None = None,
    ) -> None:
        """Append ``count`` randomly initialized HyperTensors."""

        reference = (
            reference_radius if reference_radius is not None else self.interpolation_reference
        )
        new_tensors = random_hypertensors(
            count,
            input_dim,
            output_dim,
            tau=tau,
            reference_radius=reference,
            rng=self.rng,
            dtype=self.dtype,
        )
        for tensor in new_tensors:
            self.add_tensor(tensor)

    @property
    def output_dim(self) -> int:
        if not self.tensors:
            raise ValueError("Model has no HyperTensors")
        return self.tensors[0].output_dim

    def _select_active(self, x: np.ndarray) -> List[Tuple[int, LocalResult]]:
        cache: List[Tuple[int, LocalResult]] = []
        fallback: List[Tuple[int, LocalResult]] = []
        for idx, tensor in enumerate(self.tensors):
            local = tensor.local(x)
            if (
                self.locality_radius is not None
                and abs(local.distance) > self.locality_radius
            ):
                fallback.append((idx, local))
            else:
                cache.append((idx, local))
        if not cache:
            cache = fallback
        if not cache:
            raise ValueError("No HyperTensors available")
        k = min(self.top_k, len(cache))
        # select indices of k smallest absolute distances
        absd = np.array([abs(entry[1].distance) for entry in cache])
        topk_indices = np.argpartition(absd, kth=k - 1)[:k]
        active = [cache[i] for i in topk_indices]
        # sort for determinism
        active.sort(key=lambda item: abs(item[1].distance))
        return active

    def predict(self, x: np.ndarray) -> np.ndarray:
        active = self._select_active(np.asarray(x, dtype=np.float32))
        distances = [entry[1].distance for entry in active]
        taus = [self.tensors[idx].tau for idx, _ in active]
        loc_weights = locality_weights(distances, taus, self.weight_mode, self.epsilon)
        outputs = np.stack([entry[1].value for entry in active]).astype(np.float32)
        yhat = loc_weights @ outputs
        self._record_activity([idx for idx, _ in active], distances, loss_value=None)
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
        x_arr = np.asarray(x, dtype=np.float32)
        active = self._select_active(x_arr)
        distances = [entry[1].distance for entry in active]
        taus = [self.tensors[idx].tau for idx, _ in active]
        loc_weights = locality_weights(distances, taus, self.weight_mode, self.epsilon)
        outputs = np.stack([entry[1].value for entry in active]).astype(np.float32)
        yhat = loc_weights @ outputs
        loss_value = None
        if train:
            if isinstance(y, (int, np.integer)):
                y_arr = np.zeros_like(yhat, dtype=np.float32)
                target_index = int(np.asarray(y).item())
                y_arr[target_index] = 1.0
            else:
                y_arr = np.asarray(y, dtype=np.float32)
                target_index = None
            loss_value = self._compute_loss_value(yhat, y_arr, target_index, loss)
            grad = self._loss_gradient(yhat.astype(np.float32), y_arr, loss).astype(np.float32)
            for loc_w, (tensor_idx, local) in zip(loc_weights, active):
                tensor = self.tensors[tensor_idx]
                control = tensor.C.astype(np.float32)
                control -= self.eta * loc_w * np.outer(grad, local.weights.astype(np.float32))
                tensor.C = control.astype(tensor.dtype)
                if np.any(local.distance_derivative):
                    coeff = float(grad @ local.distance_derivative.astype(np.float32))
                    normal = tensor.n.astype(np.float32)
                    normal -= self.eta_g * loc_w * coeff * x_arr
                    tensor.n = normal.astype(tensor.dtype)
                    tensor.renormalize()
                    tensor.delta = np.dtype(tensor.dtype).type(
                        float(tensor.delta) - self.eta_g * loc_w * coeff
                    )
        self._record_activity([idx for idx, _ in active], distances, loss_value)
        if loss_value is not None:
            self._enqueue_error(x_arr, loss_value)
        return yhat.astype(self.dtype)

    def _compute_loss_value(
        self,
        yhat: np.ndarray,
        y_arr: np.ndarray,
        target_index: int | None,
        mode: LossMode,
    ) -> float:
        yhat32 = np.asarray(yhat, dtype=np.float32)
        if mode == "mse":
            return float(np.mean((yhat32 - y_arr) ** 2))
        if mode == "logits_ce":
            max_val = float(np.max(yhat32))
            log_sum = max_val + math.log(np.exp(yhat32 - max_val).sum())
            log_probs = yhat32 - log_sum
            if target_index is not None:
                return float(-log_probs[target_index])
            return float(-np.sum(y_arr * log_probs))
        raise ValueError(f"Unsupported loss mode: {mode}")

    def _append_stat_slots(self, count: int) -> None:
        if count <= 0:
            return
        self._usage_counts = np.concatenate(
            [self._usage_counts, np.zeros(count, dtype=np.float32)]
        )
        self._avg_distance = np.concatenate(
            [self._avg_distance, np.zeros(count, dtype=np.float32)]
        )
        self._loss_trace = np.concatenate(
            [self._loss_trace, np.zeros(count, dtype=np.float32)]
        )

    def _record_activity(
        self, indices: Sequence[int], distances: Sequence[float], loss_value: float | None
    ) -> None:
        for tensor_idx, distance in zip(indices, distances):
            self._usage_counts[tensor_idx] = (
                self.usage_decay * self._usage_counts[tensor_idx] + 1.0
            )
            self._avg_distance[tensor_idx] = (
                self.usage_decay * self._avg_distance[tensor_idx]
                + (1.0 - self.usage_decay) * abs(distance)
            )
            if loss_value is not None:
                self._loss_trace[tensor_idx] = (
                    self.usage_decay * self._loss_trace[tensor_idx]
                    + (1.0 - self.usage_decay) * loss_value
                )

    def _enqueue_error(self, x: np.ndarray, loss_value: float) -> None:
        if loss_value < self.error_threshold:
            return
        self._error_queue.append((np.asarray(x, dtype=self.dtype), loss_value))
        if len(self._error_queue) >= self.relocation_interval:
            self._relocate_idle_tensor()

    def _relocate_idle_tensor(self) -> None:
        if not self.tensors or not self._error_queue:
            return
        idle_idx = int(np.argmin(self._usage_counts))
        sample_idx = max(range(len(self._error_queue)), key=lambda idx: self._error_queue[idx][1])
        sample, _ = self._error_queue[sample_idx]
        # Remove the sample from the deque by rebuilding order-preserving list.
        updated = deque(maxlen=self.max_error_queue)
        for idx, entry in enumerate(self._error_queue):
            if idx == sample_idx:
                continue
            updated.append(entry)
        self._error_queue = updated
        reseed_tensor_around_sample(self.tensors[idle_idx], sample.astype(np.float32), rng=self.rng)
        self._usage_counts[idle_idx] = 1.0
        self._avg_distance[idle_idx] = 0.0
        self._loss_trace[idle_idx] = 0.0

    @classmethod
    def from_tensors(
        cls,
        tensors: Iterable[HyperTensor],
        top_k: int = 4,
        weight_mode: WeightMode = "softmax",
        epsilon: float = 1e-6,
        eta: float = 0.05,
        eta_g: float = 0.005,
        dtype: np.dtype = np.float16,
        max_knn_radius: float = 1.0,
        locality_radius: float | None = None,
        interpolation_reference: float | None = None,
        interpolation_weights: Dict[str, float] | None = None,
        randomize_interpolations: bool = True,
        rng: Generator | None = None,
        error_threshold: float = 2.0,
        max_error_queue: int = 128,
        relocation_interval: int = 32,
        usage_decay: float = 0.995,
    ) -> "HTFRModel":
        model = cls(
            tensors=list(tensors),
            top_k=top_k,
            weight_mode=weight_mode,
            epsilon=epsilon,
            eta=eta,
            eta_g=eta_g,
            dtype=dtype,
            max_knn_radius=max_knn_radius,
            locality_radius=locality_radius,
            interpolation_reference=interpolation_reference,
            interpolation_weights=interpolation_weights or {},
            randomize_interpolations=randomize_interpolations,
            rng=rng if rng is not None else default_rng(),
            error_threshold=error_threshold,
            max_error_queue=max_error_queue,
            relocation_interval=relocation_interval,
            usage_decay=usage_decay,
        )
        return model
