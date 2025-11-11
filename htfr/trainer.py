"""Training helpers for the Hypertensor Field Transformer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from .context import ContextBuilder, ContextSample
from .hypertensor_field_transformer import HypertensorFieldTransformer


@dataclass(frozen=True)
class TrainingMetrics:
    """Aggregated metrics for a training/evaluation pass."""

    loss: float
    perplexity: float
    stage1_loss: float
    samples: int


@dataclass
class MetricLog:
    """Running log used for benchmarking."""

    steps: List[int] = field(default_factory=list)
    perplexities: List[float] = field(default_factory=list)

    def append(self, step: int, perplexity: float) -> None:
        self.steps.append(step)
        self.perplexities.append(perplexity)


class HTFTTrainer:
    """High-level trainer coordinating both stages."""

    def __init__(self, model: HypertensorFieldTransformer, builder: ContextBuilder) -> None:
        self.model = model
        self.builder = builder
        self.metric_log = MetricLog()
        self._steps = 0

    def train_epoch(self, samples: Sequence[ContextSample]) -> TrainingMetrics:
        return self._run(samples, train=True)

    def evaluate(self, samples: Sequence[ContextSample]) -> TrainingMetrics:
        return self._run(samples, train=False)

    def _run(self, samples: Sequence[ContextSample], train: bool) -> TrainingMetrics:
        if not samples:
            raise ValueError("No samples supplied for training/evaluation")
        total_loss = 0.0
        stage1_loss = 0.0
        total_weight = 0.0
        for sample in samples:
            stage1_input = self.builder.build_stage1_input(sample.signals)
            tail = self.builder.build_tail_embeddings(sample.signals)
            logits, embedding = self.model.step(
                stage1_input,
                tail_embeddings=tail,
                target_token=sample.target_token,
                stage1_target=sample.stage1_target,
                train=train,
            )
            ce = _cross_entropy(logits, sample.target_token)
            s1 = _mse_loss(embedding, sample.stage1_target)
            weight = sample.weight
            total_loss += ce * weight
            stage1_loss += s1 * weight
            total_weight += weight
            if train:
                self._steps += 1
        avg_loss = total_loss / total_weight
        ppl = float(math.exp(avg_loss))
        avg_stage1 = stage1_loss / total_weight
        if train:
            self.metric_log.append(self._steps, ppl)
        return TrainingMetrics(loss=float(avg_loss), perplexity=ppl, stage1_loss=float(avg_stage1), samples=int(total_weight))


def _cross_entropy(logits: np.ndarray, target: int) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    max_val = float(np.max(logits))
    log_sum = max_val + math.log(float(np.exp(logits - max_val).sum()))
    return float(-(logits[target] - log_sum))


def _mse_loss(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(diff**2))
