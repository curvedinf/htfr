"""Training helpers for the Hypertensor Field Transformer."""
from __future__ import annotations

import logging
import math
import time
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

    def __init__(
        self,
        model: HypertensorFieldTransformer,
        builder: ContextBuilder,
        *,
        log_interval_seconds: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.builder = builder
        self.metric_log = MetricLog()
        self._steps = 0
        if log_interval_seconds <= 0:
            raise ValueError("log_interval_seconds must be positive")
        self._log_interval_seconds = float(log_interval_seconds)
        self._logger = logger or logging.getLogger("train_htft")

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
        processed = 0
        start_time = time.perf_counter()
        total_samples = len(samples)
        next_log_time = start_time + self._log_interval_seconds
        last_logged = 0
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
            processed += 1
            if train:
                self._steps += 1
                now = time.perf_counter()
                if now >= next_log_time:
                    self._log_progress(
                        processed=processed,
                        total_samples=total_samples,
                        total_loss=total_loss,
                        stage1_loss=stage1_loss,
                        total_weight=total_weight,
                        start_time=start_time,
                    )
                    next_log_time = now + self._log_interval_seconds
                    last_logged = processed
        avg_loss = total_loss / total_weight
        ppl = float(math.exp(avg_loss))
        avg_stage1 = stage1_loss / total_weight
        if train:
            self.metric_log.append(self._steps, ppl)
            if last_logged != processed:
                self._log_progress(
                    processed=processed,
                    total_samples=total_samples,
                    total_loss=total_loss,
                    stage1_loss=stage1_loss,
                    total_weight=total_weight,
                    start_time=start_time,
                )
        return TrainingMetrics(loss=float(avg_loss), perplexity=ppl, stage1_loss=float(avg_stage1), samples=int(total_weight))

    def _log_progress(
        self,
        *,
        processed: int,
        total_samples: int,
        total_loss: float,
        stage1_loss: float,
        total_weight: float,
        start_time: float,
    ) -> None:
        elapsed = time.perf_counter() - start_time
        avg_loss = total_loss / total_weight
        stage1_avg = stage1_loss / total_weight
        ppl = float(math.exp(avg_loss))
        rate = total_weight / elapsed if elapsed > 0 else float("inf")
        self._logger.info(
            "Training progress | samples=%d/%d loss=%.4f stage1_loss=%.4f ppl=%.2f weight=%.1f rate=%.1f samples/s",
            processed,
            total_samples,
            avg_loss,
            stage1_avg,
            ppl,
            total_weight,
            rate,
        )


def _cross_entropy(logits: np.ndarray, target: int) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    max_val = float(np.max(logits))
    log_sum = max_val + math.log(float(np.exp(logits - max_val).sum()))
    return float(-(logits[target] - log_sum))


def _mse_loss(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(diff**2))
