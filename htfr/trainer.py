"""Training helpers for the Hypertensor Field Transformer."""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .context import ContextBatch, ContextBatchProducer, ContextBuilder, ContextSample
from .hypertensor_field_transformer import HypertensorFieldTransformer
from .model import HTFRModel, ModelStepMetrics
from .profiler import PipelineProfiler


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


@dataclass(frozen=True)
class UsageStats:
    """Snapshot of Hypertensor usage counters for logging."""

    minimum: float
    mean: float
    maximum: float


@dataclass(frozen=True)
class BatchInstrumentation:
    """Aggregated instrumentation for a processed batch."""

    stage1_active_avg: float
    stage1_active_max: int
    stage1_max_distance: float
    stage2_active_avg: float
    stage2_active_max: int
    stage2_max_distance: float
    stage1_usage: UsageStats
    stage2_usage: UsageStats


@dataclass(frozen=True)
class BatchComputation:
    """Summaries emitted by :meth:`HTFTTrainer._process_batch`."""

    loss_sum: float
    stage1_loss_sum: float
    total_weight: float
    sample_count: int
    loss_mean: float
    loss_std: float
    stage1_loss_mean: float
    stage1_loss_std: float
    stage1_time: float
    stage2_time: float
    transfer_bytes: int
    instrumentation: BatchInstrumentation

    def profile_metadata(self) -> dict[str, float | int]:
        return {
            "samples": self.sample_count,
            "loss_mean": self.loss_mean,
            "loss_std": self.loss_std,
            "stage1_loss_mean": self.stage1_loss_mean,
            "stage1_loss_std": self.stage1_loss_std,
            "stage1_time": self.stage1_time,
            "stage2_time": self.stage2_time,
            "transfer_bytes": self.transfer_bytes,
            "stage1_active_max": self.instrumentation.stage1_active_max,
            "stage2_active_max": self.instrumentation.stage2_active_max,
        }


@dataclass(frozen=True)
class _Stage1Result:
    """Return type for Stage-1 computations (sequential or pipelined)."""

    sample_idx: int
    embedding: np.ndarray
    metrics: ModelStepMetrics | None
    elapsed: float


class _RunningStats:
    """Online mean/std calculator (unweighted)."""

    __slots__ = ("count", "mean", "_m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float(math.sqrt(self._m2 / (self.count - 1)))


class _Stage1PipelineRunner:
    """Double-buffered executor that overlaps Stage-1 work with Stage-2."""

    def __init__(self, transformer: HypertensorFieldTransformer) -> None:
        self._transformer = transformer
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future[_Stage1Result] | None = None

    def prime(self, batch: ContextBatch, sample_idx: int, train: bool) -> None:
        self._pending = self._executor.submit(self._compute, batch, sample_idx, train)

    def next(self, batch: ContextBatch, sample_idx: int, train: bool) -> _Stage1Result:
        future = self._executor.submit(self._compute, batch, sample_idx, train)
        if self._pending is None:
            raise RuntimeError("Stage1 pipeline used before being primed")
        result = self._pending.result()
        self._pending = future
        return result

    def finalize(self) -> Optional[_Stage1Result]:
        if self._pending is None:
            self._executor.shutdown(wait=True)
            return None
        result = self._pending.result()
        self._pending = None
        self._executor.shutdown(wait=True)
        return result

    def _compute(self, batch: ContextBatch, sample_idx: int, train: bool) -> _Stage1Result:
        vector = batch.stage1_inputs[sample_idx]
        target = batch.stage1_targets[sample_idx]
        start = time.perf_counter()
        embedding, metrics = self._transformer.run_stage1(
            vector,
            target=target,
            train=train,
        )
        elapsed = time.perf_counter() - start
        return _Stage1Result(
            sample_idx=sample_idx,
            embedding=embedding,
            metrics=metrics,
            elapsed=elapsed,
        )


class HTFTTrainer:
    """High-level trainer coordinating both stages."""

    def __init__(
        self,
        model: HypertensorFieldTransformer,
        builder: ContextBuilder,
        *,
        log_interval_seconds: float = 10.0,
        logger: logging.Logger | None = None,
        batch_size: int = 1,
        batch_workers: Optional[int] = None,
        pipeline_enabled: bool = False,
        stage_devices: Optional[Dict[str, str]] = None,
        profiler: PipelineProfiler | None = None,
    ) -> None:
        self.model = model
        self.builder = builder
        self.metric_log = MetricLog()
        self._steps = 0
        if log_interval_seconds <= 0:
            raise ValueError("log_interval_seconds must be positive")
        self._log_interval_seconds = float(log_interval_seconds)
        self._logger = logger or logging.getLogger("train_htft")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = int(batch_size)
        self._batch_workers = None if batch_workers is None else max(1, int(batch_workers))
        self._pipeline_enabled = bool(pipeline_enabled)
        self._stage_devices = stage_devices or {"stage1": "cpu", "stage2": "cpu"}
        self._batch_counter = 0
        self._profiler = profiler

    def train_epoch(self, samples: Sequence[ContextSample]) -> TrainingMetrics:
        return self._run(samples, train=True)

    def evaluate(self, samples: Sequence[ContextSample]) -> TrainingMetrics:
        return self._run(samples, train=False)

    def _run(self, samples: Sequence[ContextSample], train: bool) -> TrainingMetrics:
        run_start = time.perf_counter()
        if not samples:
            raise ValueError("No samples supplied for training/evaluation")
        total_loss = 0.0
        stage1_loss = 0.0
        total_weight = 0.0
        processed = 0
        stage1_time_total = 0.0
        stage2_time_total = 0.0
        transfer_total = 0
        start_time = time.perf_counter()
        total_samples = len(samples)
        next_log_time = start_time + self._log_interval_seconds
        last_logged = 0
        producer = ContextBatchProducer(
            self.builder,
            samples,
            batch_size=self._batch_size,
            max_workers=self._batch_workers,
        )
        any_batches = False
        for batch in producer:
            any_batches = True
            batch_stats = self._process_batch(batch, train=train)
            total_loss += batch_stats.loss_sum
            stage1_loss += batch_stats.stage1_loss_sum
            total_weight += batch_stats.total_weight
            processed += batch_stats.sample_count
            stage1_time_total += batch_stats.stage1_time
            stage2_time_total += batch_stats.stage2_time
            transfer_total += batch_stats.transfer_bytes
            self._batch_counter += 1
            if train:
                self._steps += batch_stats.sample_count
                self._log_batch_stats(batch_stats, level=logging.INFO)
                now = time.perf_counter()
                if now >= next_log_time:
                    self._log_progress(
                        processed=processed,
                        total_samples=total_samples,
                        total_loss=total_loss,
                        stage1_loss=stage1_loss,
                        total_weight=total_weight,
                        start_time=start_time,
                        stage1_time=stage1_time_total,
                        stage2_time=stage2_time_total,
                        transfer_bytes=transfer_total,
                    )
                    next_log_time = now + self._log_interval_seconds
                    last_logged = processed
            else:
                self._log_batch_stats(batch_stats, level=logging.DEBUG)
        if not any_batches:
            raise RuntimeError("Failed to materialize training batches")
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
                    stage1_time=stage1_time_total,
                    stage2_time=stage2_time_total,
                    transfer_bytes=transfer_total,
                )
        total_duration = time.perf_counter() - run_start
        self._record_profile_event(
            "train_epoch" if train else "eval_epoch",
            total_duration,
            {
                "samples": processed,
                "avg_loss": float(avg_loss),
                "avg_stage1_loss": float(avg_stage1),
                "stage1_time": stage1_time_total,
                "stage2_time": stage2_time_total,
            },
        )
        return TrainingMetrics(
            loss=float(avg_loss),
            perplexity=ppl,
            stage1_loss=float(avg_stage1),
            samples=int(total_weight),
        )

    def _stage1_iterator(self, batch: ContextBatch, train: bool) -> Iterable[_Stage1Result]:
        """Yield Stage-1 results, optionally overlapped with Stage-2 work."""

        if not self._pipeline_enabled or batch.sample_count <= 1:
            for idx in range(batch.sample_count):
                yield self._compute_stage1(batch, idx, train)
            return
        pipeline = _Stage1PipelineRunner(self.model)
        pipeline.prime(batch, 0, train)
        for idx in range(1, batch.sample_count):
            yield pipeline.next(batch, idx, train)
        final = pipeline.finalize()
        if final is not None:
            yield final

    def _compute_stage1(self, batch: ContextBatch, sample_idx: int, train: bool) -> _Stage1Result:
        vector = batch.stage1_inputs[sample_idx]
        target = batch.stage1_targets[sample_idx]
        start = time.perf_counter()
        embedding, metrics = self.model.run_stage1(
            vector,
            target=target,
            train=train,
        )
        elapsed = time.perf_counter() - start
        return _Stage1Result(
            sample_idx=sample_idx,
            embedding=embedding,
            metrics=metrics,
            elapsed=elapsed,
        )

    def _process_batch(self, batch: ContextBatch, train: bool) -> BatchComputation:
        batch_start = time.perf_counter()
        ce_stats = _RunningStats()
        stage1_stats = _RunningStats()
        loss_sum = 0.0
        stage1_loss_sum = 0.0
        total_weight = 0.0
        stage1_time = 0.0
        stage2_time = 0.0
        stage1_active_sum = 0.0
        stage2_active_sum = 0.0
        stage1_active_max = 0
        stage2_active_max = 0
        stage1_max_distance = 0.0
        stage2_max_distance = 0.0
        stage1_metrics_count = 0
        stage2_metrics_count = 0
        tail_dim = batch.tail_embeddings.shape[1]
        for result in self._stage1_iterator(batch, train):
            idx = result.sample_idx
            weight = float(batch.weights[idx])
            target_token = int(batch.target_tokens[idx])
            stage1_time += result.elapsed
            stage1_target = batch.stage1_targets[idx]
            s1_loss = _mse_loss(result.embedding, stage1_target)
            stage1_stats.update(s1_loss)
            stage1_loss_sum += s1_loss * weight
            tail_input = None
            if tail_dim > 0:
                tail_input = batch.tail_embeddings[idx]
            stage2_start = time.perf_counter()
            logits, stage2_metrics = self.model.run_stage2(
                result.embedding,
                tail_input,
                target_token=target_token,
                train=train,
            )
            stage2_time += time.perf_counter() - stage2_start
            ce = _cross_entropy(logits, target_token)
            ce_stats.update(ce)
            loss_sum += ce * weight
            total_weight += weight
            if result.metrics is not None:
                stage1_active_sum += result.metrics.active_count
                stage1_active_max = max(stage1_active_max, result.metrics.active_count)
                stage1_max_distance = max(stage1_max_distance, result.metrics.max_abs_distance)
                stage1_metrics_count += 1
            if stage2_metrics is not None:
                stage2_active_sum += stage2_metrics.active_count
                stage2_active_max = max(stage2_active_max, stage2_metrics.active_count)
                stage2_max_distance = max(stage2_max_distance, stage2_metrics.max_abs_distance)
                stage2_metrics_count += 1
        stage1_active_avg = (
            stage1_active_sum / max(1, stage1_metrics_count) if stage1_metrics_count else 0.0
        )
        stage2_active_avg = (
            stage2_active_sum / max(1, stage2_metrics_count) if stage2_metrics_count else 0.0
        )
        instrumentation = BatchInstrumentation(
            stage1_active_avg=stage1_active_avg,
            stage1_active_max=stage1_active_max,
            stage1_max_distance=stage1_max_distance,
            stage2_active_avg=stage2_active_avg,
            stage2_active_max=stage2_active_max,
            stage2_max_distance=stage2_max_distance,
            stage1_usage=self._usage_stats(self.model.stage1.model),
            stage2_usage=self._usage_stats(self.model.stage2.model),
        )
        result = BatchComputation(
            loss_sum=float(loss_sum),
            stage1_loss_sum=float(stage1_loss_sum),
            total_weight=float(total_weight),
            sample_count=batch.sample_count,
            loss_mean=float(ce_stats.mean),
            loss_std=float(ce_stats.std()),
            stage1_loss_mean=float(stage1_stats.mean),
            stage1_loss_std=float(stage1_stats.std()),
            stage1_time=float(stage1_time),
            stage2_time=float(stage2_time),
            transfer_bytes=batch.transfer_bytes,
            instrumentation=instrumentation,
        )
        duration = time.perf_counter() - batch_start
        self._record_profile_event(
            ("train_batch" if train else "eval_batch"),
            duration,
            result.profile_metadata(),
        )
        return result

    def _usage_stats(self, model: HTFRModel) -> UsageStats:
        usage = np.asarray(model._usage_counts, dtype=np.float32)
        if usage.size == 0:
            return UsageStats(0.0, 0.0, 0.0)
        return UsageStats(
            minimum=float(np.min(usage)),
            mean=float(np.mean(usage)),
            maximum=float(np.max(usage)),
        )

    def _log_batch_stats(self, batch: BatchComputation, level: int = logging.INFO) -> None:
        inst = batch.instrumentation
        transfer_mb = batch.transfer_bytes / (1024**2)
        self._logger.log(
            level,
            (
                "Batch %d | samples=%d loss=%.4f±%.4f stage1_loss=%.4f±%.4f "
                "stage1_active=%.1f/%d stage2_active=%.1f/%d "
                "stage1_dist=%.3f stage2_dist=%.3f h2d=%.2f MiB "
                "usage1[min=%.2f mean=%.2f max=%.2f] "
                "usage2[min=%.2f mean=%.2f max=%.2f]"
            ),
            self._batch_counter,
            batch.sample_count,
            batch.loss_mean,
            batch.loss_std,
            batch.stage1_loss_mean,
            batch.stage1_loss_std,
            inst.stage1_active_avg,
            inst.stage1_active_max,
            inst.stage2_active_avg,
            inst.stage2_active_max,
            inst.stage1_max_distance,
            inst.stage2_max_distance,
            transfer_mb,
            inst.stage1_usage.minimum,
            inst.stage1_usage.mean,
            inst.stage1_usage.maximum,
            inst.stage2_usage.minimum,
            inst.stage2_usage.mean,
            inst.stage2_usage.maximum,
        )

    def _record_profile_event(self, name: str, duration: float, metadata: dict[str, float | int | str]) -> None:
        if self._profiler is None:
            return
        self._profiler.record(name, duration, metadata)

    def _log_progress(
        self,
        *,
        processed: int,
        total_samples: int,
        total_loss: float,
        stage1_loss: float,
        total_weight: float,
        start_time: float,
        stage1_time: float,
        stage2_time: float,
        transfer_bytes: int,
    ) -> None:
        elapsed = time.perf_counter() - start_time
        avg_loss = total_loss / total_weight
        stage1_avg = stage1_loss / total_weight
        ppl = float(math.exp(avg_loss))
        rate = total_weight / elapsed if elapsed > 0 else float("inf")
        stage1_rate = processed / stage1_time if stage1_time > 0 else float("inf")
        stage2_rate = processed / stage2_time if stage2_time > 0 else float("inf")
        transfer_mb = transfer_bytes / (1024**2)
        self._logger.info(
            (
                "Training progress | samples=%d/%d loss=%.4f stage1_loss=%.4f ppl=%.2f "
                "weight=%.1f rate=%.1f samples/s stage1_rate=%.1f stage2_rate=%.1f h2d=%.2f MiB"
            ),
            processed,
            total_samples,
            avg_loss,
            stage1_avg,
            ppl,
            total_weight,
            rate,
            stage1_rate,
            stage2_rate,
            transfer_mb,
        )


def _cross_entropy(logits: np.ndarray, target: int) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    max_val = float(np.max(logits))
    log_sum = max_val + math.log(float(np.exp(logits - max_val).sum()))
    return float(-(logits[target] - log_sum))


def _mse_loss(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(diff**2))
