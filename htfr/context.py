"""Context building utilities for Hypertensor Field Transformer training."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Iterator, Optional, Protocol, Sequence

import numpy as np

from .feature_ops import hashed_ngram_features


@dataclass(frozen=True)
class ContextSignals:
    """Container holding teacher signals for a single prediction window."""

    token_ids: np.ndarray  # shape (window,)
    hidden_states: np.ndarray  # shape (window, hidden_dim)
    attention_mask: Optional[np.ndarray] = None  # shape (window,)
    rope_phases: Optional[np.ndarray] = None  # shape (window, 2)


@dataclass(frozen=True)
class ContextSample:
    """Fully prepared training example."""

    signals: ContextSignals
    target_token: int
    stage1_target: np.ndarray
    teacher_logits: Optional[np.ndarray] = None
    weight: float = 1.0


@dataclass(frozen=True)
class ContextBuilderConfig:
    """Configuration controlling context → feature projection."""

    window_size: int
    hidden_dim: int
    hashed_dim: int = 8192
    ngram: int = 2
    num_hashes: int = 2
    tail_tokens: int = 16
    stage1_target_dim: int = 1024


class ContextBuilder:
    """Constructs Stage-1 and Stage-2 inputs from teacher signals."""

    def __init__(self, config: ContextBuilderConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(0)
        self._target_matrix = self._rng.standard_normal(
            (config.hidden_dim, config.stage1_target_dim)
        ).astype(np.float32) / max(1.0, float(config.hidden_dim) ** 0.5)
        self._raw_dim = (
            config.window_size * config.hidden_dim  # flattened hidden states
            + config.hashed_dim
            + 2 * config.window_size  # positional sin/cos
        )
        self._tail_raw_dim = min(config.tail_tokens, config.window_size) * config.hidden_dim

    @property
    def raw_dim(self) -> int:
        """Number of float32 elements emitted for Stage-1 before projection."""

        return self._raw_dim

    @property
    def tail_raw_dim(self) -> int:
        """Flattened tail embedding dimension consumed by Stage-2."""

        return self._tail_raw_dim

    @property
    def stage1_target_dim(self) -> int:
        """Dimension of the compressed Stage-1 regression target."""

        return self.config.stage1_target_dim

    def build_stage1_input(self, signals: ContextSignals) -> np.ndarray:
        """Return a flattened feature vector covering the full context window."""

        hidden = np.asarray(signals.hidden_states, dtype=np.float32)
        if hidden.shape != (self.config.window_size, self.config.hidden_dim):
            raise ValueError(
                "Hidden-state shape mismatch: expected "
                f"{(self.config.window_size, self.config.hidden_dim)}, got {hidden.shape}"
            )
        pieces = [hidden.reshape(-1)]
        hashed = hashed_ngram_features(
            signals.token_ids.tolist(),
            dim=self.config.hashed_dim,
            ngram=self.config.ngram,
            num_hashes=self.config.num_hashes,
        )
        pieces.append(hashed.astype(np.float32))
        pieces.append(self._positional_features(signals.rope_phases))
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def build_tail_embeddings(self, signals: ContextSignals) -> np.ndarray:
        """Return the flattened representation of the final tail tokens."""

        tail_len = min(self.config.tail_tokens, self.config.window_size)
        tail = signals.hidden_states[-tail_len:]
        return tail.astype(np.float32).reshape(-1)

    def stage1_target_from_signals(self, signals: ContextSignals) -> np.ndarray:
        """Return the compressed target vector for Stage-1."""

        source = np.asarray(signals.hidden_states[-1], dtype=np.float32)
        return source @ self._target_matrix

    def _positional_features(self, rope: Optional[np.ndarray]) -> np.ndarray:
        if rope is not None:
            expected = (self.config.window_size, 2)
            if rope.shape != expected:
                raise ValueError(f"ROPE shape mismatch: expected {expected}, got {rope.shape}")
            return rope.astype(np.float32).reshape(-1)
        # Fallback: synthesize sinusoidal phases.
        positions = np.arange(self.config.window_size, dtype=np.float32)
        phase = positions / max(1, self.config.window_size - 1)
        sin = np.sin(np.pi * phase)
        cos = np.cos(np.pi * phase)
        return np.stack([sin, cos], axis=1).reshape(-1).astype(np.float32)


def build_context_samples(
    windows: Sequence["TeacherWindowLike"],
    builder: ContextBuilder,
    vocab_mapping: np.ndarray,
) -> list[ContextSample]:
    """Convert teacher windows into ContextSample objects."""

    samples: list[ContextSample] = []
    for window in windows:
        token_ids = np.asarray(window.tokens, dtype=np.int64)
        hidden = np.asarray(window.hidden_states, dtype=np.float32)
        signals = ContextSignals(
            token_ids=token_ids,
            hidden_states=hidden,
            attention_mask=None,
            rope_phases=getattr(window, "rope_phases", None),
        )
        target = int(window.target_token)
        compact = int(vocab_mapping[target])
        stage1_target = builder.stage1_target_from_signals(signals)
        samples.append(
            ContextSample(
                signals=signals,
                target_token=compact,
                stage1_target=stage1_target,
                teacher_logits=(
                    None if window.logits is None else np.asarray(window.logits, dtype=np.float32)
                ),
            )
        )
    return samples


class TeacherWindowLike(Protocol):
    """Protocol describing the data expected from teacher adapters."""

    tokens: np.ndarray
    hidden_states: np.ndarray
    target_token: int
    logits: Optional[np.ndarray]
    rope_phases: Optional[np.ndarray]


@dataclass(frozen=True)
class ContextBatch:
    """Materialized set of samples ready for projection."""

    stage1_inputs: np.ndarray
    stage1_targets: np.ndarray
    tail_embeddings: np.ndarray
    target_tokens: np.ndarray
    weights: np.ndarray
    sample_indices: np.ndarray
    transfer_bytes: int

    @property
    def sample_count(self) -> int:
        return int(self.stage1_inputs.shape[0])


class ContextBatchProducer:
    """Precomputes Stage-1 inputs/tails in worker threads for trainer batches."""

    def __init__(
        self,
        builder: ContextBuilder,
        samples: Sequence[ContextSample],
        batch_size: int,
        *,
        max_workers: Optional[int] = None,
        prefetch_batches: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.builder = builder
        self.samples = samples
        self.batch_size = batch_size
        cpu_count = max(1, _safe_cpu_count())
        if max_workers is None or max_workers <= 0:
            self.max_workers = min(4, cpu_count)
        else:
            self.max_workers = max(1, max_workers)
        if prefetch_batches is None or prefetch_batches <= 0:
            self.prefetch_batches = max(2, self.max_workers * 2)
        else:
            self.prefetch_batches = prefetch_batches

    def __iter__(self) -> Iterator[ContextBatch]:
        total = len(self.samples)
        if total == 0:
            return
        futures: deque[Future[ContextBatch]] = deque()
        cursor = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while cursor < total or futures:
                while cursor < total and len(futures) < self.prefetch_batches:
                    start = cursor
                    end = min(total, start + self.batch_size)
                    futures.append(executor.submit(self._build_batch, start, end))
                    cursor = end
                future = futures.popleft()
                yield future.result()

    def _build_batch(self, start: int, end: int) -> ContextBatch:
        size = end - start
        stage1_inputs = np.empty((size, self.builder.raw_dim), dtype=np.float32)
        stage1_targets = np.empty((size, self.builder.stage1_target_dim), dtype=np.float32)
        tail_dim = self.builder.tail_raw_dim
        tail_embeddings = (
            np.zeros((size, tail_dim), dtype=np.float32)
            if tail_dim == 0
            else np.empty((size, tail_dim), dtype=np.float32)
        )
        target_tokens = np.empty(size, dtype=np.int64)
        weights = np.empty(size, dtype=np.float32)
        indices = np.arange(start, end, dtype=np.int64)
        for offset, sample_idx in enumerate(range(start, end)):
            sample = self.samples[sample_idx]
            stage1_inputs[offset] = self.builder.build_stage1_input(sample.signals)
            stage1_targets[offset] = np.asarray(sample.stage1_target, dtype=np.float32)
            if tail_dim:
                tail_embeddings[offset] = self.builder.build_tail_embeddings(sample.signals)
            target_tokens[offset] = int(sample.target_token)
            weights[offset] = float(sample.weight)
        transfer_bytes = (
            stage1_inputs.nbytes + stage1_targets.nbytes + tail_embeddings.nbytes
        )
        return ContextBatch(
            stage1_inputs=stage1_inputs,
            stage1_targets=stage1_targets,
            tail_embeddings=tail_embeddings,
            target_tokens=target_tokens,
            weights=weights,
            sample_indices=indices,
            transfer_bytes=transfer_bytes,
        )


def _safe_cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:  # pragma: no cover - defensive path
        return 1
