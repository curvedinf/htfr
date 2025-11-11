"""Context building utilities for HyperField Transformer training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

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
