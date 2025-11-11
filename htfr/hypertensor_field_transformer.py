"""Two-stage Hypertensor Field Transformer built on HTFR primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .feature_ops import ProjectionStack
from .model import HTFRModel, LossMode, ModelStepMetrics


@dataclass(frozen=True)
class StageRuntime:
    """Container describing a single HTFR stage."""

    projector: ProjectionStack
    model: HTFRModel
    loss: LossMode


@dataclass(frozen=True)
class StepResult:
    """Return value for :meth:`HypertensorFieldTransformer.step`."""

    logits: np.ndarray
    embedding: np.ndarray
    stage1_metrics: ModelStepMetrics | None
    stage2_metrics: ModelStepMetrics | None


class HypertensorFieldTransformer:
    """Implements the Stage-1/Stage-2 coupling described in the design notes."""

    def __init__(
        self,
        stage1: StageRuntime,
        stage2: StageRuntime,
        tail_token_count: int = 16,
        tail_embedding_dim: int = 0,
    ) -> None:
        if tail_token_count < 0 or tail_embedding_dim < 0:
            raise ValueError("Tail configuration must be non-negative")
        self.stage1 = stage1
        self.stage2 = stage2
        self.tail_token_count = tail_token_count
        self.tail_embedding_dim = tail_embedding_dim
        self.tail_raw_dim = tail_token_count * tail_embedding_dim
        self.embedding_dim = stage1.model.output_dim
        expected_stage2_dim = self.embedding_dim + self.tail_raw_dim
        if stage2.projector.raw_dim != expected_stage2_dim:
            raise ValueError(
                "Stage 2 projector raw_dim must equal embedding + tail dim "
                f"({expected_stage2_dim}), got {stage2.projector.raw_dim}"
            )

    def step(
        self,
        context_vector: np.ndarray,
        tail_embeddings: Optional[np.ndarray],
        target_token: Optional[int],
        *,
        stage1_target: Optional[np.ndarray] = None,
        stage1_extra: Optional[np.ndarray] = None,
        train: bool = False,
    ) -> StepResult:
        """Run one autoregressive step through both stages."""

        embedding, stage1_metrics = self.run_stage1(
            context_vector,
            extra=stage1_extra,
            target=stage1_target,
            train=train and stage1_target is not None,
        )
        logits, stage2_metrics = self.run_stage2(
            embedding,
            tail_embeddings,
            target_token,
            train=train,
        )
        return StepResult(
            logits=logits,
            embedding=embedding,
            stage1_metrics=stage1_metrics,
            stage2_metrics=stage2_metrics,
        )

    def run_stage1(
        self,
        context_vector: np.ndarray,
        *,
        extra: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        train: bool = False,
    ) -> tuple[np.ndarray, ModelStepMetrics | None]:
        """Execute Stage 1 independently for parallel/pipelined use."""

        embedding = self._run_stage1(
            context_vector,
            extra=extra,
            target=target,
            train=train and target is not None,
        )
        return embedding, self.stage1.model.last_step_metrics()

    def run_stage2(
        self,
        embedding: np.ndarray,
        tail_embeddings: Optional[np.ndarray],
        target_token: Optional[int],
        *,
        train: bool = False,
    ) -> tuple[np.ndarray, ModelStepMetrics | None]:
        """Execute Stage 2 given a prepared Stage 1 embedding."""

        tail_vector = self._prepare_tail(tail_embeddings)
        stage2_raw = np.concatenate([embedding.astype(np.float32), tail_vector], axis=0)
        logits = self._run_stage2(
            stage2_raw,
            target_token,
            train=train and target_token is not None,
        )
        return logits, self.stage2.model.last_step_metrics()

    def _run_stage1(
        self,
        context_vector: np.ndarray,
        extra: Optional[np.ndarray],
        target: Optional[np.ndarray],
        train: bool,
    ) -> np.ndarray:
        projected = self.stage1.projector.project(context_vector, extra=extra)
        if train and target is not None:
            return self.stage1.model.predict_and_update(
                projected,
                target,
                loss=self.stage1.loss,
                train=True,
            ).astype(np.float32)
        return self.stage1.model.predict(projected).astype(np.float32)

    def _run_stage2(
        self,
        raw_features: np.ndarray,
        target: Optional[int],
        train: bool,
    ) -> np.ndarray:
        projected = self.stage2.projector.project(raw_features)
        if train and target is not None:
            return self.stage2.model.predict_and_update(
                projected,
                int(target),
                loss=self.stage2.loss,
                train=True,
            )
        return self.stage2.model.predict(projected)

    def _prepare_tail(self, tail_embeddings: Optional[np.ndarray]) -> np.ndarray:
        if self.tail_raw_dim == 0:
            return np.zeros(0, dtype=np.float32)
        if tail_embeddings is None:
            return np.zeros(self.tail_raw_dim, dtype=np.float32)
        tail = np.asarray(tail_embeddings, dtype=np.float32)
        if tail.ndim == 1:
            if self.tail_embedding_dim == 0:
                raise ValueError("tail_embedding_dim must be >0 when tail embeddings are 1-D")
            token_count = tail.size // self.tail_embedding_dim
            tail = tail.reshape(token_count, self.tail_embedding_dim)
        if tail.shape[1] != self.tail_embedding_dim:
            raise ValueError(
                f"Tail embedding dim mismatch ({tail.shape[1]} vs {self.tail_embedding_dim})"
            )
        if tail.shape[0] >= self.tail_token_count:
            trimmed = tail[-self.tail_token_count :]
        else:
            pad = np.zeros(
                (self.tail_token_count - tail.shape[0], self.tail_embedding_dim), dtype=np.float32
            )
            trimmed = np.vstack([pad, tail])
        return trimmed.reshape(-1)

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Return usage/loss diagnostics for verification tests."""

        return {
            "stage1_usage": self.stage1.model._usage_counts.copy(),
            "stage1_loss": self.stage1.model._loss_trace.copy(),
             "stage1_updates": self.stage1.model._update_counts.copy(),
            "stage2_usage": self.stage2.model._usage_counts.copy(),
            "stage2_loss": self.stage2.model._loss_trace.copy(),
             "stage2_updates": self.stage2.model._update_counts.copy(),
        }
