"""Train the Hypertensor Field Transformer against Gemma 3 teacher data."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from htfr.checkpoint import HTFTCheckpoint, StageState, save_htft_checkpoint
from htfr.context import ContextBuilder, ContextBuilderConfig, build_context_samples
from htfr.data.gemma_adapter import (
    GemmaConfig,
    build_token_stream,
    build_vocab_mapping,
    collect_teacher_windows,
    ensure_authentication,
    load_dataset_split,
    load_teacher,
    truncated_teacher_perplexity,
)
from htfr.feature_ops import (
    CountSketchParameters,
    ProjectionStack,
    SRHTParameters,
    make_block_srht,
    make_count_sketch,
)
from htfr.hypertensor_field_transformer import HypertensorFieldTransformer, StageRuntime
from htfr.initialization import random_hypertensors
from htfr.model import HTFRModel
from htfr.trainer import HTFTTrainer
from htfr.profiler import PipelineProfiler

LOGGER_NAME = "train_htft"
logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    """Configure a stdout logger once per process."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[train_htft][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


@contextmanager
def log_section(name: str, profiler: PipelineProfiler | None = None):
    """Context manager that logs the start/end (with duration) of a section."""

    logger.info("Starting %s", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info("Finished %s in %.2f s", name, duration)
        if profiler is not None:
            profiler.record(f"section:{name}", duration, {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    parser.add_argument("--model", default="google/gemma-3-270m", help="Teacher model identifier")
    parser.add_argument("--dataset", default="wikitext", help="Dataset identifier")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="Dataset config name")
    parser.add_argument("--train-tokens", type=int, default=200_000, help="Tokens for training stream")
    parser.add_argument("--eval-tokens", type=int, default=50_000, help="Tokens for evaluation stream")
    parser.add_argument("--seq-len", type=int, default=128, help="Context window size")
    parser.add_argument("--stride", type=int, default=64, help="Stride between windows")
    parser.add_argument("--max-train-examples", type=int, default=2048, help="Max windows for training")
    parser.add_argument("--max-eval-examples", type=int, default=512, help="Max windows for evaluation")
    parser.add_argument("--vocab-limit", type=int, default=4096, help="Compact vocabulary size")
    parser.add_argument("--stage1-tensors", type=int, default=8000, help="Number of Stage-1 Hypertensors")
    parser.add_argument("--stage2-tensors", type=int, default=16000, help="Number of Stage-2 Hypertensors")
    parser.add_argument("--stage1-countsketch-dim", type=int, default=32768, help="Stage-1 CountSketch dim")
    parser.add_argument("--stage1-srht-dim", type=int, default=16_384, help="Stage-1 SRHT dim")
    parser.add_argument("--stage2-countsketch-dim", type=int, default=8192, help="Stage-2 CountSketch dim")
    parser.add_argument("--stage2-srht-dim", type=int, default=4096, help="Stage-2 SRHT dim")
    parser.add_argument("--stage1-target-dim", type=int, default=1024, help="Stage-1 embedding size")
    parser.add_argument("--tail-tokens", type=int, default=16, help="Tokens copied into Stage-2 tail")
    parser.add_argument("--hashed-dim", type=int, default=8192, help="Dimension of hashed indicators")
    parser.add_argument("--stage1-eta", type=float, default=0.04, help="Stage-1 learning rate")
    parser.add_argument("--stage1-eta-g", type=float, default=0.004, help="Stage-1 geometry LR")
    parser.add_argument("--stage2-eta", type=float, default=0.03, help="Stage-2 learning rate")
    parser.add_argument("--stage2-eta-g", type=float, default=0.003, help="Stage-2 geometry LR")
    parser.add_argument("--stage1-tau", type=float, default=1.0, help="Stage-1 tau")
    parser.add_argument("--stage2-tau", type=float, default=0.8, help="Stage-2 tau")
    parser.add_argument("--metrics-path", type=str, default=None, help="Optional JSONL metrics output")
    parser.add_argument("--output", type=str, default=None, help="Optional checkpoint path (.npz)")
    parser.add_argument("--profile-output", type=str, default=None, help="Optional JSONL profiling output")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Trainer batch size for context processing")
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=8,
        help="CPU workers for batch preparation (0=auto)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Enable Stage1/Stage2 overlap with a double-buffered pipeline",
    )
    parser.add_argument(
        "--stage1-device",
        type=str,
        default="hip:0",
        help="Logical device assignment for Stage 1 (metadata only for now)",
    )
    parser.add_argument(
        "--stage2-device",
        type=str,
        default="hip:0",
        help="Logical device assignment for Stage 2 (metadata/logging)",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=10.0,
        help="Seconds between training progress logs",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    profiler = PipelineProfiler(args.profile_output)
    logger.info(
        "HTFT training start | model=%s dataset=%s/%s seq_len=%d stride=%d seed=%d",
        args.model,
        args.dataset,
        args.dataset_config,
        args.seq_len,
        args.stride,
        args.seed,
    )

    with log_section("Authenticating with Hugging Face Hub", profiler):
        ensure_authentication(args.hf_token)

    with log_section(f"Loading teacher model and tokenizer ({args.model})", profiler):
        tokenizer, model, device = load_teacher(args.model, args.hf_token)

    with log_section(f"Loading dataset splits ({args.dataset}/{args.dataset_config})", profiler):
        dataset = load_dataset_split(args.dataset, args.dataset_config)

    with log_section("Building train token stream", profiler):
        train_tokens = build_token_stream(
            dataset["train"]["text"], tokenizer, args.train_tokens + args.seq_len + 1
        )
    with log_section("Building eval token stream", profiler):
        eval_tokens = build_token_stream(
            dataset["validation"]["text"], tokenizer, args.eval_tokens + args.seq_len + 1
        )

    gemma_train_cfg = GemmaConfig(
        model_id=args.model,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_train_examples,
        collect_logits=False,
        device=str(device),
    )
    gemma_eval_cfg = GemmaConfig(
        model_id=args.model,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_eval_examples,
        collect_logits=True,
        device=str(device),
    )
    with log_section("Collecting teacher windows (train)", profiler):
        train_windows = collect_teacher_windows(model, train_tokens, gemma_train_cfg, phase="train")
    with log_section("Collecting teacher windows (eval)", profiler):
        eval_windows = collect_teacher_windows(model, eval_tokens, gemma_eval_cfg, phase="eval")

    mapping, shortlist = build_vocab_mapping(
        targets=train_tokens.numpy(), vocab_size=tokenizer.vocab_size, vocab_limit=args.vocab_limit
    )
    unk_index = shortlist.size
    logger.info(
        "Vocabulary mapping ready | shortlist=%d unk_index=%d",
        shortlist.size,
        unk_index,
    )

    builder = ContextBuilder(
        ContextBuilderConfig(
            window_size=args.seq_len,
            hidden_dim=model.config.hidden_size,
            hashed_dim=args.hashed_dim,
            tail_tokens=args.tail_tokens,
            stage1_target_dim=args.stage1_target_dim,
        )
    )

    rng = np.random.default_rng(args.seed)
    stage1_state = _build_stage_state(
        builder=builder,
        srht_dim=args.stage1_srht_dim,
        countsketch_dim=args.stage1_countsketch_dim,
        tensor_count=args.stage1_tensors,
        output_dim=builder.stage1_target_dim,
        tau=args.stage1_tau,
        eta=args.stage1_eta,
        eta_g=args.stage1_eta_g,
        rng=rng,
    )
    stage2_state = _build_stage_state(
        builder=None,
        srht_dim=args.stage2_srht_dim,
        countsketch_dim=args.stage2_countsketch_dim,
        tensor_count=args.stage2_tensors,
        output_dim=args.vocab_limit + 1,
        tau=args.stage2_tau,
        eta=args.stage2_eta,
        eta_g=args.stage2_eta_g,
        rng=rng,
        raw_dim=builder.stage1_target_dim + builder.tail_raw_dim,
    )
    logger.info(
        "Stage1 config | tensors=%d target_dim=%d countsketch_dim=%d srht_dim=%d",
        args.stage1_tensors,
        builder.stage1_target_dim,
        args.stage1_countsketch_dim,
        args.stage1_srht_dim,
    )
    logger.info(
        "Stage2 config | tensors=%d target_dim=%d countsketch_dim=%d srht_dim=%d",
        args.stage2_tensors,
        args.vocab_limit + 1,
        args.stage2_countsketch_dim,
        args.stage2_srht_dim,
    )

    stage1_runtime = StageRuntime(
        projector=stage1_state.projector,
        model=stage1_state.model,
        loss="mse",
    )
    stage2_runtime = StageRuntime(
        projector=stage2_state.projector,
        model=stage2_state.model,
        loss="logits_ce",
    )
    hypertensor_transformer = HypertensorFieldTransformer(
        stage1_runtime,
        stage2_runtime,
        tail_token_count=args.tail_tokens,
        tail_embedding_dim=model.config.hidden_size,
    )

    logger.info(
        "Context builder configured | window=%d hidden_dim=%d hashed_dim=%d tail_tokens=%d",
        args.seq_len,
        model.config.hidden_size,
        args.hashed_dim,
        args.tail_tokens,
    )

    train_samples = build_context_samples(train_windows, builder, mapping)
    eval_samples = build_context_samples(eval_windows, builder, mapping)
    logger.info(
        "Prepared context samples | train=%d eval=%d",
        len(train_samples),
        len(eval_samples),
    )

    log_interval_seconds = args.log_interval_seconds
    if log_interval_seconds <= 0:
        logger.warning(
            "log_interval_seconds must be positive; overriding %.3f -> 10.0",
            log_interval_seconds,
        )
        log_interval_seconds = 10.0
    batch_workers = args.batch_workers if args.batch_workers > 0 else None
    trainer = HTFTTrainer(
        hypertensor_transformer,
        builder,
        log_interval_seconds=log_interval_seconds,
        batch_size=max(1, args.batch_size),
        batch_workers=batch_workers,
        pipeline_enabled=args.pipeline,
        stage_devices={
            "stage1": args.stage1_device or "cpu",
            "stage2": args.stage2_device or "cpu",
        },
        profiler=profiler,
    )
    logger.info(
        "Batch config | batch_size=%d workers=%s pipeline=%s stage_devices=(%s, %s)",
        max(1, args.batch_size),
        batch_workers if batch_workers is not None else "auto",
        args.pipeline,
        args.stage1_device or "cpu",
        args.stage2_device or "cpu",
    )
    with log_section("Training hypertensor transformer", profiler):
        train_metrics = trainer.train_epoch(train_samples)
    pruned_stage1 = stage1_state.model.prune_unmodified()
    pruned_stage2 = stage2_state.model.prune_unmodified()
    if pruned_stage1 or pruned_stage2:
        logger.info(
            "Pruned inactive hypertensors | stage1=%d stage2=%d",
            pruned_stage1,
            pruned_stage2,
        )
        profiler.record(
            "prune_inactive_tensors",
            0.0,
            {"stage1_removed": pruned_stage1, "stage2_removed": pruned_stage2},
        )
    with log_section("Evaluating hypertensor transformer", profiler):
        eval_metrics = trainer.evaluate(eval_samples)

        teacher_logits = np.stack([window.logits for window in eval_windows if window.logits is not None])
    teacher_targets = np.array([window.target_token for window in eval_windows], dtype=np.int64)
    teacher_ppl = truncated_teacher_perplexity(teacher_logits, teacher_targets, shortlist, mapping)
    logger.info("Teacher perplexity (truncated vocab): %.3f", teacher_ppl)

    metrics_payload = {
        "train_loss": train_metrics.loss,
        "train_perplexity": train_metrics.perplexity,
        "eval_loss": eval_metrics.loss,
        "eval_perplexity": eval_metrics.perplexity,
        "teacher_perplexity": teacher_ppl,
    }
    logger.info("Training metrics:\n%s", json.dumps(metrics_payload, indent=2))
    profiler.record("metrics", 0.0, metrics_payload)
    if args.metrics_path:
        pathlib.Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": trainer.metric_log.steps[-1] if trainer.metric_log.steps else 0, **metrics_payload}) + "\n")
        logger.info("Metrics appended to %s", args.metrics_path)

    if args.output:
        logger.info("Preparing checkpoint for %s", args.output)
        checkpoint = HTFTCheckpoint(
            stage1=StageState(
                model=stage1_state.model,
                srht=tuple(stage1_state.srht_params),
                countsketch=stage1_state.countsketch,
                metadata={
                    "stage": 1,
                    "device": args.stage1_device or "cpu",
                    "pipeline": "producer" if args.pipeline else "sequential",
                    "batch_size": max(1, args.batch_size),
                },
            ),
            stage2=StageState(
                model=stage2_state.model,
                srht=tuple(stage2_state.srht_params),
                countsketch=stage2_state.countsketch,
                metadata={
                    "stage": 2,
                    "device": args.stage2_device or "cpu",
                    "pipeline": "consumer" if args.pipeline else "sequential",
                    "batch_size": max(1, args.batch_size),
                },
            ),
            mapping=mapping,
            shortlist=shortlist,
            unk_index=unk_index,
            tail_config={
                "tail_tokens": args.tail_tokens,
                "hidden_dim": model.config.hidden_size,
            },
            metadata={
                "train_metrics": metrics_payload,
                "config": vars(args),
            },
        )
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_htft_checkpoint(args.output, checkpoint)
        logger.info("Checkpoint saved to %s", args.output)
    profiler.flush()


@dataclass
class _StageBuildResult:
    model: HTFRModel
    projector: ProjectionStack
    srht_params: Tuple[SRHTParameters, ...]
    countsketch: Optional[CountSketchParameters]
    projector_raw_dim: int


def _build_stage_state(
    builder: Optional[ContextBuilder],
    srht_dim: int,
    countsketch_dim: int,
    tensor_count: int,
    output_dim: int,
    tau: float,
    eta: float,
    eta_g: float,
    rng: np.random.Generator,
    raw_dim: Optional[int] = None,
) -> _StageBuildResult:
    if raw_dim is None:
        if builder is None:
            raise ValueError("Builder must be provided when raw_dim is not explicit")
        raw_dim = builder.raw_dim
    countsketch = make_count_sketch(raw_dim, countsketch_dim, rng=rng)
    srht = make_block_srht(input_dim=countsketch_dim, target_dim=srht_dim, rng=rng)
    projector = ProjectionStack(
        raw_dim=raw_dim,
        srht_params=(srht,),
        countsketch=countsketch,
        output_dtype=np.float16,
    )
    tensors = random_hypertensors(
        tensor_count,
        input_dim=projector.output_dim,
        output_dim=output_dim,
        tau=tau,
        rng=rng,
    )
    model = HTFRModel(
        tensors=tensors,
        top_k=32,
        train_top_k=128,
        eta=eta,
        eta_g=eta_g,
        weight_mode="softmax",
    )
    return _StageBuildResult(
        model=model,
        projector=projector,
        srht_params=(srht,),
        countsketch=countsketch,
        projector_raw_dim=raw_dim,
    )


if __name__ == "__main__":
    main()
