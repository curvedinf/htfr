"""Train the Hypertensor Field Transformer against Gemma 3 teacher data."""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

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
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_authentication(args.hf_token)
    tokenizer, model, device = load_teacher(args.model, args.hf_token)
    dataset = load_dataset_split(args.dataset, args.dataset_config)
    train_tokens = build_token_stream(
        dataset["train"]["text"], tokenizer, args.train_tokens + args.seq_len + 1
    )
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
    train_windows = collect_teacher_windows(model, train_tokens, gemma_train_cfg)
    eval_windows = collect_teacher_windows(model, eval_tokens, gemma_eval_cfg)

    mapping, shortlist = build_vocab_mapping(
        targets=train_tokens.numpy(), vocab_size=tokenizer.vocab_size, vocab_limit=args.vocab_limit
    )
    unk_index = shortlist.size

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

    train_samples = build_context_samples(train_windows, builder, mapping)
    eval_samples = build_context_samples(eval_windows, builder, mapping)

    trainer = HTFTTrainer(hypertensor_transformer, builder)
    train_metrics = trainer.train_epoch(train_samples)
    eval_metrics = trainer.evaluate(eval_samples)

    teacher_logits = np.stack([window.logits for window in eval_windows if window.logits is not None])
    teacher_targets = np.array([window.target_token for window in eval_windows], dtype=np.int64)
    teacher_ppl = truncated_teacher_perplexity(teacher_logits, teacher_targets, shortlist, mapping)

    metrics_payload = {
        "train_loss": train_metrics.loss,
        "train_perplexity": train_metrics.perplexity,
        "eval_loss": eval_metrics.loss,
        "eval_perplexity": eval_metrics.perplexity,
        "teacher_perplexity": teacher_ppl,
    }
    print("Training metrics:", json.dumps(metrics_payload, indent=2))
    if args.metrics_path:
        pathlib.Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": trainer.metric_log.steps[-1] if trainer.metric_log.steps else 0, **metrics_payload}) + "\n")

    if args.output:
        checkpoint = HTFTCheckpoint(
            stage1=StageState(
                model=stage1_state.model,
                srht=tuple(stage1_state.srht_params),
                countsketch=stage1_state.countsketch,
                metadata={"stage": 1},
            ),
            stage2=StageState(
                model=stage2_state.model,
                srht=tuple(stage2_state.srht_params),
                countsketch=stage2_state.countsketch,
                metadata={"stage": 2},
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
        print(f"Checkpoint saved to {args.output}")


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
        top_k=64,
        train_top_k=1024,
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
