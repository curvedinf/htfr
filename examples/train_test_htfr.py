"""Train a Gemma3-based HTFR test model with SRHT features."""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from htfr.feature_ops import apply_block_srht, make_block_srht
from htfr.gemma import (
    build_token_stream,
    build_vocab_mapping,
    collect_teacher_outputs,
    ensure_authentication,
    load_gemma_model,
    load_wikitext,
    log,
    truncated_teacher_perplexity,
)
from htfr.initialization import initialize_hypertensors
from htfr.model import HTFRModel
from htfr.serialization import save_htfr_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--train-tokens", type=int, default=200_000)
    parser.add_argument("--eval-tokens", type=int, default=50_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-train-examples", type=int, default=65_536)
    parser.add_argument("--max-eval-examples", type=int, default=16_384)
    parser.add_argument("--vocab-limit", type=int, default=4096)
    parser.add_argument("--output", required=True, help="Path to the output checkpoint (.npz)")
    parser.add_argument("--metadata", type=str, default=None, help="Optional JSON metadata to include")
    parser.add_argument("--init-samples", type=int, default=100_000)
    parser.add_argument("--init-tensors", type=int, default=650)
    parser.add_argument("--srht-dim", type=int, default=4096)
    parser.add_argument("--srht-block", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--eta-g", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    max_val = np.max(x)
    shifted = x - max_val
    log_sum = math.log(np.exp(shifted).sum())
    return (shifted - log_sum).astype(np.float64)


def train_model(
    model: HTFRModel,
    features: np.ndarray,
    targets: np.ndarray,
) -> None:
    for vec, cls in zip(features, targets):
        model.predict_and_update(vec, int(cls), loss="logits_ce", train=True)


def evaluate_model(
    model: HTFRModel,
    features: np.ndarray,
    targets: np.ndarray,
) -> float:
    losses: List[float] = []
    for vec, cls in zip(features, targets):
        logits = model.predict_and_update(vec, int(cls), loss="logits_ce", train=False)
        log_probs = log_softmax_np(logits)
        losses.append(-log_probs[int(cls)])
    return float(math.exp(float(np.mean(losses))))


def main() -> None:
    args = parse_args()
    ensure_authentication(args.hf_token)

    try:
        log(f"Loading teacher model {args.model}...")
        tokenizer, model = load_gemma_model(args.model, args.hf_token)
    except OSError as exc:  # pragma: no cover - depends on external auth
        raise SystemExit(
            "Failed to load the Gemma 3 reference model. Ensure you have accepted "
            "the license at https://huggingface.co/google/gemma-3-270m and pass a valid token"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    log(
        f"Loading dataset {args.dataset} ({args.dataset_config}) with train/eval targets "
        f"{args.train_tokens}/{args.eval_tokens}"
    )
    dataset = load_wikitext(args.dataset, args.dataset_config)
    train_tokens = build_token_stream(
        dataset["train"]["text"], tokenizer, args.train_tokens + args.seq_len + 1
    )
    eval_tokens = build_token_stream(
        dataset["validation"]["text"], tokenizer, args.eval_tokens + args.seq_len + 1
    )

    log("Collecting teacher outputs (train)...")
    train_hidden, _, train_targets = collect_teacher_outputs(
        model,
        train_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_train_examples,
        device=device,
        collect_logits=False,
    )
    log(
        f"Collected {train_hidden.shape[0]} training examples with hidden size {train_hidden.shape[1]}"
    )
    log("Collecting teacher outputs (eval)...")
    eval_hidden, eval_logits, eval_targets = collect_teacher_outputs(
        model,
        eval_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_eval_examples,
        device=device,
        collect_logits=True,
    )
    log(f"Collected {eval_hidden.shape[0]} evaluation examples")
    if eval_logits is None:
        raise RuntimeError("Evaluation logits missing")

    mapping, shortlist = build_vocab_mapping(train_targets, tokenizer.vocab_size, args.vocab_limit)
    num_classes = shortlist.numel() + 1
    log(f"Shortlist size: {shortlist.numel()} (num_classes={num_classes})")

    train_targets_compact = mapping[train_targets].numpy()
    eval_targets_compact = mapping[eval_targets].numpy()

    teacher_ppl = truncated_teacher_perplexity(eval_logits, eval_targets, shortlist, mapping)
    log(f"Teacher perplexity under shortlist: {teacher_ppl:.3f}")

    rng = np.random.default_rng(args.seed)
    srht_params = make_block_srht(
        input_dim=train_hidden.shape[1],
        target_dim=args.srht_dim,
        block_size=args.srht_block,
        rng=rng,
    )
    log(
        f"Applying SRHT projection (input_dim={train_hidden.shape[1]}, target_dim={srht_params.target_dim}, "
        f"block={srht_params.block_size})"
    )
    train_features = apply_block_srht(train_hidden.numpy().astype(np.float32), srht_params)
    eval_features = apply_block_srht(eval_hidden.numpy().astype(np.float32), srht_params)

    init_count = min(args.init_samples, train_features.shape[0])
    log(f"Initializing {args.init_tensors} HyperTensors from {init_count} samples...")
    one_hot = np.eye(num_classes, dtype=np.float32)[train_targets_compact[:init_count]]
    base_tensors = initialize_hypertensors(
        train_features[:init_count],
        one_hot,
        k=args.init_tensors,
        tau=1.0,
        rng=rng,
    )

    log(f"Cloning {len(base_tensors)} tensors for training...")
    tensors = [tensor.clone() for tensor in base_tensors]
    model = HTFRModel(
        tensors=tensors,
        top_k=args.top_k,
        weight_mode="softmax",
        eta=args.eta,
        eta_g=args.eta_g,
    )
    log("Training HTFR model...")
    train_model(model, train_features, train_targets_compact)
    log("Evaluating trained model...")
    ppl = evaluate_model(model, eval_features, eval_targets_compact)
    log(f"HTFR perplexity: {ppl:.3f}")

    metadata: Dict[str, object] = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "train_tokens": args.train_tokens,
        "eval_tokens": args.eval_tokens,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_train_examples": args.max_train_examples,
        "max_eval_examples": args.max_eval_examples,
        "teacher_perplexity": teacher_ppl,
        "htfr_perplexity": ppl,
        "vocab_limit": args.vocab_limit,
        "srht_dim": srht_params.target_dim,
        "srht_block": srht_params.block_size,
        "top_k": args.top_k,
        "eta": args.eta,
        "eta_g": args.eta_g,
        "init_samples": args.init_samples,
        "init_tensors": args.init_tensors,
        "trained_tensors": len(model.tensors),
        "seed": args.seed,
    }
    if args.metadata:
        metadata.update(json.loads(args.metadata))

    log(f"Saving checkpoint to {args.output}...")
    save_htfr_checkpoint(
        args.output,
        model=model,
        srht=srht_params,
        mapping=mapping.numpy(),
        shortlist=shortlist.numpy(),
        unk_index=num_classes - 1,
        metadata=metadata,
    )
    log("Done.")


if __name__ == "__main__":
    main()
