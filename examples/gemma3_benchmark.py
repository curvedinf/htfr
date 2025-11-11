"""Benchmark HTFR against the Gemma 3 270M reference model.

This script downloads the `google/gemma-3-270m` weights (a gated repo on
Hugging Face) and evaluates perplexity for both the reference model and a
collection of HTFR configurations.  The benchmark follows these steps:

1. Download and authenticate against the gated Gemma 3 270M repository.
2. Build a compact training/evaluation corpus from WikiText-2.
3. Run the transformer to obtain hidden states and teacher perplexities.
4. Train several HTFR variants that operate on the transformer's hidden
   representations and predict the next-token distribution (with an
   aggregated ``UNK`` class for all tokens outside a configurable
   high-frequency shortlist).
5. Report the resulting perplexities and optionally dump them as JSON.

Usage
-----

.. code-block:: bash

    python examples/gemma3_benchmark.py \
        --hf-token <HF_TOKEN> \
        --train-tokens 6000 \
        --eval-tokens 3000 \
        --seq-len 128 \
        --output results.json

The Hugging Face token can also be supplied via the ``HF_TOKEN``
environment variable.  You must explicitly accept the Gemma 3 usage
license on Hugging Face before running the script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from htfr.feature_ops import apply_block_srht
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
from htfr.serialization import load_htfr_checkpoint


@dataclass(frozen=True)
class HTFRBenchmarkConfig:
    """Configuration describing a single HTFR run."""

    name: str
    top_k: int
    weight_mode: str
    eta: float
    eta_g: float
    tau: float = 1.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token with access to google/gemma-3-270m",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-270m",
        help="Identifier of the Gemma 3 reference model",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Dataset hub identifier used for the benchmark",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="Configuration name for the dataset",
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=8000,
        help="Number of tokens to draw for HTFR training",
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=4000,
        help="Number of tokens to draw for evaluation",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Context length used when running the teacher model",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride (in tokens) between teacher windows",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=4096,
        help="Maximum number of training examples fed into HTFR",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=2048,
        help="Maximum number of evaluation examples",
    )
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=4096,
        help=(
            "Maximum number of high-frequency tokens preserved explicitly. "
            "All remaining tokens are aggregated into a single UNK class."
        ),
    )
    parser.add_argument(
        "--init-samples",
        type=int,
        default=512,
        help="Number of samples used to initialize HyperTensors",
    )
    parser.add_argument(
        "--init-tensors",
        type=int,
        default=32,
        help="Number of HyperTensors produced during initialization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path for dumping benchmark results",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        default=None,
        help="Optional path to a pre-trained HTFR checkpoint to evaluate",
    )
    return parser.parse_args()


def log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    max_val = np.max(x)
    shifted = x - max_val
    log_sum = math.log(np.exp(shifted).sum())
    return (shifted - log_sum).astype(np.float64)


def initialize_htfr(
    hidden: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    init_samples: int,
    init_tensors: int,
    tau: float,
    seed: int,
) -> List:
    rng = np.random.default_rng(seed)
    sample_count = min(init_samples, hidden.shape[0])
    init_hidden = hidden[:sample_count]
    init_targets = targets[:sample_count]
    one_hot = np.eye(num_classes, dtype=np.float32)[init_targets]
    return initialize_hypertensors(init_hidden, one_hot, k=init_tensors, tau=tau, rng=rng)


def train_htfr(
    model: HTFRModel,
    hidden: np.ndarray,
    targets: np.ndarray,
) -> None:
    for vec, cls in zip(hidden, targets):
        model.predict_and_update(vec, int(cls), loss="logits_ce", train=True)


def evaluate_htfr(
    model: HTFRModel,
    hidden: np.ndarray,
    targets: np.ndarray,
) -> float:
    losses: List[float] = []
    for vec, cls in zip(hidden, targets):
        logits = model.predict_and_update(vec, int(cls), loss="logits_ce", train=False)
        log_probs = log_softmax_np(logits)
        losses.append(-log_probs[int(cls)])
    return float(math.exp(float(np.mean(losses))))


def make_default_configs() -> List[HTFRBenchmarkConfig]:
    return [
        HTFRBenchmarkConfig(name="top4_softmax", top_k=4, weight_mode="softmax", eta=0.05, eta_g=0.005),
        HTFRBenchmarkConfig(name="top8_softmax", top_k=8, weight_mode="softmax", eta=0.05, eta_g=0.005),
        HTFRBenchmarkConfig(name="top4_inverse", top_k=4, weight_mode="inverse", eta=0.05, eta_g=0.005),
        HTFRBenchmarkConfig(name="top4_softmax_lr", top_k=4, weight_mode="softmax", eta=0.02, eta_g=0.002),
    ]


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
    log(f"Teacher model ready on {device}.")

    log(
        f"Loading dataset {args.dataset} ({args.dataset_config}) with train/eval token targets "
        f"{args.train_tokens}/{args.eval_tokens}..."
    )
    dataset = load_wikitext(args.dataset, args.dataset_config)
    train_tokens = build_token_stream(
        dataset["train"]["text"], tokenizer, args.train_tokens + args.seq_len + 1
    )
    eval_tokens = build_token_stream(
        dataset["validation"]["text"], tokenizer, args.eval_tokens + args.seq_len + 1
    )

    log("Collecting teacher outputs for training split...")
    train_hidden, _, train_targets = collect_teacher_outputs(
        model,
        train_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_train_examples,
        device=device,
        collect_logits=False,
    )
    log(f"Collected {train_hidden.shape[0]} training examples with hidden size {train_hidden.shape[1]}.")
    log("Collecting teacher outputs for evaluation split...")
    eval_hidden, eval_logits, eval_targets = collect_teacher_outputs(
        model,
        eval_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_eval_examples,
        device=device,
        collect_logits=True,
    )
    assert eval_logits is not None
    log(f"Collected {eval_hidden.shape[0]} evaluation examples.")

    log(f"Building truncated vocabulary with limit {args.vocab_limit}...")
    mapping, shortlist = build_vocab_mapping(train_targets, tokenizer.vocab_size, args.vocab_limit)
    num_classes = shortlist.numel() + 1
    unk_index = num_classes - 1
    log(f"Shortlist size: {shortlist.numel()} tokens (+UNK index {unk_index}).")

    train_targets_compact = mapping[train_targets].numpy()
    eval_targets_compact = mapping[eval_targets].numpy()

    teacher_ppl = truncated_teacher_perplexity(eval_logits, eval_targets, shortlist, mapping)
    log(f"Teacher truncated perplexity: {teacher_ppl:.3f}")

    train_hidden_np = train_hidden.numpy().astype(np.float32)
    eval_hidden_np = eval_hidden.numpy().astype(np.float32)

    log(
        "Initializing HyperTensors from {:d} samples...".format(
            min(args.init_samples, train_hidden_np.shape[0])
        )
    )
    base_tensors = initialize_htfr(
        train_hidden_np,
        train_targets_compact,
        num_classes=num_classes,
        init_samples=args.init_samples,
        init_tensors=args.init_tensors,
        tau=1.0,
        seed=args.seed,
    )
    log(f"Initialization produced {len(base_tensors)} HyperTensors.")

    results = []
    if args.test_model:
        log(f"Loading pre-trained HTFR checkpoint from {args.test_model}...")
        checkpoint = load_htfr_checkpoint(args.test_model)
        if checkpoint.mapping.shape[0] != tokenizer.vocab_size:
            log(
                "Warning: checkpoint mapping size {} differs from tokenizer vocab {}".format(
                    checkpoint.mapping.shape[0], tokenizer.vocab_size
                )
            )
        srht_hidden = apply_block_srht(eval_hidden_np, checkpoint.srht)
        mapped_targets = checkpoint.mapping[eval_targets.numpy()]
        losses = []
        for vec, cls in zip(srht_hidden, mapped_targets):
            logits = checkpoint.model.predict_and_update(vec, int(cls), loss="logits_ce", train=False)
            log_probs = log_softmax_np(logits)
            losses.append(-log_probs[int(cls)])
        ppl = float(math.exp(float(np.mean(losses))))
        teacher_baseline = truncated_teacher_perplexity(
            eval_logits,
            eval_targets,
            torch.tensor(checkpoint.shortlist, dtype=torch.long),
            torch.tensor(checkpoint.mapping, dtype=torch.long),
        )
        log(
            "Loaded test HTFR perplexity {:.3f} (teacher {:.3f}, shortlist size {})".format(
                ppl, teacher_baseline, checkpoint.shortlist.size
            )
        )
        results.append({"config": "test_model", "perplexity": ppl})
    for config in make_default_configs():
        log(
            "Training HTFR configuration '{name}' (top_k={top_k}, weight_mode={weight_mode}, eta={eta}, eta_g={eta_g})...".format(
                name=config.name,
                top_k=config.top_k,
                weight_mode=config.weight_mode,
                eta=config.eta,
                eta_g=config.eta_g,
            )
        )
        model_tensors = [tensor.clone() for tensor in base_tensors]
        htfr_model = HTFRModel(
            tensors=model_tensors,
            top_k=config.top_k,
            weight_mode=config.weight_mode,
            eta=config.eta,
            eta_g=config.eta_g,
        )
        train_htfr(htfr_model, train_hidden_np, train_targets_compact)
        ppl = evaluate_htfr(htfr_model, eval_hidden_np, eval_targets_compact)
        results.append({"config": config.name, "perplexity": ppl})
        log(f"Finished '{config.name}' with perplexity {ppl:.3f}.")

    print("Gemma 3 (truncated) perplexity: {:.3f}".format(teacher_ppl))
    for entry in results:
        print("HTFR {:>20s}: {:.3f}".format(entry["config"], entry["perplexity"]))

    if args.output:
        payload = {
            "teacher_perplexity": teacher_ppl,
            "htfr": results,
            "metadata": {
                "model": args.model,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "train_tokens": args.train_tokens,
                "eval_tokens": args.eval_tokens,
                "seq_len": args.seq_len,
                "stride": args.stride,
                "vocab_limit": args.vocab_limit,
                "num_classes": num_classes,
                "unk_index": unk_index,
                "test_model": args.test_model,
            },
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
