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
from typing import Iterable, List

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from htfr.initialization import initialize_hypertensors
from htfr.model import HTFRModel


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
    return parser.parse_args()


def ensure_authentication(token: str | None) -> None:
    """Log into Hugging Face if a token is provided."""

    if token:
        login(token=token, add_to_git_credential=False)


def build_token_stream(
    dataset_split: Iterable[str],
    tokenizer: AutoTokenizer,
    token_limit: int,
) -> torch.Tensor:
    """Tokenize enough text samples to reach ``token_limit`` tokens."""

    ids: List[int] = []
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        ids.append(int(bos_id))
    for text in dataset_split:
        if not text:
            continue
        piece = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(piece)
        if len(ids) >= token_limit:
            break
    if len(ids) < token_limit:
        raise RuntimeError(
            "Unable to gather enough tokens. Try increasing --train-tokens or --eval-tokens"
        )
    return torch.tensor(ids[:token_limit], dtype=torch.long)


def collect_teacher_outputs(
    model: AutoModelForCausalLM,
    tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    max_examples: int | None,
    device: torch.device,
    collect_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the teacher model and collect hidden states/targets/(optional) logits."""

    hidden_chunks: List[torch.Tensor] = []
    logits_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    collected = 0
    upper = len(tokens) - seq_len - 1
    for start in range(0, max(0, upper) + 1, stride):
        stop = start + seq_len
        window = tokens[start:stop].to(device)
        targets = tokens[start + 1 : stop + 1]
        if targets.numel() != seq_len:
            break
        with torch.no_grad():
            outputs = model(window.unsqueeze(0), output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].detach().cpu()
        hidden_chunks.append(hidden)
        target_chunks.append(targets.cpu())
        if collect_logits:
            logits = outputs.logits[0].detach().cpu()
            logits_chunks.append(logits)
        collected += hidden.shape[0]
        if max_examples is not None and collected >= max_examples:
            break
    if not hidden_chunks:
        raise RuntimeError("No teacher outputs collected; check token limits and sequence length")
    hidden = torch.cat(hidden_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0)
    if max_examples is not None:
        hidden = hidden[:max_examples]
        targets = targets[:max_examples]
    logits = None
    if collect_logits:
        logits = torch.cat(logits_chunks, dim=0)
        if max_examples is not None:
            logits = logits[:max_examples]
    return hidden, logits, targets


def build_vocab_mapping(
    targets: torch.Tensor,
    vocab_size: int,
    vocab_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mapping tensor, shortlist ids) for the compact vocabulary."""

    counts = torch.bincount(targets, minlength=vocab_size)
    topk = torch.topk(counts, k=min(vocab_limit, vocab_size))
    shortlist = topk.indices
    unk_index = shortlist.numel()
    mapping = torch.full((vocab_size,), fill_value=unk_index, dtype=torch.long)
    mapping[shortlist] = torch.arange(shortlist.numel(), dtype=torch.long)
    return mapping, shortlist


def truncated_teacher_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    shortlist: torch.Tensor,
    mapping: torch.Tensor,
) -> float:
    """Compute perplexity for the teacher model under the truncated vocabulary."""

    probs = torch.softmax(logits, dim=-1)
    selected = probs.index_select(dim=-1, index=shortlist)
    other = (1.0 - selected.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    compact = torch.cat([selected, other], dim=-1)
    compact_targets = mapping[targets]
    idx = torch.arange(compact.size(0))
    token_probs = compact[idx, compact_targets]
    nll = -torch.log(token_probs.clamp_min(1e-12))
    return float(torch.exp(nll.mean()).cpu().item())


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
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
        model = AutoModelForCausalLM.from_pretrained(args.model, token=args.hf_token)
    except OSError as exc:  # pragma: no cover - depends on external auth
        raise SystemExit(
            "Failed to load the Gemma 3 reference model. Ensure you have accepted "
            "the license at https://huggingface.co/google/gemma-3-270m and pass a valid token"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = load_dataset(args.dataset, args.dataset_config)
    train_tokens = build_token_stream(
        dataset["train"]["text"], tokenizer, args.train_tokens + args.seq_len + 1
    )
    eval_tokens = build_token_stream(
        dataset["validation"]["text"], tokenizer, args.eval_tokens + args.seq_len + 1
    )

    train_hidden, _, train_targets = collect_teacher_outputs(
        model,
        train_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_examples=args.max_train_examples,
        device=device,
        collect_logits=False,
    )
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

    mapping, shortlist = build_vocab_mapping(train_targets, tokenizer.vocab_size, args.vocab_limit)
    num_classes = shortlist.numel() + 1
    unk_index = num_classes - 1

    train_targets_compact = mapping[train_targets].numpy()
    eval_targets_compact = mapping[eval_targets].numpy()

    teacher_ppl = truncated_teacher_perplexity(eval_logits, eval_targets, shortlist, mapping)

    train_hidden_np = train_hidden.numpy().astype(np.float32)
    eval_hidden_np = eval_hidden.numpy().astype(np.float32)

    base_tensors = initialize_htfr(
        train_hidden_np,
        train_targets_compact,
        num_classes=num_classes,
        init_samples=args.init_samples,
        init_tensors=args.init_tensors,
        tau=1.0,
        seed=args.seed,
    )

    results = []
    for config in make_default_configs():
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
            },
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
