"""Helpers for working with the Gemma 3 teacher model."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(message: str) -> None:
    """Print ``message`` immediately (flush-enabled)."""

    print(message, flush=True)


def ensure_authentication(token: str | None) -> None:
    """Log into Hugging Face if ``token`` is provided."""

    if token:
        login(token=token, add_to_git_credential=False)


def load_gemma_model(
    model_id: str,
    token: str | None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the Gemma tokenizer and model with optional authentication."""

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
    return tokenizer, model


def load_wikitext(
    dataset: str = "wikitext",
    config: str = "wikitext-2-raw-v1",
) -> DatasetDict:
    """Load the dataset used for Gemma teacher distillation."""

    return load_dataset(dataset, config)


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


__all__ = [
    "build_token_stream",
    "collect_teacher_outputs",
    "ensure_authentication",
    "load_gemma_model",
    "load_wikitext",
    "log",
    "build_vocab_mapping",
    "truncated_teacher_perplexity",
]

