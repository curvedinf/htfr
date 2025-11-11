"""Gemma-specific teacher data adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GemmaConfig:
    """Configuration describing how to run the teacher model."""

    model_id: str = "google/gemma-3-270m"
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    seq_len: int = 128
    stride: int = 64
    max_examples: int = 2048
    device: Optional[str] = None
    collect_logits: bool = True


@dataclass(frozen=True)
class TeacherWindow:
    """Container representing one context window pulled from the teacher."""

    tokens: np.ndarray
    hidden_states: np.ndarray
    target_token: int
    logits: Optional[np.ndarray]
    rope_phases: Optional[np.ndarray] = None


def ensure_authentication(token: Optional[str]) -> None:
    """Authenticate with Hugging Face Hub when a token is supplied."""

    if token:
        login(token=token, add_to_git_credential=False)


def load_teacher(model_id: str, token: Optional[str], device: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load the tokenizer/model pair for the teacher."""

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_dataset_split(dataset: str, config: str) -> DatasetDict:
    """Load the configured dataset."""

    return load_dataset(dataset, config)


def build_token_stream(
    dataset_split: Sequence[str],
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
            "Unable to gather enough tokens. Increase token_limit or provide more data."
        )
    return torch.tensor(ids[:token_limit], dtype=torch.long)


def collect_teacher_windows(
    model: AutoModelForCausalLM,
    tokens: torch.Tensor,
    config: GemmaConfig,
) -> List[TeacherWindow]:
    """Run the teacher and capture hidden states/logits per context window."""

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    windows: List[TeacherWindow] = []
    max_start = tokens.size(0) - config.seq_len - 1
    for start in range(0, max(0, max_start) + 1, config.stride):
        stop = start + config.seq_len
        window = tokens[start:stop].to(device)
        target = tokens[stop].item()
        with torch.no_grad():
            outputs = model(window.unsqueeze(0), output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].detach().cpu().numpy()
        logits = None
        if config.collect_logits:
            logits = outputs.logits[0, -1].detach().cpu().numpy()
        windows.append(
            TeacherWindow(
                tokens=window.detach().cpu().numpy(),
                hidden_states=hidden.astype(np.float32),
                target_token=int(target),
                logits=None if logits is None else logits.astype(np.float32),
            )
        )
        if len(windows) >= config.max_examples:
            break
    if not windows:
        raise RuntimeError("No teacher windows collected; increase token budget or adjust stride.")
    return windows


def build_vocab_mapping(
    targets: np.ndarray,
    vocab_size: int,
    vocab_limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mapping tensor, shortlist ids) for the compact vocabulary."""

    counts = np.bincount(targets, minlength=vocab_size)
    topk = np.argsort(-counts)[: min(vocab_limit, vocab_size)]
    shortlist = topk
    mapping = np.full(vocab_size, fill_value=shortlist.size, dtype=np.int64)
    mapping[shortlist] = np.arange(shortlist.size, dtype=np.int64)
    return mapping, shortlist


def truncated_teacher_perplexity(
    logits: np.ndarray,
    targets: np.ndarray,
    shortlist: np.ndarray,
    mapping: np.ndarray,
) -> float:
    """Compute perplexity for the teacher under the truncated vocabulary."""

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    selected = probs[..., shortlist]
    other = np.clip(1.0 - selected.sum(axis=-1, keepdims=True), 1e-12, None)
    compact = np.concatenate([selected, other], axis=-1)
    compact_targets = mapping[targets]
    row = np.arange(compact.shape[0])
    token_probs = np.clip(compact[row, compact_targets], 1e-12, None)
    nll = -np.log(token_probs)
    return float(np.exp(np.mean(nll)))
