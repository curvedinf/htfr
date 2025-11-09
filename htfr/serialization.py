"""Serialization helpers for HTFR models and SRHT parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .feature_ops import SRHTParameters
from .model import HTFRModel
from .tensor import HyperTensor


@dataclass(frozen=True)
class HTFRCheckpoint:
    """Container bundling an HTFR model with projection metadata."""

    model: HTFRModel
    srht: SRHTParameters
    mapping: np.ndarray
    shortlist: np.ndarray
    unk_index: int
    metadata: Dict[str, Any]


def _stack_tensors(tensors: Iterable[HyperTensor]) -> Dict[str, np.ndarray]:
    normals = []
    deltas = []
    dneg = []
    dpos = []
    controls = []
    taus = []
    for tensor in tensors:
        normals.append(tensor.n.astype(np.float32))
        deltas.append(np.float32(tensor.delta))
        dneg.append(np.float32(tensor.dneg))
        dpos.append(np.float32(tensor.dpos))
        controls.append(tensor.C.astype(np.float32))
        taus.append(np.float32(tensor.tau))
    return {
        "tensor_normals": np.stack(normals, axis=0),
        "tensor_deltas": np.stack(deltas, axis=0),
        "tensor_dneg": np.stack(dneg, axis=0),
        "tensor_dpos": np.stack(dpos, axis=0),
        "tensor_controls": np.stack(controls, axis=0),
        "tensor_taus": np.stack(taus, axis=0),
    }


def save_htfr_checkpoint(
    path: str | Path,
    model: HTFRModel,
    srht: SRHTParameters,
    mapping: np.ndarray,
    shortlist: np.ndarray,
    unk_index: int,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Persist the HTFR model, SRHT parameters, and vocabulary mapping."""

    tensors = _stack_tensors(model.tensors)
    arrays: Dict[str, np.ndarray] = {}
    arrays.update(srht.to_dict())
    arrays.update(tensors)
    arrays.update(
        {
            "mapping": np.asarray(mapping, dtype=np.int64),
            "shortlist": np.asarray(shortlist, dtype=np.int64),
            "unk_index": np.array([unk_index], dtype=np.int32),
            "model_top_k": np.array([model.top_k], dtype=np.int32),
            "model_eta": np.array([model.eta], dtype=np.float32),
            "model_eta_g": np.array([model.eta_g], dtype=np.float32),
            "model_epsilon": np.array([model.epsilon], dtype=np.float32),
            "model_weight_mode": np.array([model.weight_mode], dtype=np.dtype("U16")),
        }
    )
    if metadata is None:
        metadata = {}
    arrays["metadata_json"] = np.array([json.dumps(metadata)], dtype=np.dtype("U"))
    np.savez_compressed(Path(path), **arrays)


def _rebuild_tensors(arrays: Dict[str, np.ndarray]) -> list[HyperTensor]:
    normals = np.asarray(arrays["tensor_normals"], dtype=np.float32)
    deltas = np.asarray(arrays["tensor_deltas"], dtype=np.float32)
    dneg = np.asarray(arrays["tensor_dneg"], dtype=np.float32)
    dpos = np.asarray(arrays["tensor_dpos"], dtype=np.float32)
    controls = np.asarray(arrays["tensor_controls"], dtype=np.float32)
    taus = np.asarray(arrays["tensor_taus"], dtype=np.float32)
    tensors: list[HyperTensor] = []
    for idx in range(normals.shape[0]):
        tensors.append(
            HyperTensor(
                normals[idx],
                float(deltas[idx]),
                float(dneg[idx]),
                float(dpos[idx]),
                controls[idx],
                tau=float(taus[idx]),
            )
        )
    return tensors


def load_htfr_checkpoint(path: str | Path) -> HTFRCheckpoint:
    """Load an HTFR checkpoint from ``path``."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        array_dict = {key: arrays[key] for key in arrays.files}
    srht = SRHTParameters.from_arrays(array_dict)
    tensors = _rebuild_tensors(array_dict)
    top_k = int(np.asarray(array_dict["model_top_k"]).item())
    eta = float(np.asarray(array_dict["model_eta"]).item())
    eta_g = float(np.asarray(array_dict["model_eta_g"]).item())
    epsilon = float(np.asarray(array_dict["model_epsilon"]).item())
    weight_mode = str(np.asarray(array_dict["model_weight_mode"], dtype=np.dtype("U"))[0])
    model = HTFRModel.from_tensors(
        tensors,
        top_k=top_k,
        weight_mode=weight_mode,
        epsilon=epsilon,
        eta=eta,
        eta_g=eta_g,
    )
    mapping = np.asarray(array_dict["mapping"], dtype=np.int64)
    shortlist = np.asarray(array_dict["shortlist"], dtype=np.int64)
    unk_index = int(np.asarray(array_dict["unk_index"]).item())
    metadata = json.loads(str(np.asarray(array_dict["metadata_json"], dtype=np.dtype("U"))[0]))
    return HTFRCheckpoint(
        model=model,
        srht=srht,
        mapping=mapping,
        shortlist=shortlist,
        unk_index=unk_index,
        metadata=metadata,
    )

