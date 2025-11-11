"""Serialization helpers for HTFR models and SRHT parameters."""
from __future__ import annotations

from json import dumps, loads
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
    interpolations = []
    references = []
    for tensor in tensors:
        normals.append(tensor.n.astype(np.float32))
        deltas.append(np.float32(tensor.delta))
        dneg.append(np.float32(tensor.dneg))
        dpos.append(np.float32(tensor.dpos))
        controls.append(tensor.C.astype(np.float32))
        taus.append(np.float32(tensor.tau))
        interpolations.append(tensor.interpolation)
        references.append(np.float32(tensor.reference_radius))
    return {
        "tensor_normals": np.stack(normals, axis=0),
        "tensor_deltas": np.stack(deltas, axis=0),
        "tensor_dneg": np.stack(dneg, axis=0),
        "tensor_dpos": np.stack(dpos, axis=0),
        "tensor_controls": np.stack(controls, axis=0),
        "tensor_taus": np.stack(taus, axis=0),
        "tensor_interpolation": np.asarray(interpolations, dtype=np.dtype("U32")),
        "tensor_reference_radius": np.stack(references, axis=0),
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
            "model_max_knn_radius": np.array([model.max_knn_radius], dtype=np.float32),
            "model_interpolation_reference": np.array(
                [model.interpolation_reference], dtype=np.float32
            ),
            "model_locality_radius": np.array(
                [model.locality_radius if model.locality_radius is not None else np.nan],
                dtype=np.float32,
            ),
            "model_randomize_interpolations": np.array(
                [model.randomize_interpolations], dtype=np.bool_
            ),
        }
    )
    arrays["model_interpolation_weights_json"] = np.array(
        [dumps(model.interpolation_weights)], dtype=np.dtype("U")
    )
    if metadata is None:
        metadata = {}
    arrays["metadata_json"] = np.array([dumps(metadata)], dtype=np.dtype("U"))
    np.savez_compressed(Path(path), **arrays)


def _rebuild_tensors(arrays: Dict[str, np.ndarray]) -> list[HyperTensor]:
    normals = np.asarray(arrays["tensor_normals"], dtype=np.float32)
    deltas = np.asarray(arrays["tensor_deltas"], dtype=np.float32)
    dneg = np.asarray(arrays["tensor_dneg"], dtype=np.float32)
    dpos = np.asarray(arrays["tensor_dpos"], dtype=np.float32)
    controls = np.asarray(arrays["tensor_controls"], dtype=np.float32)
    taus = np.asarray(arrays["tensor_taus"], dtype=np.float32)
    interpolations = np.asarray(
        arrays.get("tensor_interpolation", np.array([])), dtype=np.dtype("U32")
    )
    references = np.asarray(
        arrays.get("tensor_reference_radius", np.full(normals.shape[0], 5.0)),
        dtype=np.float32,
    )
    tensors: list[HyperTensor] = []
    for idx in range(normals.shape[0]):
        interpolation = (
            str(interpolations[idx]) if interpolations.size else "lerp"
        )
        tensors.append(
            HyperTensor(
                normals[idx],
                float(deltas[idx]),
                float(dneg[idx]),
                float(dpos[idx]),
                controls[idx],
                tau=float(taus[idx]),
                interpolation=interpolation,
                reference_radius=float(references[idx]),
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
    max_knn_radius = float(
        np.asarray(array_dict.get("model_max_knn_radius", np.array([1.0]))).item()
    )
    interpolation_reference = float(
        np.asarray(
            array_dict.get("model_interpolation_reference", np.array([5.0 * max_knn_radius]))
        ).item()
    )
    locality_radius_raw = float(
        np.asarray(array_dict.get("model_locality_radius", np.array([np.nan]))).item()
    )
    locality_radius = None if np.isnan(locality_radius_raw) else locality_radius_raw
    randomize_interpolations = bool(
        np.asarray(
            array_dict.get("model_randomize_interpolations", np.array([True]))
        ).item()
    )
    weights_json = array_dict.get("model_interpolation_weights_json")
    if weights_json is not None:
        interpolation_weights = loads(
            str(np.asarray(weights_json, dtype=np.dtype("U"))[0])
        )
    else:
        interpolation_weights = None
    model = HTFRModel.from_tensors(
        tensors,
        top_k=top_k,
        weight_mode=weight_mode,
        epsilon=epsilon,
        eta=eta,
        eta_g=eta_g,
        max_knn_radius=max_knn_radius,
        locality_radius=locality_radius,
        interpolation_reference=interpolation_reference,
        interpolation_weights=interpolation_weights,
        randomize_interpolations=randomize_interpolations,
    )
    mapping = np.asarray(array_dict["mapping"], dtype=np.int64)
    shortlist = np.asarray(array_dict["shortlist"], dtype=np.int64)
    unk_index = int(np.asarray(array_dict["unk_index"]).item())
    metadata = loads(str(np.asarray(array_dict["metadata_json"], dtype=np.dtype("U"))[0]))
    return HTFRCheckpoint(
        model=model,
        srht=srht,
        mapping=mapping,
        shortlist=shortlist,
        unk_index=unk_index,
        metadata=metadata,
    )

