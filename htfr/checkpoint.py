"""Serialization helpers for Hypertensor Field Transformer checkpoints."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .feature_ops import CountSketchParameters, SRHTParameters
from .model import HTFRModel
from .hypertensor import Hypertensor


@dataclass(frozen=True)
class StageState:
    """Model + projection parameters for a single stage."""

    model: HTFRModel
    srht: Tuple[SRHTParameters, ...]
    countsketch: Optional[CountSketchParameters]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class HTFTCheckpoint:
    """Bundle containing both stages of the Hypertensor Field Transformer."""

    stage1: StageState
    stage2: StageState
    mapping: np.ndarray
    shortlist: np.ndarray
    unk_index: int
    tail_config: Dict[str, Any]
    metadata: Dict[str, Any]


def save_htft_checkpoint(path: str | Path, checkpoint: HTFTCheckpoint) -> None:
    """Persist the provided checkpoint to ``path``."""

    arrays: Dict[str, np.ndarray] = {}
    arrays.update(_serialize_stage("stage1", checkpoint.stage1))
    arrays.update(_serialize_stage("stage2", checkpoint.stage2))
    arrays["mapping"] = np.asarray(checkpoint.mapping, dtype=np.int64)
    arrays["shortlist"] = np.asarray(checkpoint.shortlist, dtype=np.int64)
    arrays["unk_index"] = np.array([checkpoint.unk_index], dtype=np.int32)
    arrays["tail_config_json"] = np.array(
        [json.dumps(checkpoint.tail_config or {})], dtype=np.dtype("U")
    )
    arrays["metadata_json"] = np.array([json.dumps(checkpoint.metadata or {})], dtype=np.dtype("U"))
    np.savez_compressed(Path(path), **arrays)


def load_htft_checkpoint(path: str | Path) -> HTFTCheckpoint:
    """Load a checkpoint previously saved via :func:`save_htft_checkpoint`."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    stage1 = _deserialize_stage("stage1", data)
    stage2 = _deserialize_stage("stage2", data)
    mapping = np.asarray(data["mapping"], dtype=np.int64)
    shortlist = np.asarray(data["shortlist"], dtype=np.int64)
    unk_index = int(np.asarray(data["unk_index"]).item())
    tail_config = json.loads(str(np.asarray(data["tail_config_json"], dtype=np.dtype("U"))[0]))
    metadata = json.loads(str(np.asarray(data["metadata_json"], dtype=np.dtype("U"))[0]))
    return HTFTCheckpoint(
        stage1=stage1,
        stage2=stage2,
        mapping=mapping,
        shortlist=shortlist,
        unk_index=unk_index,
        tail_config=tail_config,
        metadata=metadata,
    )


def _serialize_stage(prefix: str, state: StageState) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    arrays.update(_stack_tensors(state.model.tensors, prefix))
    arrays.update(_serialize_model_config(prefix, state.model))
    arrays.update(_serialize_srht(prefix, state.srht))
    arrays.update(_serialize_countsketch(prefix, state.countsketch))
    arrays[f"{prefix}_metadata_json"] = np.array(
        [json.dumps(state.metadata or {})], dtype=np.dtype("U")
    )
    return arrays


def _deserialize_stage(prefix: str, arrays: Dict[str, np.ndarray]) -> StageState:
    tensors = _rebuild_tensors(prefix, arrays)
    model = HTFRModel.from_tensors(
        tensors,
        top_k=int(np.asarray(arrays[f"{prefix}_model_top_k"]).item()),
        train_top_k=int(np.asarray(arrays[f"{prefix}_model_train_top_k"]).item())
        if f"{prefix}_model_train_top_k" in arrays
        else None,
        weight_mode=str(np.asarray(arrays[f"{prefix}_model_weight_mode"], dtype=np.dtype("U"))[0]),
        epsilon=float(np.asarray(arrays[f"{prefix}_model_epsilon"]).item()),
        eta=float(np.asarray(arrays[f"{prefix}_model_eta"]).item()),
        eta_g=float(np.asarray(arrays[f"{prefix}_model_eta_g"]).item()),
        max_knn_radius=float(np.asarray(arrays[f"{prefix}_model_max_knn_radius"]).item()),
        locality_radius=float(np.asarray(arrays[f"{prefix}_model_locality_radius"]).item()),
        interpolation_reference=float(
            np.asarray(arrays[f"{prefix}_model_interpolation_reference"]).item()
        ),
        interpolation_weights=json.loads(
            str(np.asarray(arrays[f"{prefix}_model_interpolation_weights"], dtype=np.dtype("U"))[0])
        ),
        randomize_interpolations=bool(
            np.asarray(arrays[f"{prefix}_model_randomize_interpolations"]).item()
        ),
    )
    srht_count = int(np.asarray(arrays[f"{prefix}_srht_count"]).item())
    srht_params = []
    for idx in range(srht_count):
        srht_arrays = {
            key.replace(f"{prefix}_srht_{idx}_", "", 1): arrays[key]
            for key in arrays
            if key.startswith(f"{prefix}_srht_{idx}_")
        }
        srht_params.append(SRHTParameters.from_arrays(srht_arrays))
    countsketch = None
    if f"{prefix}_cs_input_dim" in arrays:
        countsketch = CountSketchParameters.from_arrays(
            {
                "cs_bucket_indices": arrays[f"{prefix}_cs_bucket_indices"],
                "cs_signs": arrays[f"{prefix}_cs_signs"],
                "cs_input_dim": arrays[f"{prefix}_cs_input_dim"],
                "cs_output_dim": arrays[f"{prefix}_cs_output_dim"],
            }
        )
    metadata = json.loads(
        str(np.asarray(arrays[f"{prefix}_metadata_json"], dtype=np.dtype("U"))[0])
    )
    return StageState(model=model, srht=tuple(srht_params), countsketch=countsketch, metadata=metadata)


def _stack_tensors(tensors: Iterable[Hypertensor], prefix: str) -> Dict[str, np.ndarray]:
    normals = []
    deltas = []
    dneg = []
    dpos = []
    controls = []
    taus = []
    interpolations = []
    references = []
    for tensor in tensors:
        normals.append(tensor.n.astype(np.float16))
        deltas.append(np.float16(tensor.delta))
        dneg.append(np.float16(tensor.dneg))
        dpos.append(np.float16(tensor.dpos))
        controls.append(tensor.C.astype(np.float16))
        taus.append(np.float16(tensor.tau))
        interpolations.append(tensor.interpolation)
        references.append(np.float16(tensor.reference_radius))
    return {
        f"{prefix}_tensor_normals": np.stack(normals, axis=0),
        f"{prefix}_tensor_deltas": np.stack(deltas, axis=0),
        f"{prefix}_tensor_dneg": np.stack(dneg, axis=0),
        f"{prefix}_tensor_dpos": np.stack(dpos, axis=0),
        f"{prefix}_tensor_controls": np.stack(controls, axis=0),
        f"{prefix}_tensor_taus": np.stack(taus, axis=0),
        f"{prefix}_tensor_interpolations": np.asarray(interpolations, dtype=np.dtype("U16")),
        f"{prefix}_tensor_reference_radius": np.stack(references, axis=0),
    }


def _rebuild_tensors(prefix: str, arrays: Dict[str, np.ndarray]) -> list[Hypertensor]:
    normals = np.asarray(arrays[f"{prefix}_tensor_normals"], dtype=np.float32)
    deltas = np.asarray(arrays[f"{prefix}_tensor_deltas"], dtype=np.float32)
    dneg = np.asarray(arrays[f"{prefix}_tensor_dneg"], dtype=np.float32)
    dpos = np.asarray(arrays[f"{prefix}_tensor_dpos"], dtype=np.float32)
    controls = np.asarray(arrays[f"{prefix}_tensor_controls"], dtype=np.float32)
    taus = np.asarray(arrays[f"{prefix}_tensor_taus"], dtype=np.float32)
    interpolations = np.asarray(
        arrays[f"{prefix}_tensor_interpolations"], dtype=np.dtype("U16")
    )
    references = np.asarray(arrays[f"{prefix}_tensor_reference_radius"], dtype=np.float32)
    tensors: list[Hypertensor] = []
    for idx in range(normals.shape[0]):
        tensors.append(
            Hypertensor(
                normals[idx],
                float(deltas[idx]),
                float(dneg[idx]),
                float(dpos[idx]),
                controls[idx],
                tau=float(taus[idx]),
                interpolation=str(interpolations[idx]),
                reference_radius=float(references[idx]),
            )
        )
    return tensors


def _serialize_model_config(prefix: str, model: HTFRModel) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}_model_top_k": np.array([model.top_k], dtype=np.int32),
        f"{prefix}_model_train_top_k": np.array([model.train_top_k], dtype=np.int32),
        f"{prefix}_model_eta": np.array([model.eta], dtype=np.float32),
        f"{prefix}_model_eta_g": np.array([model.eta_g], dtype=np.float32),
        f"{prefix}_model_epsilon": np.array([model.epsilon], dtype=np.float32),
        f"{prefix}_model_weight_mode": np.array([model.weight_mode], dtype=np.dtype("U16")),
        f"{prefix}_model_max_knn_radius": np.array([model.max_knn_radius], dtype=np.float32),
        f"{prefix}_model_locality_radius": np.array(
            [model.locality_radius if model.locality_radius is not None else model.interpolation_reference],
            dtype=np.float32,
        ),
        f"{prefix}_model_interpolation_reference": np.array(
            [model.interpolation_reference], dtype=np.float32
        ),
        f"{prefix}_model_randomize_interpolations": np.array(
            [model.randomize_interpolations], dtype=np.bool_
        ),
        f"{prefix}_model_interpolation_weights": np.array(
            [json.dumps(model.interpolation_weights)], dtype=np.dtype("U")
        ),
    }


def _serialize_srht(prefix: str, srht_params: Sequence[SRHTParameters]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        f"{prefix}_srht_count": np.array([len(srht_params)], dtype=np.int32)
    }
    for idx, params in enumerate(srht_params):
        for key, value in params.to_dict().items():
            arrays[f"{prefix}_srht_{idx}_{key}"] = value
    return arrays


def _serialize_countsketch(prefix: str, params: Optional[CountSketchParameters]) -> Dict[str, np.ndarray]:
    if params is None:
        return {}
    return {
        f"{prefix}_cs_bucket_indices": params.bucket_indices,
        f"{prefix}_cs_signs": params.signs,
        f"{prefix}_cs_input_dim": np.array([params.input_dim], dtype=np.int32),
        f"{prefix}_cs_output_dim": np.array([params.output_dim], dtype=np.int32),
    }
