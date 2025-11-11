from pathlib import Path

import numpy as np

from htfr.checkpoint import HTFTCheckpoint, StageState, load_htft_checkpoint, save_htft_checkpoint
from htfr.feature_ops import ProjectionStack, make_block_srht
from htfr.initialization import random_hypertensors
from htfr.model import HTFRModel


def _make_stage_state(rng: np.random.Generator) -> StageState:
    srht = make_block_srht(input_dim=4, target_dim=4, rng=rng)
    projector = ProjectionStack(raw_dim=4, srht_params=(srht,))
    tensors = random_hypertensors(
        count=4,
        input_dim=projector.output_dim,
        output_dim=3,
        rng=rng,
    )
    model = HTFRModel(
        tensors=tensors,
        top_k=32,
        train_top_k=128,
        eta=0.01,
        eta_g=0.001,
        randomize_interpolations=False,
    )
    return StageState(model=model, srht=(srht,), countsketch=None, metadata={"stage": 1})


def test_htft_checkpoint_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    stage1 = _make_stage_state(rng)
    stage2 = _make_stage_state(rng)
    mapping = np.arange(6, dtype=np.int64)
    shortlist = np.arange(5, dtype=np.int64)
    checkpoint = HTFTCheckpoint(
        stage1=stage1,
        stage2=stage2,
        mapping=mapping,
        shortlist=shortlist,
        unk_index=shortlist.size,
        tail_config={"tail_tokens": 4},
        metadata={"tag": "unit-test"},
    )
    save_htft_checkpoint(tmp_path / "model.npz", checkpoint)
    loaded = load_htft_checkpoint(tmp_path / "model.npz")
    assert loaded.mapping.shape == mapping.shape
    assert loaded.shortlist.shape == shortlist.shape
    assert loaded.stage1.model.top_k == 32
    assert loaded.stage1.model.train_top_k == 128
    assert loaded.stage2.model.tensors[0].C.shape[1] == 3
