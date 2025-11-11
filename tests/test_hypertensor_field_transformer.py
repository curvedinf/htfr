import numpy as np

from htfr.feature_ops import ProjectionStack, make_block_srht
from htfr.hypertensor_field_transformer import HypertensorFieldTransformer, StageRuntime
from htfr.initialization import random_hypertensors
from htfr.model import HTFRModel


def _make_stage(raw_dim: int, target_dim: int, output_dim: int, tensor_count: int) -> tuple[StageRuntime, int]:
    rng = np.random.default_rng(0)
    srht = make_block_srht(input_dim=raw_dim, target_dim=target_dim, rng=rng)
    projector = ProjectionStack(raw_dim=raw_dim, srht_params=(srht,), output_dtype=np.float16)
    tensors = random_hypertensors(
        tensor_count,
        input_dim=target_dim,
        output_dim=output_dim,
        rng=np.random.default_rng(1),
    )
    k = min(2, tensor_count)
    model = HTFRModel.from_tensors(
        tensors,
        top_k=k,
        train_top_k=k,
        randomize_interpolations=False,
    )
    return StageRuntime(projector=projector, model=model, loss="mse" if output_dim != 3 else "logits_ce"), output_dim


def test_hypertensor_field_transformer_step_and_diagnostics() -> None:
    stage1, embedding_dim = _make_stage(raw_dim=3, target_dim=4, output_dim=2, tensor_count=3)
    stage2_projector_dim = embedding_dim + 2  # tail tokens (2) × 1-dim embedding
    stage2_srht = make_block_srht(input_dim=stage2_projector_dim, target_dim=4, rng=np.random.default_rng(2))
    stage2_projector = ProjectionStack(raw_dim=stage2_projector_dim, srht_params=(stage2_srht,), output_dtype=np.float16)
    stage2_tensors = random_hypertensors(3, input_dim=4, output_dim=3, rng=np.random.default_rng(3))
    stage2_model = HTFRModel.from_tensors(
        stage2_tensors,
        top_k=2,
        train_top_k=2,
        randomize_interpolations=False,
    )
    stage2 = StageRuntime(projector=stage2_projector, model=stage2_model, loss="logits_ce")

    htft_model = HypertensorFieldTransformer(stage1, stage2, tail_token_count=2, tail_embedding_dim=1)
    context = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    tail_embeddings = np.array([[0.05], [0.1]], dtype=np.float32)
    stage1_target = np.zeros(embedding_dim, dtype=np.float32)
    logits, embedding = htft_model.step(
        context,
        tail_embeddings,
        target_token=1,
        stage1_target=stage1_target,
        train=True,
    )
    assert logits.shape == (3,)
    assert embedding.shape == (embedding_dim,)

    # Ensure inference path works without targets.
    eval_logits, _ = htft_model.step(context, tail_embeddings, target_token=None, stage1_target=None, train=False)
    assert eval_logits.shape == (3,)

    diag = htft_model.diagnostics()
    assert set(diag.keys()) == {"stage1_usage", "stage1_loss", "stage2_usage", "stage2_loss"}
