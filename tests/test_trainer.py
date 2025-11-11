import numpy as np

from htfr.context import ContextBuilder, ContextBuilderConfig, ContextSample, ContextSignals
from htfr.feature_ops import ProjectionStack, make_block_srht
from htfr.hypertensor_field_transformer import HypertensorFieldTransformer, StageRuntime
from htfr.initialization import random_hypertensors
from htfr.model import HTFRModel
from htfr.trainer import HTFTTrainer


def _make_stage(projector: ProjectionStack, output_dim: int, rng: np.random.Generator) -> StageRuntime:
    tensors = random_hypertensors(
        count=4,
        input_dim=projector.output_dim,
        output_dim=output_dim,
        rng=rng,
    )
    model = HTFRModel(
        tensors=tensors,
        top_k=4,
        train_top_k=4,
        eta=0.05,
        eta_g=0.01,
        randomize_interpolations=False,
    )
    loss = "mse" if output_dim != 2 else "logits_ce"
    return StageRuntime(projector=projector, model=model, loss=loss)


def test_htft_trainer_updates_metrics() -> None:
    rng = np.random.default_rng(0)
    builder = ContextBuilder(
        ContextBuilderConfig(window_size=2, hidden_dim=2, hashed_dim=4, tail_tokens=1, stage1_target_dim=3)
    )
    stage1_cs = make_block_srht(input_dim=builder.raw_dim, target_dim=builder.raw_dim, rng=rng)
    stage1_projector = ProjectionStack(raw_dim=builder.raw_dim, srht_params=(stage1_cs,))
    stage2_raw_dim = builder.stage1_target_dim + builder.tail_raw_dim
    stage2_cs = make_block_srht(input_dim=stage2_raw_dim, target_dim=stage2_raw_dim, rng=rng)
    stage2_projector = ProjectionStack(raw_dim=stage2_raw_dim, srht_params=(stage2_cs,))
    stage1 = _make_stage(stage1_projector, builder.stage1_target_dim, rng)
    stage2 = _make_stage(stage2_projector, output_dim=2, rng=rng)
    htft_model = HypertensorFieldTransformer(stage1, stage2, tail_token_count=1, tail_embedding_dim=2)

    samples = []
    for idx in range(10):
        hidden = np.full((2, 2), idx, dtype=np.float32)
        signals = ContextSignals(
            token_ids=np.array([idx, idx + 1], dtype=np.int64),
            hidden_states=hidden,
        )
        samples.append(
            ContextSample(
                signals=signals,
                target_token=idx % 2,
                stage1_target=builder.stage1_target_from_signals(signals),
                teacher_logits=None,
            )
        )

    trainer = HTFTTrainer(htft_model, builder)
    train_metrics = trainer.train_epoch(samples)
    eval_metrics = trainer.evaluate(samples)
    assert train_metrics.samples == len(samples)
    assert eval_metrics.perplexity > 0.0
