import numpy as np

from htfr.context import ContextBuilder, ContextBuilderConfig, ContextSignals


def test_context_builder_shapes() -> None:
    config = ContextBuilderConfig(window_size=4, hidden_dim=3, hashed_dim=8, tail_tokens=2, stage1_target_dim=5)
    builder = ContextBuilder(config)
    signals = ContextSignals(
        token_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        hidden_states=np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    stage1_input = builder.build_stage1_input(signals)
    assert stage1_input.shape == (builder.raw_dim,)
    tail = builder.build_tail_embeddings(signals)
    assert tail.shape == (builder.tail_raw_dim,)
    target = builder.stage1_target_from_signals(signals)
    assert target.shape == (builder.stage1_target_dim,)
