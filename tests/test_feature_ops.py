import numpy as np
import pytest

from htfr.feature_ops import (
    SRHTParameters,
    ProjectionStack,
    _next_power_of_two,
    apply_block_srht,
    apply_count_sketch,
    block_rmsnorm,
    hashed_ngram_features,
    make_block_srht,
    make_count_sketch,
    srht_feature_tuple,
)


def test_next_power_of_two_exact_and_rounded() -> None:
    assert _next_power_of_two(8) == 8
    assert _next_power_of_two(9) == 16


def test_block_rmsnorm_handles_disabled_blocks() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(block_rmsnorm(data, block_size=0), data)


def test_block_rmsnorm_normalizes_each_block() -> None:
    data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    normalized = block_rmsnorm(data, block_size=2, eps=0.0)
    block1 = normalized[:, :2]
    block2 = normalized[:, 2:]
    np.testing.assert_allclose(np.sqrt(np.mean(block1**2, axis=1)), 1.0)
    np.testing.assert_allclose(np.sqrt(np.mean(block2**2, axis=1)), 1.0)


def test_make_block_srht_and_apply_block_srht_shape() -> None:
    rng = np.random.default_rng(0)
    params = make_block_srht(input_dim=4, target_dim=4, block_size=2, rng=rng)
    data = np.eye(3, 4, dtype=np.float32)
    projected = apply_block_srht(data, params)
    assert projected.shape == (3, 4)
    # block RMS normalization preserves RMS of 1 per block
    np.testing.assert_allclose(
        np.sqrt(np.mean(projected[:, :2] ** 2, axis=1)), 1.0, atol=1e-5
    )


def test_apply_block_srht_validates_input_shape() -> None:
    params = make_block_srht(input_dim=4, target_dim=2, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        apply_block_srht(np.ones((2, 3), dtype=np.float32), params)


def test_srht_parameters_roundtrip_and_feature_tuple() -> None:
    rng = np.random.default_rng(42)
    params = make_block_srht(input_dim=2, target_dim=2, rng=rng)
    arrays = params.to_dict()
    rebuilt = SRHTParameters.from_arrays(arrays)
    assert rebuilt.input_dim == params.input_dim
    assert rebuilt.target_dim == params.target_dim
    data = np.ones((1, 2), dtype=np.float32)
    features, returned = srht_feature_tuple(data, rebuilt)
    assert returned is rebuilt
    assert features.shape == (1, 2)
    assert features.dtype == np.float16


def test_apply_count_sketch_matches_dimensions() -> None:
    params = make_count_sketch(input_dim=6, output_dim=3, rng=np.random.default_rng(0))
    data = np.ones((2, 6), dtype=np.float32)
    projected = apply_count_sketch(data, params)
    assert projected.shape == (2, 3)
    # Since inputs are ones, the summed columns must be finite.
    assert np.all(np.isfinite(projected))


def test_hashed_ngram_features_respects_dimension() -> None:
    tokens = [1, 2, 3, 4]
    feats = hashed_ngram_features(tokens, dim=8, ngram=2, num_hashes=2, seed=7)
    assert feats.shape == (8,)
    assert feats.sum() == (len(tokens) - 1) * 2


def test_projection_stack_runs_full_pipeline() -> None:
    rng = np.random.default_rng(0)
    countsketch = make_count_sketch(input_dim=6, output_dim=4, rng=rng)
    srht_a = make_block_srht(input_dim=4, target_dim=4, rng=rng)
    srht_b = make_block_srht(input_dim=4, target_dim=4, rng=rng)
    stack = ProjectionStack(
        raw_dim=2,
        srht_params=(srht_a, srht_b),
        countsketch=countsketch,
        output_dtype=np.float16,
    )
    extra = np.zeros(4, dtype=np.float32)
    vec = np.array([0.5, -0.5], dtype=np.float32)
    result = stack.project(vec, extra=extra)
    assert result.shape == (srht_a.target_dim + srht_b.target_dim,)
    assert result.dtype == np.float16
