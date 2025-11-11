import numpy as np
import pytest

from htfr.feature_ops import (
    SRHTParameters,
    _next_power_of_two,
    apply_block_srht,
    block_rmsnorm,
    make_block_srht,
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
