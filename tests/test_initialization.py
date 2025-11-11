import numpy as np
import pytest

from htfr.initialization import (
    ClusterResult,
    initialize_hypertensors,
    kmeans,
    principal_direction,
    random_hypertensor,
    random_hypertensors,
    reseed_tensor_around_sample,
)


def test_kmeans_validates_cluster_count() -> None:
    data = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        kmeans(data, k=3)


def test_kmeans_returns_centers_and_assignments() -> None:
    data = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    result = kmeans(data, k=2, iters=5, rng=np.random.default_rng(0))
    assert isinstance(result, ClusterResult)
    assert result.centers.shape == (2, 1)
    assert set(result.assignments.tolist()).issubset({0, 1})


def test_principal_direction_matches_first_axis() -> None:
    pts = np.stack([np.linspace(0.0, 1.0, 8), np.zeros(8)], axis=1)
    direction = principal_direction(pts)
    np.testing.assert_allclose(np.abs(direction), [1.0, 0.0])


def test_initialize_hypertensors_honors_reference_radius() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(16, 3)).astype(np.float32)
    outputs = rng.normal(size=(16, 2)).astype(np.float32)
    tensors = initialize_hypertensors(
        data,
        outputs,
        k=3,
        tau=0.5,
        reference_radius=7.5,
        rng=rng,
    )
    assert len(tensors) == 3
    for tensor in tensors:
        assert np.isclose(tensor.reference_radius, 7.5)
        np.testing.assert_allclose(np.linalg.norm(tensor.n.astype(np.float32)), 1.0, atol=5e-4)


def test_initialize_hypertensors_validates_lengths() -> None:
    data = np.ones((4, 2), dtype=np.float32)
    outputs = np.ones((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        initialize_hypertensors(data, outputs, k=2)


def test_random_hypertensor_shapes_match_expectations() -> None:
    tensor = random_hypertensor(input_dim=4, output_dim=3, rng=np.random.default_rng(0))
    assert tensor.n.shape == (4,)
    assert tensor.C.shape == (3, 3)
    assert tensor.C.dtype == np.float16


def test_random_hypertensors_batch_and_reseed() -> None:
    tensors = random_hypertensors(2, input_dim=2, output_dim=2, rng=np.random.default_rng(1))
    assert len(tensors) == 2
    sample = np.array([0.25, -0.5], dtype=np.float32)
    before = tensors[0].distance(sample)
    reseed_tensor_around_sample(tensors[0], sample, rng=np.random.default_rng(2))
    after = tensors[0].distance(sample)
    assert abs(before) > abs(after)
