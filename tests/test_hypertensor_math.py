import numpy as np
import pytest

from htfr.hypertensor import Hypertensor, LocalResult


def make_tensor() -> Hypertensor:
    return Hypertensor(
        n=np.array([2.0, 0.0], dtype=np.float32),
        delta=0.0,
        dneg=-1.0,
        dpos=1.0,
        C=np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        tau=0.5,
    )


def test_hypertensor_normalizes_and_computes_distance() -> None:
    tensor = make_tensor()
    np.testing.assert_allclose(np.linalg.norm(tensor.n), 1.0)
    dist = tensor.distance(np.array([0.25, 0.0], dtype=np.float32))
    assert pytest.approx(dist) == 0.25


def test_hypertensor_local_returns_local_result() -> None:
    tensor = make_tensor()
    result = tensor.local(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(result, LocalResult)
    np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-6)


def test_hypertensor_from_tuple_backward_compatibility() -> None:
    params6 = (
        np.array([1.0, 0.0]),
        0.0,
        -1.0,
        1.0,
        np.ones((1, 3), dtype=np.float32),
        0.5,
    )
    tensor6 = Hypertensor.from_tuple(params6)
    assert tensor6.interpolation == "lerp"
    params8 = params6 + ("local_poly", 3.0)
    tensor8 = Hypertensor.from_tuple(params8)
    assert tensor8.reference_radius == pytest.approx(3.0)


def test_hypertensor_clone_and_to_tuple_roundtrip() -> None:
    tensor = make_tensor()
    clone = tensor.clone()
    assert clone is not tensor
    np.testing.assert_allclose(clone.to_tuple()[0], tensor.to_tuple()[0])


def test_hypertensor_validation_errors() -> None:
    with pytest.raises(ValueError):
        Hypertensor(
            n=np.array([0.0, 0.0]),
            delta=0.0,
            dneg=-1.0,
            dpos=1.0,
            C=np.ones((1, 3)),
            tau=1.0,
        )
    with pytest.raises(ValueError):
        Hypertensor(
            n=np.array([1.0, 0.0]),
            delta=0.0,
            dneg=1.0,
            dpos=2.0,
            C=np.ones((1, 3)),
            tau=1.0,
        )
    with pytest.raises(ValueError):
        Hypertensor(
            n=np.array([1.0, 0.0]),
            delta=0.0,
            dneg=-1.0,
            dpos=1.0,
            C=np.ones((1, 2)),
            tau=1.0,
        )
    tensor = make_tensor()
    tensor.n = np.zeros_like(tensor.n)
    with pytest.raises(ValueError):
        tensor.renormalize()
