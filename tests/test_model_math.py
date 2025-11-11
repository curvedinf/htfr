import numpy as np
import pytest

from htfr.model import HTFRModel, locality_weights
from htfr.tensor import HyperTensor


def make_tensor(output_dim: int = 1, delta: float = 0.0) -> HyperTensor:
    base = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    controls = np.repeat(base, output_dim, axis=0)
    return HyperTensor(
        n=np.array([1.0], dtype=np.float32),
        delta=delta,
        dneg=-1.0,
        dpos=1.0,
        C=controls,
        tau=1.0,
        interpolation="lerp",
    )


def make_model(**kwargs) -> HTFRModel:
    tensor = make_tensor(**kwargs.pop("tensor_kwargs", {}))
    return HTFRModel.from_tensors(
        [tensor],
        randomize_interpolations=False,
        **kwargs,
    )


def test_locality_weights_softmax_and_inverse() -> None:
    distances = [0.1, 0.2]
    taus = [1.0, 1.0]
    softmax = locality_weights(distances, taus, mode="softmax")
    inverse = locality_weights(distances, taus, mode="inverse", epsilon=1e-3)
    np.testing.assert_allclose(softmax.sum(), 1.0)
    np.testing.assert_allclose(inverse.sum(), 1.0)


def test_locality_weights_invalid_mode() -> None:
    with pytest.raises(ValueError):
        locality_weights([0.1], [1.0], mode="unknown")


def test_model_output_dim_requires_tensors() -> None:
    model = HTFRModel(tensors=[])
    with pytest.raises(ValueError):
        _ = model.output_dim


def test_model_select_active_uses_fallback_when_radius_small() -> None:
    tensor = make_tensor()
    model = HTFRModel.from_tensors(
        [tensor],
        top_k=1,
        locality_radius=0.01,
        randomize_interpolations=False,
    )
    x = np.array([10.0], dtype=np.float32)
    active = model._select_active(x)
    assert len(active) == 1
    assert active[0][0] == 0


def test_model_predict_and_update_mse_changes_prediction() -> None:
    model = make_model(top_k=1)
    x = np.array([0.0], dtype=np.float32)
    target = np.array([0.25], dtype=np.float32)
    before = model.predict(x).copy()
    after = model.predict_and_update(x, target, loss="mse", train=True)
    assert after.shape == target.shape
    assert not np.allclose(before, model.predict(x))


def test_model_predict_and_update_logits_ce_with_int_label() -> None:
    model = make_model(top_k=1, tensor_kwargs={"output_dim": 2})
    x = np.array([0.0], dtype=np.float32)
    pred = model.predict_and_update(x, 0, loss="logits_ce", train=True)
    assert pred.shape == (2,)


def test_model_interpolation_weights_must_sum_positive() -> None:
    tensor = make_tensor()
    model = HTFRModel.from_tensors(
        [tensor],
        interpolation_weights={name: 0.0 for name in ["lerp", "hermite"]},
        randomize_interpolations=False,
    )
    model.interpolation_weights = {name: 0.0 for name in model.interpolation_weights}
    with pytest.raises(ValueError):
        model._interpolation_probabilities()
