import numpy as np
import pytest

from htfr.model import HTFRModel, locality_weights
from htfr.hypertensor import Hypertensor


def make_tensor(output_dim: int = 1, delta: float = 0.0) -> Hypertensor:
    base = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    controls = np.repeat(base, output_dim, axis=0)
    return Hypertensor(
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
    active = model._select_active(x, k=1)
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


def test_seed_random_tensors_appends_new_entries() -> None:
    tensor = make_tensor()
    model = HTFRModel.from_tensors([tensor], randomize_interpolations=False)
    model.seed_random_tensors(count=2, input_dim=1, output_dim=tensor.output_dim)
    assert len(model.tensors) == 3


def test_high_error_queue_triggers_relocation() -> None:
    tensor = make_tensor()
    model = HTFRModel.from_tensors(
        [tensor],
        randomize_interpolations=False,
        error_threshold=1e-4,
        relocation_interval=1,
        max_error_queue=4,
    )
    x = np.array([0.0], dtype=np.float32)
    target = np.array([10.0], dtype=np.float32)
    model.predict_and_update(x, target, loss="mse", train=True)
    assert len(model._error_queue) == 0
    assert model._usage_counts[0] >= 1.0


def test_model_select_active_invokes_faiss_index() -> None:
    model = make_model(top_k=1)

    class StubIndex:
        def __init__(self) -> None:
            self.size = 1
            self.searched = False

        def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            self.searched = True
            return np.array([0.0], dtype=np.float32), np.array([0], dtype=np.int64)

    stub = StubIndex()
    model._faiss_index = stub  # type: ignore[attr-defined]
    model._faiss_dirty = False  # type: ignore[attr-defined]
    _ = model.predict(np.array([0.0], dtype=np.float32))
    assert stub.searched
