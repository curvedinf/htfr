import math
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


def make_tensor_2d(delta: float = 0.0) -> Hypertensor:
    base = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    controls = np.repeat(base, 1, axis=0)
    return Hypertensor(
        n=np.array([1.0, 0.0], dtype=np.float32),
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


def test_model_select_active_reports_geometric_distance() -> None:
    tensor_a = make_tensor_2d(delta=0.0)
    tensor_b = make_tensor_2d(delta=0.5)
    model = HTFRModel.from_tensors(
        [tensor_a, tensor_b],
        top_k=2,
        train_top_k=2,
        randomize_interpolations=False,
        distance_mode="hybrid",
    )
    sample = np.array([0.2, 1.0], dtype=np.float32)
    active = model._select_active(sample, k=2)
    assert len(active) == 2
    geo_map = {idx: geo for idx, _, geo in active}
    for idx, tensor in enumerate(model.tensors):
        planar = tensor.local(sample).distance
        anchor = model._tensor_anchor(tensor).astype(np.float32)
        euclid = float(np.linalg.norm(sample - anchor))
        expected = math.sqrt(max(abs(planar), 1e-12) * max(euclid, 1e-12))
        assert math.isclose(geo_map[idx], expected, rel_tol=1e-6)


def _distance_from_mode(mode: str, sample: np.ndarray | None = None) -> float:
    tensor = make_tensor_2d(delta=0.0)
    model = HTFRModel.from_tensors(
        [tensor],
        top_k=1,
        train_top_k=1,
        randomize_interpolations=False,
        distance_mode=mode,  # type: ignore[arg-type]
        distance_exp_lambda=0.5,
        distance_clip=1e3,
    )
    vec = sample if sample is not None else np.array([0.25, 0.75], dtype=np.float32)
    active = model._select_active(vec, k=1)
    return active[0][2]


def test_distance_mode_planar() -> None:
    sample = np.array([0.25, 0.0], dtype=np.float32)
    dist = _distance_from_mode("planar", sample)
    tensor = make_tensor_2d()
    planar = abs(tensor.local(sample).distance)
    assert math.isclose(dist, planar, rel_tol=1e-6)


def test_distance_mode_euclidean() -> None:
    sample = np.array([0.25, -0.25], dtype=np.float32)
    dist = _distance_from_mode("euclidean", sample)
    tensor = make_tensor_2d()
    anchor = HTFRModel.from_tensors([tensor], randomize_interpolations=False)._tensor_anchor(tensor).astype(np.float32)
    euclid = float(np.linalg.norm(sample - anchor))
    assert math.isclose(dist, euclid, rel_tol=1e-6)


def test_distance_mode_hybrid_and_exp() -> None:
    sample = np.array([0.5, 0.0], dtype=np.float32)
    hybrid = _distance_from_mode("hybrid", sample)
    hybrid_exp = _distance_from_mode("hybrid_exp", sample)
    assert hybrid_exp > hybrid > 0.0


def test_distance_mode_invalid() -> None:
    tensor = make_tensor()
    model = HTFRModel.from_tensors([tensor], randomize_interpolations=False, distance_mode="planar")
    model.distance_mode = "unknown"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        _ = model.predict(np.array([0.0], dtype=np.float32))


def test_distance_clip_prevents_overflow() -> None:
    tensor = make_tensor_2d()
    model = HTFRModel.from_tensors(
        [tensor],
        randomize_interpolations=False,
        distance_mode="hybrid_exp",
        distance_exp_lambda=0.01,
        distance_clip=10.0,
    )
    sample = np.array([1e3, 1e3], dtype=np.float32)
    active = model._select_active(sample, k=1)
    assert math.isfinite(active[0][2])
    assert active[0][2] <= 10.0 + 1e-6


def test_prune_unmodified_removes_inactive_tensors() -> None:
    tensors = [make_tensor(), make_tensor(delta=2.0)]
    model = HTFRModel.from_tensors(
        tensors,
        top_k=1,
        train_top_k=1,
        randomize_interpolations=False,
    )
    x = np.array([0.0], dtype=np.float32)
    target = np.array([0.25], dtype=np.float32)
    model.predict_and_update(x, target, loss="mse", train=True)
    removed, revived = model.prune_unmodified()
    assert removed == 0
    assert revived >= 1
    assert len(model.tensors) == 1
    assert model._update_counts.shape[0] == 1


def test_prune_unmodified_skips_when_no_updates() -> None:
    tensors = [make_tensor(), make_tensor(delta=1.5)]
    model = HTFRModel.from_tensors(
        tensors,
        top_k=2,
        train_top_k=2,
        randomize_interpolations=False,
    )
    removed, revived = model.prune_unmodified()
    assert removed == 0
    assert revived == 0
    assert len(model.tensors) == 2
