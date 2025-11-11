import unittest

import numpy as np

from htfr.interpolation import available_interpolations, get_interpolation_module
from htfr.model import HTFRModel
from htfr.hypertensor import Hypertensor


def _make_tensor(interpolation: str = "lerp") -> Hypertensor:
    return Hypertensor(
        n=np.array([1.0, 0.0], dtype=np.float32),
        delta=0.0,
        dneg=-1.0,
        dpos=1.0,
        C=np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        tau=1.0,
        interpolation=interpolation,
        reference_radius=5.0,
    )


class InterpolationModuleTests(unittest.TestCase):
    def test_default_reference_scales_with_max_knn_radius(self) -> None:
        tensor = _make_tensor()
        model = HTFRModel(
            tensors=[tensor],
            max_knn_radius=2.5,
            randomize_interpolations=False,
        )
        self.assertAlmostEqual(model.interpolation_reference, 12.5)
        self.assertAlmostEqual(model.tensors[0].reference_radius, 12.5)
        self.assertEqual(model.tensors[0].interpolation, "lerp")

    def test_randomization_respects_weight_mask(self) -> None:
        tensors = [_make_tensor() for _ in range(16)]
        allowed = {"lerp", "local_poly"}
        weights = {name: (1.0 if name in allowed else 0.0) for name in available_interpolations()}
        model = HTFRModel(
            tensors=tensors,
            interpolation_weights=weights,
            rng=np.random.default_rng(42),
        )
        for tensor in model.tensors:
            self.assertIn(tensor.interpolation, allowed)

    def test_get_interpolation_module_invalid(self) -> None:
        with self.assertRaises(ValueError):
            get_interpolation_module("does-not-exist")

if __name__ == "__main__":
    unittest.main()
