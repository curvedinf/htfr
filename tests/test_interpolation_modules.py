import os
import tempfile
import unittest

import numpy as np

from htfr.interpolation import available_interpolations, get_interpolation_module
from htfr.model import HTFRModel
from htfr.serialization import load_htfr_checkpoint, save_htfr_checkpoint
from htfr.tensor import HyperTensor
from htfr.feature_ops import SRHTParameters


def _make_tensor(interpolation: str = "lerp") -> HyperTensor:
    return HyperTensor(
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

    def test_checkpoint_roundtrip_preserves_interpolation_config(self) -> None:
        tensor = _make_tensor()
        model = HTFRModel(
            tensors=[tensor],
            max_knn_radius=0.75,
            interpolation_reference=3.0,
            interpolation_weights={"lerp": 2.0, "hermite": 0.0},
            randomize_interpolations=False,
        )
        srht = SRHTParameters(
            signs=np.ones(2, dtype=np.float32),
            permutation=np.arange(2, dtype=np.int64),
            input_dim=2,
            padded_dim=2,
            target_dim=2,
            block_size=2,
            block_eps=1e-6,
            scale=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.npz")
            save_htfr_checkpoint(
                path,
                model,
                srht,
                mapping=np.arange(2, dtype=np.int64),
                shortlist=np.arange(1, dtype=np.int64),
                unk_index=0,
                metadata={"tag": "unit-test"},
            )
            checkpoint = load_htfr_checkpoint(path)
        loaded_model = checkpoint.model
        self.assertAlmostEqual(loaded_model.interpolation_reference, 3.0)
        self.assertEqual(loaded_model.max_knn_radius, 0.75)
        self.assertFalse(loaded_model.randomize_interpolations)
        self.assertEqual(loaded_model.locality_radius, 3.0)
        self.assertIn("lerp", loaded_model.interpolation_weights)
        self.assertEqual(loaded_model.interpolation_weights["hermite"], 0.0)
        self.assertAlmostEqual(loaded_model.tensors[0].reference_radius, 3.0)


if __name__ == "__main__":
    unittest.main()
