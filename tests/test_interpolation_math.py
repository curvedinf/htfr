import numpy as np
import pytest

from htfr.interpolation import available_interpolations, get_interpolation_module


def test_available_interpolations_expected_names() -> None:
    names = set(available_interpolations())
    assert names == {"lerp", "hermite", "catmull_rom", "local_poly", "wavelet"}


def test_lerp_positive_branch_weights_and_derivative() -> None:
    module = get_interpolation_module("lerp")
    controls = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    result = module.evaluate(controls, 0.5, -1.0, 1.0, 1.0)
    np.testing.assert_allclose(result.weights, [0.0, 0.5, 0.5])
    np.testing.assert_allclose(result.value, controls @ result.weights)
    np.testing.assert_allclose(result.distance_derivative, [1.0, -1.0])


def test_lerp_negative_branch_weights_and_derivative() -> None:
    module = get_interpolation_module("lerp")
    controls = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    result = module.evaluate(controls, -0.5, -1.0, 1.0, 1.0)
    np.testing.assert_allclose(result.weights, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(result.value, controls @ result.weights)
    np.testing.assert_allclose(result.distance_derivative, [1.0, -1.0])


def test_hermite_saturates_and_zeros_derivative_outside_band() -> None:
    module = get_interpolation_module("hermite")
    controls = np.array([[0.0, 1.0, 3.0]], dtype=np.float32)
    result = module.evaluate(controls, 2.0, -1.0, 1.0, 1.0)
    np.testing.assert_allclose(result.weights, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.value, [3.0])
    np.testing.assert_allclose(result.distance_derivative, [0.0])


def test_catmull_rom_negative_branch_matches_expected_weights() -> None:
    module = get_interpolation_module("catmull_rom")
    controls = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    result = module.evaluate(controls, -0.5, -1.0, 1.0, 1.0)
    np.testing.assert_allclose(result.weights, [0.4375, 0.625, -0.0625])
    np.testing.assert_allclose(result.value, [0.5, 1.5])
    np.testing.assert_allclose(result.distance_derivative, [1.0, -1.0])


def test_local_poly_center_prefers_central_control() -> None:
    module = get_interpolation_module("local_poly")
    controls = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    result = module.evaluate(controls, 0.0, -1.0, 1.0, 1.0)
    np.testing.assert_allclose(result.weights, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(result.value, [1.0])
    np.testing.assert_allclose(result.distance_derivative, [1.0])


def test_wavelet_uses_reference_radius_for_scaling() -> None:
    module = get_interpolation_module("wavelet")
    controls = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    # dpos is small but reference radius is large, so scale should follow reference.
    result = module.evaluate(controls, 0.05, -0.1, 0.1, reference_radius=5.0)
    np.testing.assert_allclose(result.weights, [0.0, 0.02, 0.98], rtol=1e-4)
    np.testing.assert_allclose(result.value, [1.98])
    # derivative reflects slope toward the positive control.
    np.testing.assert_allclose(result.distance_derivative, [-0.4], rtol=1e-4)
