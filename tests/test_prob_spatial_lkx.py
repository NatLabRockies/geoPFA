"""Tests for the spatial-field wrapper in geopfa.prob.spatial (latticekrigx backend).

These tests exercise ``fit_spatial_field_gp``, which internally calls the
``build_and_fit_gp`` / ``get_predictions`` contract backed by latticekrigx.
The wrapper is used by the sequential-inference path of the probabilistic
engine to model spatial residual fields ``u_c(s)`` in logit space.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from geopfa.prob.spatial import SpatialFieldResult, fit_spatial_field_gp


def _train_grid(
    n: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 10.0, size=n)
    ys = rng.uniform(0.0, 10.0, size=n)
    residual = 0.3 * xs + 0.2 * ys + rng.normal(0.0, 0.02, size=n)
    return xs, ys, residual


def _grid_points(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.linspace(0.0, 10.0, n), np.linspace(0.0, 10.0, n))
    return xs.ravel(), ys.ravel()


def test_fit_spatial_field_gp_returns_typed_result() -> None:
    xs, ys, residual = _train_grid()
    gx, gy = _grid_points()
    train_coords = np.column_stack([xs, ys])
    grid_coords = np.column_stack([gx, gy])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_spatial_field_gp(
            train_coords,
            residual,
            grid_coords,
        )
    assert isinstance(result, SpatialFieldResult)
    assert result.u_mean.shape == (len(grid_coords),)
    assert result.u_std.shape == (len(grid_coords),)
    assert np.all(np.isfinite(result.u_mean))
    assert np.all(result.u_std >= 0)


def test_fit_spatial_field_gp_recovers_linear_residual() -> None:
    """On a noise-free linear field, posterior mean should be close to truth."""
    xs, ys, _ = _train_grid(n=60, seed=1)
    residual = 0.3 * xs + 0.2 * ys
    train_coords = np.column_stack([xs, ys])
    grid_coords = np.column_stack([xs, ys])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_spatial_field_gp(
            train_coords,
            residual,
            grid_coords,
        )
    assert np.mean(np.abs(result.u_mean - residual)) < 0.5


def test_fit_spatial_field_gp_with_constant_residual() -> None:
    """A flat residual field preserves its identified constant correction."""
    n = 30
    xs = np.linspace(0.0, 10.0, n)
    ys = np.linspace(0.0, 10.0, n)
    residual = np.full(n, 0.4)
    train_coords = np.column_stack([xs, ys])
    grid_coords = train_coords.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_spatial_field_gp(
            train_coords,
            residual,
            grid_coords,
        )
    np.testing.assert_allclose(result.u_mean, 0.4)
    np.testing.assert_array_equal(result.u_std, np.zeros(n))
    assert result.diagnostics["degenerate"] == "constant_residuals"
    assert result.diagnostics["n_train"] == n


@pytest.mark.parametrize(
    ("train_coords", "train_residuals", "grid_coords", "message"),
    [
        (
            np.zeros((6, 2)),
            np.zeros(5),
            np.zeros((3, 2)),
            "train_residuals",
        ),
        (
            np.zeros((6, 2)),
            np.zeros((6, 1)),
            np.zeros((3, 2)),
            "train_residuals",
        ),
        (
            np.array([[np.nan, 0.0], *([[0.0, 0.0]] * 5)]),
            np.zeros(6),
            np.zeros((3, 2)),
            "finite",
        ),
        (
            np.zeros((6, 2)),
            np.array([0.0, 0.0, 0.0, np.inf, 0.0, 0.0]),
            np.zeros((3, 2)),
            "finite",
        ),
    ],
)
def test_fit_spatial_field_gp_rejects_malformed_scientific_inputs(
    train_coords: np.ndarray,
    train_residuals: np.ndarray,
    grid_coords: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_spatial_field_gp(train_coords, train_residuals, grid_coords)


def test_fit_spatial_field_gp_too_few_training_points_fails_closed() -> None:
    """With fewer than four points, the spatial field is not estimable."""
    train_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    residual = np.array([0.1, -0.1])
    grid_coords = np.array([[0.5, 0.5], [2.0, 2.0]])
    with pytest.raises(ValueError, match="too few"):
        fit_spatial_field_gp(
            train_coords,
            residual,
            grid_coords,
        )


def test_fit_spatial_field_gp_diagnostics_carry_kernel_info() -> None:
    xs, ys, residual = _train_grid(n=30)
    train_coords = np.column_stack([xs, ys])
    grid_coords = train_coords.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_spatial_field_gp(
            train_coords,
            residual,
            grid_coords,
        )
    assert "kernel" in result.diagnostics
    assert "n_train" in result.diagnostics
    assert result.diagnostics["n_train"] == len(train_coords)
