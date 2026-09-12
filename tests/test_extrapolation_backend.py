"""Tests for the backend seam in build_and_fit_gp / get_predictions (P2-S01 / P9-S01)."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.extrapolation import build_and_fit_gp, get_predictions
from geopfa.spatial_lkx import LkxModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_smooth_2d(n: int = 60, seed: int = 42) -> tuple:
    """Return standardized (X, Y) from a smooth 2-D sine field."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    Y = (np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])).reshape(-1, 1)
    Y_std_val = float(Y.std())
    if Y_std_val < 1e-10:
        Y_std_val = 1.0
    Y_stdized = (Y - Y.mean()) / Y_std_val
    return X.astype(np.float64), Y_stdized.astype(np.float64)


@pytest.fixture(scope="module")
def smooth_2d():
    return _make_smooth_2d()


# ---------------------------------------------------------------------------
# latticekrigx backend: build_and_fit_gp
# ---------------------------------------------------------------------------


def test_build_and_fit_gp_lkx_returns_lkx_model(smooth_2d):
    X, Y = smooth_2d
    model, constraint_info = build_and_fit_gp(X, Y, backend="latticekrigx")
    assert isinstance(model, LkxModel)
    assert isinstance(constraint_info, dict)


def test_build_and_fit_gp_lkx_constraint_info_non_empty(smooth_2d):
    X, Y = smooth_2d
    model, constraint_info = build_and_fit_gp(X, Y, backend="latticekrigx")
    assert len(constraint_info) > 0


def test_build_and_fit_gp_default_backend_returns_lkx_model(smooth_2d):
    """Default backend (latticekrigx) should return an LkxModel."""
    X, Y = smooth_2d
    model, _ = build_and_fit_gp(X, Y)
    assert isinstance(model, LkxModel)


# ---------------------------------------------------------------------------
# latticekrigx backend: get_predictions
# ---------------------------------------------------------------------------


def test_get_predictions_lkx_shapes(smooth_2d):
    X, Y = smooth_2d
    model, _ = build_and_fit_gp(X, Y, backend="latticekrigx")
    mean, std = get_predictions(model, X, backend="latticekrigx")
    assert mean.shape == (X.shape[0],)
    assert std.shape == (X.shape[0],)


def test_get_predictions_lkx_finite(smooth_2d):
    X, Y = smooth_2d
    model, _ = build_and_fit_gp(X, Y, backend="latticekrigx")
    mean, std = get_predictions(model, X, backend="latticekrigx")
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0.0)


def test_get_predictions_lkx_destandardize(smooth_2d):
    X, Y = smooth_2d
    Y_mean = 5.0
    Y_std_val = 2.0
    model, _ = build_and_fit_gp(X, Y, backend="latticekrigx")
    mean_std, _ = get_predictions(model, X, backend="latticekrigx")
    mean_dest, std_dest = get_predictions(
        model, X, Y_mean=Y_mean, Y_std=Y_std_val, backend="latticekrigx"
    )
    np.testing.assert_allclose(
        mean_dest, mean_std * Y_std_val + Y_mean, rtol=1e-10
    )
    np.testing.assert_allclose(std_dest, _ * Y_std_val, rtol=1e-10)


def test_get_predictions_lkx_grid_reshape():
    """get_predictions with kvals_df returns (grid_mean, grid_std, xs, ys)."""
    rng = np.random.default_rng(0)
    xs = np.linspace(-1, 1, 5)
    ys = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(xs, ys)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    X_train = rng.uniform(-1, 1, (50, 2))
    Y_train = np.sin(X_train[:, 0]).reshape(-1, 1)
    model, _ = build_and_fit_gp(X_train, Y_train, backend="latticekrigx")
    import pandas as pd

    kvals_df = pd.DataFrame({"x": X_grid[:, 0], "y": X_grid[:, 1]})
    result = get_predictions(
        model, X_grid, kvals_df=kvals_df, backend="latticekrigx"
    )
    assert len(result) == 4
    grid_mean, grid_std, x_u, y_u = result
    assert grid_mean.shape == (5, 5)
    assert grid_std.shape == (5, 5)
