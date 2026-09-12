"""Tests for 3D extrapolation and the generalized compute_global_radius."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.extrapolation import (
    backfill_gdf_3d,
    compute_global_radius,
    standardize_xy,
)


# ---------------------------------------------------------------------------
# compute_global_radius — now D-dimensional
# ---------------------------------------------------------------------------


def test_compute_global_radius_2d_unchanged():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (50, 2))
    # Should still produce a positive radius
    R = compute_global_radius(X)
    assert R > 0
    # Range in dim-0 is max range; R = 0.5 * that
    dim_ranges = X.max(axis=0) - X.min(axis=0)
    expected = 0.5 * dim_ranges.max()
    assert abs(R - expected) < 1e-10


def test_compute_global_radius_3d():
    rng = np.random.default_rng(1)
    # z range is largest
    X = rng.uniform(0, 5, (50, 3))
    X[:, 2] = rng.uniform(0, 100, 50)  # z has much larger range
    R = compute_global_radius(X)
    dim_ranges = X.max(axis=0) - X.min(axis=0)
    expected = 0.5 * dim_ranges.max()
    assert abs(R - expected) < 1e-10


def test_compute_global_radius_1d():
    X = np.array([[0.0], [1.0], [3.0]])
    R = compute_global_radius(X)
    assert abs(R - 1.5) < 1e-10


# ---------------------------------------------------------------------------
# standardize_xy — zero-variance protection
# ---------------------------------------------------------------------------


def test_standardize_xy_3d():
    rng = np.random.default_rng(2)
    X_tr = rng.uniform(0, 10, (20, 3))
    X_full = rng.uniform(-2, 12, (100, 3))
    Xtr_std, Xf_std, mean, std = standardize_xy(X_tr, X_full)
    assert Xtr_std.shape == (20, 3)
    assert Xf_std.shape == (100, 3)
    # Training mean ≈ 0
    assert np.allclose(Xtr_std.mean(axis=0), 0, atol=0.1)


def test_standardize_xy_zero_variance_dim():
    # One dimension is constant — should not produce NaN
    X = np.column_stack([np.linspace(0, 1, 10), np.ones(10)])
    Xtr_std, _, _, std = standardize_xy(X, X)
    assert np.all(np.isfinite(Xtr_std))
    # The constant dim should have std=1 (protected), not NaN
    assert std[1] == 1.0


# ---------------------------------------------------------------------------
# backfill_gdf_3d
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_3d_gdf():
    """50 known + 20 unknown points in a 3D domain."""
    import geopandas as gpd  # noqa: PLC0415
    import shapely.geometry as shp  # noqa: PLC0415

    rng = np.random.default_rng(42)
    n_total = 70
    xs = rng.uniform(0, 1000, n_total)
    ys = rng.uniform(0, 1000, n_total)
    zs = rng.uniform(-5000, -1000, n_total)
    # True value = depth-scaled sine
    values = np.sin(xs / 300) + np.cos(ys / 300) + zs / 5000
    # Set 20 rows to NaN
    nan_idx = rng.choice(n_total, 20, replace=False)
    values[nan_idx] = np.nan

    geom = [shp.Point(x, y) for x, y in zip(xs, ys)]
    gdf = gpd.GeoDataFrame(
        {"x": xs, "y": ys, "z": zs, "value": values, "geometry": geom},
        crs="EPSG:32610",
    )
    return gdf


def test_backfill_gdf_3d_fills_nans(synthetic_3d_gdf):
    result = backfill_gdf_3d(
        synthetic_3d_gdf, value_col="value", verbose=False
    )
    assert (
        "value_extrapolated" not in synthetic_3d_gdf.columns
    )  # original unchanged
    assert "value_extrapolated" in result.columns
    assert result["value_extrapolated"].notna().all()


def test_backfill_gdf_3d_preserves_known_values(synthetic_3d_gdf):
    """Known values should not be modified (only NaN rows are filled)."""
    known_mask = synthetic_3d_gdf["value"].notna()
    result = backfill_gdf_3d(
        synthetic_3d_gdf, value_col="value", verbose=False
    )
    orig_known = synthetic_3d_gdf.loc[known_mask, "value"].to_numpy()
    filled_known = result.loc[known_mask, "value_extrapolated"].to_numpy()
    np.testing.assert_array_almost_equal(orig_known, filled_known)


def test_backfill_gdf_3d_few_points_warns():
    import geopandas as gpd  # noqa: PLC0415
    import shapely.geometry as shp  # noqa: PLC0415

    gdf = gpd.GeoDataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [-1000.0, -2000.0],
            "value": [1.0, np.nan],
            "geometry": [shp.Point(0, 0), shp.Point(1, 1)],
        },
        crs="EPSG:32610",
    )
    with pytest.warns(UserWarning, match="fewer than 4"):
        result = backfill_gdf_3d(gdf, value_col="value")
    assert "value_extrapolated" in result.columns
