"""Tests for joint three-dimensional LatticeKrigX backfilling.

Verifies that the model fills all NaNs and captures lateral and vertical
structure.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as shp

from geopfa.extrapolation import backfill_gdf_3d


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def _make_synthetic_3d_gdf(
    n_total: int = 200,
    missing_frac: float = 0.30,
    seed: int = 0,
) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """Return (gdf_with_nans, truth_all, nan_mask).

    True field has clear lateral (x, y sinusoids) plus vertical (linear z)
    structure so that a joint 3D model must exploit both to fit well.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1000, n_total)
    ys = rng.uniform(0, 1000, n_total)
    zs = rng.uniform(-5000, -1000, n_total)
    truth = np.sin(xs / 250.0) + np.cos(ys / 250.0) + zs / 2500.0

    values = truth.copy()
    n_missing = int(missing_frac * n_total)
    nan_idx = rng.choice(n_total, n_missing, replace=False)
    nan_mask = np.zeros(n_total, dtype=bool)
    nan_mask[nan_idx] = True
    values[nan_mask] = np.nan

    geom = [shp.Point(x, y) for x, y in zip(xs, ys)]
    gdf = gpd.GeoDataFrame(
        {"x": xs, "y": ys, "z": zs, "value": values, "geometry": geom},
        crs="EPSG:32610",
    )
    return gdf, truth, nan_mask


@pytest.fixture(scope="module")
def synthetic_3d():
    return _make_synthetic_3d_gdf()


def test_backfill_gdf_3d_lkx_fills_all_nans(synthetic_3d):
    gdf, _truth, _nan_mask = synthetic_3d
    result = backfill_gdf_3d(
        gdf.copy(),
        value_col="value",
        verbose=False,
    )
    vals = result["value_extrapolated"].to_numpy()
    assert np.all(~np.isnan(vals)), (
        "latticekrigx 3D backend left NaNs in value_extrapolated"
    )


def test_backfill_gdf_3d_lkx_captures_lateral_and_vertical(synthetic_3d):
    """LKX predictions on missing rows should be positively correlated
    with truth in both lateral (x, y) and vertical (z) senses. We check
    that the joint residual is small relative to the total variance.
    """
    gdf, truth, nan_mask = synthetic_3d
    result = backfill_gdf_3d(
        gdf.copy(),
        value_col="value",
        verbose=False,
    )
    pred = result["value_extrapolated"].to_numpy()[nan_mask]
    y_true = truth[nan_mask]

    var_true = float(np.var(y_true))
    rmse = _rmse(y_true, pred)
    assert rmse < np.sqrt(var_true), (
        f"latticekrigx 3D RMSE {rmse} exceeds truth stddev "
        f"{np.sqrt(var_true)}; joint model not capturing structure"
    )


def test_backfill_gdf_3d_is_reproducible(synthetic_3d):
    gdf, _truth, _nan_mask = synthetic_3d
    first = backfill_gdf_3d(
        gdf.copy(),
        value_col="value",
        verbose=False,
    )
    default = backfill_gdf_3d(
        gdf.copy(),
        value_col="value",
        verbose=False,
    )
    np.testing.assert_array_equal(
        first["value_extrapolated"].to_numpy(),
        default["value_extrapolated"].to_numpy(),
    )
