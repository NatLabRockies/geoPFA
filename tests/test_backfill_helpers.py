import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pytest

from geopfa.extrapolation import (
    update_gdf_with_predictions,
    backfill_gdf_at_height,
)
from tests.fixtures.data_generators import generate_campbell2d_grid


def test_update_gdf_fills_nan_values_only():
    gdf, X, Y, Z_true, Z_obs, mask = generate_campbell2d_grid(
        nx=12, ny=12, missing_pattern="center_block"
    )

    preds = Z_obs.copy()
    preds[mask] = 10.0

    gdf2 = update_gdf_with_predictions(gdf.copy(), preds, X, Y)

    before_vals = gdf["value"].to_numpy()
    after_vals = gdf2["value_extrapolated"].to_numpy()

    assert np.allclose(
        before_vals[~np.isnan(before_vals)], after_vals[~np.isnan(before_vals)]
    )

    assert np.allclose(after_vals[np.isnan(before_vals)], 10.0)


def test_update_gdf_rounding_is_internal_only():
    """Your implementation always rounds internally, so we do not test ndigits."""
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0)],
            "value": [np.nan],
        }
    )

    X = np.array([[0]])
    Y = np.array([[0]])
    preds = np.array([[1.23456789]])

    gdf2 = update_gdf_with_predictions(gdf, preds, X, Y)
    assert np.isclose(gdf2.iloc[0]["value_extrapolated"], 1.23456789)


def test_backfill_gdf_at_height_maps_to_grid():
    gdf, X, Y, Z_true, Z_obs, mask = generate_campbell2d_grid(
        nx=10, ny=10, missing_pattern="none"
    )

    preds = Z_true

    # Insert x,y,z columns (production code expects them)
    gdf = gdf.copy()
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf["z"] = 0.0

    out = backfill_gdf_at_height(
        gdf,
        value_col="value",
        z_value=0.0,
        z_tol=1e6,
        X_grid=X,
        Y_grid=Y,
        backfilled_array=preds,
    )

    assert np.allclose(
        np.sort(out["value"]), np.sort(Z_true.ravel()), atol=1e-3
    )


def test_backfill_gdf_at_height_shape_mismatch_raises():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(1, 1), Point(2, 2)],
            "value": [np.nan, np.nan],
            "x": [1, 2],
            "y": [1, 2],
            "z": [0, 0],
        }
    )

    wrong_preds = np.ones((3, 3))
    X = np.ones((2, 2))
    Y = np.ones((2, 2))

    with pytest.raises(ValueError):
        backfill_gdf_at_height(
            gdf,
            value_col="value",
            z_value=0.0,
            z_tol=0.1,
            X_grid=X,
            Y_grid=Y,
            backfilled_array=wrong_preds,
        )
