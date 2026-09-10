"""Tests for backend param on Processing.extrapolate_2d / extrapolate_3d (P3-S01 / P9-S01)."""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from geopfa.processing import Processing
from tests.fixtures.campbell2d import DEFAULT_THETA
from tests.fixtures.data_generators import generate_campbell2d_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pfa_2d(data_col: str = "value_interpolated"):
    gdf, *_ = generate_campbell2d_grid(
        nx=20,
        ny=20,
        theta=DEFAULT_THETA,
        noise=0.0,
        missing_pattern="center_block",
    )
    gdf = gdf.rename(columns={"value": data_col})
    pfa = {
        "criteria": {
            "crit": {
                "components": {
                    "comp": {
                        "layers": {
                            "layer": {
                                "model": gdf.copy(),
                                "units": "degC",
                            }
                        }
                    }
                }
            }
        }
    }
    return pfa


def _make_pfa_3d(data_col: str = "value_interpolated"):
    rng = np.random.default_rng(42)
    n = 200
    xs = rng.uniform(-3, 3, n)
    ys = rng.uniform(-3, 3, n)
    zs = rng.uniform(0, 5, n)
    vals = np.sin(xs) * np.cos(ys) + 0.1 * zs
    nan_mask = rng.random(n) < 0.3
    vals[nan_mask] = np.nan

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(x, y) for x, y in zip(xs, ys)],
            "x": xs,
            "y": ys,
            "z": zs,
            data_col: vals,
        }
    )
    pfa = {
        "criteria": {
            "crit": {
                "components": {
                    "comp": {
                        "layers": {
                            "layer": {
                                "model": gdf.copy(),
                                "units": "m",
                            }
                        }
                    }
                }
            }
        }
    }
    return pfa


def _layer(pfa):
    return pfa["criteria"]["crit"]["components"]["comp"]["layers"]["layer"]


# ---------------------------------------------------------------------------
# extrapolate_2d
# ---------------------------------------------------------------------------


def test_extrapolate_2d_lkx_fills_all_nans():
    pfa = _make_pfa_2d()
    result = Processing.extrapolate_2d(
        pfa,
        "crit",
        "comp",
        "layer",
        backend="latticekrigx",
        verbose=False,
    )
    layer = _layer(result)
    vals = layer["model"]["value_extrapolated"].to_numpy()
    assert np.all(~np.isnan(vals)), "latticekrigx left NaNs in value_extrapolated"


def test_extrapolate_2d_lkx_pfa_metadata():
    pfa = _make_pfa_2d()
    result = Processing.extrapolate_2d(
        pfa, "crit", "comp", "layer", backend="latticekrigx", verbose=False
    )
    layer = _layer(result)
    assert layer["model_data_col"] == "value_extrapolated"
    assert layer["model_units"] == "degC"


def test_extrapolate_2d_default_backend_is_latticekrigx():
    pfa_default = _make_pfa_2d()
    pfa_explicit = _make_pfa_2d()

    result_default = Processing.extrapolate_2d(
        pfa_default, "crit", "comp", "layer", verbose=False
    )
    result_explicit = Processing.extrapolate_2d(
        pfa_explicit, "crit", "comp", "layer", backend="latticekrigx", verbose=False
    )

    np.testing.assert_array_equal(
        _layer(result_default)["model"]["value_extrapolated"].to_numpy(),
        _layer(result_explicit)["model"]["value_extrapolated"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# extrapolate_3d
# ---------------------------------------------------------------------------


def test_extrapolate_3d_lkx_fills_all_nans():
    pfa = _make_pfa_3d()
    result = Processing.extrapolate_3d(
        pfa,
        "crit",
        "comp",
        "layer",
        backend="latticekrigx",
        verbose=False,
    )
    layer = _layer(result)
    vals = layer["model"]["value_extrapolated"].to_numpy()
    assert np.all(~np.isnan(vals)), (
        "latticekrigx 3d left NaNs in value_extrapolated"
    )


def test_extrapolate_3d_lkx_pfa_metadata():
    pfa = _make_pfa_3d()
    result = Processing.extrapolate_3d(
        pfa, "crit", "comp", "layer", backend="latticekrigx", verbose=False
    )
    layer = _layer(result)
    assert layer["model_data_col"] == "value_extrapolated"
    assert layer["model_units"] == "m"


def test_extrapolate_3d_default_backend_is_latticekrigx():
    pfa_default = _make_pfa_3d()
    pfa_explicit = _make_pfa_3d()

    result_default = Processing.extrapolate_3d(
        pfa_default, "crit", "comp", "layer", verbose=False
    )
    result_explicit = Processing.extrapolate_3d(
        pfa_explicit, "crit", "comp", "layer", backend="latticekrigx", verbose=False
    )

    np.testing.assert_array_equal(
        _layer(result_default)["model"]["value_extrapolated"].to_numpy(),
        _layer(result_explicit)["model"]["value_extrapolated"].to_numpy(),
    )
