"""Tests for backfill_gdf 2D backend routing (P2-S02 / P9-S01).

Verifies that ``backend="latticekrigx"`` fills all NaNs in the 2D pipeline
and that predictions are finite and within a reasonable RMSE.
"""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.extrapolation import backfill_gdf
from tests.fixtures.campbell2d import DEFAULT_THETA
from tests.fixtures.data_generators import generate_campbell2d_grid


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


@pytest.fixture(scope="module")
def campbell_gdf():
    gdf, _X, _Y, _Zt, _Zo, _nan = generate_campbell2d_grid(
        nx=20,
        ny=20,
        theta=DEFAULT_THETA,
        noise=0.0,
        missing_pattern="center_block",
    )
    return gdf.copy()


def _run(gdf, backend: str = "latticekrigx"):
    return backfill_gdf(
        gdf.copy(),
        value_col="value",
        z_value=None,
        test_size=0.20,
        seed=123,
        verbose=False,
        backend=backend,
    )


def test_backfill_gdf_lkx_fills_all_nans(campbell_gdf):
    filled = _run(campbell_gdf)
    vals = filled["value_extrapolated"].to_numpy()
    assert np.all(~np.isnan(vals)), (
        "latticekrigx backend left NaNs in value_extrapolated"
    )


def test_backfill_gdf_lkx_rmse_within_tolerance(campbell_gdf):
    truth = campbell_gdf["value"].to_numpy()
    known_mask = ~np.isnan(truth)

    filled_lkx = _run(campbell_gdf)
    rmse_lkx = _rmse(truth[known_mask], filled_lkx["value_extrapolated"].to_numpy()[known_mask])

    assert rmse_lkx < 0.30, f"latticekrigx RMSE too high: {rmse_lkx}"


def test_backfill_gdf_default_backend_is_latticekrigx(campbell_gdf):
    explicit = _run(campbell_gdf, backend="latticekrigx")
    default = backfill_gdf(
        campbell_gdf.copy(),
        value_col="value",
        z_value=None,
        test_size=0.20,
        seed=123,
        verbose=False,
    )
    np.testing.assert_array_equal(
        explicit["value_extrapolated"].to_numpy(),
        default["value_extrapolated"].to_numpy(),
    )
