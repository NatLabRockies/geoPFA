import numpy as np
import pytest

from geopfa.extrapolation import backfill_gdf
from tests.fixtures.data_generators import generate_campbell2d_grid
from tests.fixtures.campbell2d import (
    DEFAULT_THETA,
    NEG_THETA,
    POS_THETA,
)


# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------

THETA_SCENARIOS = {
    "default": DEFAULT_THETA,
    "negative": NEG_THETA,
    "positive": POS_THETA,
}

MISSING_PATTERNS = [
    "none",
    "center_block",
    "border_missing",
]

NOISE_LEVELS = [
    0.0,
    0.05,
]


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def r2(a, b):
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------
# Main end-to-end pipeline test
# ---------------------------------------------------------------------

@pytest.mark.parametrize("theta_name", THETA_SCENARIOS.keys())
@pytest.mark.parametrize("missing_pattern", MISSING_PATTERNS)
@pytest.mark.parametrize("noise", NOISE_LEVELS)
def test_full_pipeline(theta_name, missing_pattern, noise):
    """
    TRUE end-to-end test of the production extrapolation pipeline.
    This calls only backfill_gdf() and compares final outputs against
    known truths from the synthetic Campbell2D generator.

    The goal is NOT perfect accuracy — only that the GP is consistent and
    the pipeline fills missing values and preserves baseline structure.
    """

    theta = THETA_SCENARIOS[theta_name]

    # ------------------------------------------------------------------
    # Create synthetic 2D data with missingness
    # ------------------------------------------------------------------
    gdf, X_grid, Y_grid, Z_true, Z_obs, nan_mask = generate_campbell2d_grid(
        nx=20,
        ny=20,
        theta=theta,
        noise=noise,
        missing_pattern=missing_pattern,
    )

    # Copy required value column to canonical name
    gdf = gdf.copy()
    gdf["value"] = gdf["value"]  # required by backfill_gdf signature

    # ------------------------------------------------------------------
    # Run production GP extrapolation pipeline
    # ------------------------------------------------------------------
    gdf_filled = backfill_gdf(
        gdf.copy(),
        value_col="value",
        z_value=None,      # 2D mode (no Z-slicing)
        test_size=0.20,
        seed=123,
        verbose=False,
    )

    # Extract fields
    filled_values = gdf_filled["value_extrapolated"].to_numpy()
    true_values = gdf["value"].to_numpy()

    # Mask for known values (non-missing)
    known_mask = ~np.isnan(true_values)
    missing_mask = ~known_mask

    # Predictions at known locations
    y_pred = filled_values[known_mask]
    y_true = true_values[known_mask]

    # ------------------------------------------------------------------
    # Quality constraints — moderate tolerances for fast-mode GP
    # ------------------------------------------------------------------
    # Realistic tolerances: GP with sparse points on noisy Campbell2D
    assert rmse(y_true, y_pred) < 0.25, "RMSE too high"
    assert r2(y_true, y_pred) > 0.85, "R² too low"

    # Ensure missing values were actually filled
    assert np.all(~np.isnan(filled_values[missing_mask])), \
        "Missing values were not filled by backfill_gdf()"

