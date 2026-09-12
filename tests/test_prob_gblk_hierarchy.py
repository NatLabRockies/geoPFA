"""Tests for regional pooling mapping via ``pool_regional_coefficients``."""

from __future__ import annotations

import numpy as np
import pytest
from latticekrigx.glk.hierarchy import EBPoolResult

from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.gblk_assemble import pool_regional_coefficients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rich_var(n: int = 1) -> np.ndarray:
    """Very small sampling variance — data-rich scenario."""
    return np.full(n, 1e-8)


def _poor_var(n: int = 1) -> np.ndarray:
    """Very large sampling variance — data-poor scenario."""
    return np.full(n, 1e6)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_eb_pool_result():
    d_hat = np.array([1.0, 2.0, 3.0])
    d_var = np.array([0.1, 0.1, 0.1])
    play_types = ["extensional", "extensional", "magmatic"]
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    assert isinstance(result, EBPoolResult)


# ---------------------------------------------------------------------------
# Acceptance: data-rich regions recover near-unpooled coefficients
# ---------------------------------------------------------------------------


def test_data_rich_regions_recover_unpooled():
    rng = np.random.default_rng(0)
    R = 6
    d_hat = rng.standard_normal(R)
    d_var = np.full(R, 1e-10)
    play_types = ["extensional"] * 3 + ["magmatic"] * 3
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    pooled = result.d_pooled.squeeze(-1)
    np.testing.assert_allclose(pooled, d_hat, atol=1e-4)


def test_data_rich_shrinkage_near_zero():
    R = 4
    d_hat = np.array([1.0, 2.0, 5.0, 6.0])
    d_var = np.full(R, 1e-12)
    play_types = ["convective"] * 4
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    shrinkage = result.shrinkage.squeeze(-1)
    assert np.all(shrinkage < 0.01), (
        f"expected shrinkage near 0, got {shrinkage}"
    )


# ---------------------------------------------------------------------------
# Acceptance: data-poor regions shrink toward the play-type mean
# ---------------------------------------------------------------------------


def test_data_poor_regions_shrink_to_group_mean():
    d_hat = np.array([0.0, 10.0, 0.0, 10.0])
    d_var = np.array([1e-8, 1e-8, 1e6, 1e6])
    play_types = ["extensional", "extensional", "magmatic", "magmatic"]
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    pooled = result.d_pooled.squeeze(-1)

    group_labels = result.group_labels.tolist()
    idx_ext = group_labels.index("extensional")
    idx_mag = group_labels.index("magmatic")
    mu_ext = float(result.group_mean[idx_ext].squeeze())
    mu_mag = float(result.group_mean[idx_mag].squeeze())

    # Regions 2 and 3 are data-poor — they should be near the group mean.
    assert abs(pooled[2] - mu_mag) < abs(d_hat[2] - mu_mag) + 1e-6
    assert abs(pooled[3] - mu_mag) < abs(d_hat[3] - mu_mag) + 1e-6
    # Regions 0 and 1 are data-rich — they stay near d_hat.
    np.testing.assert_allclose(pooled[:2], d_hat[:2], atol=1e-3)


def test_data_poor_shrinkage_near_one():
    R = 3
    d_hat = np.array([1.0, 2.0, 3.0])
    d_var = np.full(R, 1e8)
    play_types = ["extensional"] * R
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    shrinkage = result.shrinkage.squeeze(-1)
    assert np.all(shrinkage > 0.99), (
        f"expected shrinkage near 1, got {shrinkage}"
    )


# ---------------------------------------------------------------------------
# Acceptance: single region reduces to a fixed coefficient vector
# ---------------------------------------------------------------------------


def test_single_region_returns_d_hat():
    d_hat = np.array([5.0])
    d_var = np.array([0.5])
    result = pool_regional_coefficients(d_hat, d_var, ["extensional"])
    pooled = float(result.d_pooled.squeeze())
    assert abs(pooled - d_hat[0]) < 1e-9, (
        f"single-region pooled {pooled} should equal d_hat {d_hat[0]}"
    )


def test_single_region_multivariate_returns_d_hat():
    d_hat = np.array([[1.0, 2.0, 3.0]])
    d_var = np.array([[0.1, 0.2, 0.3]])
    result = pool_regional_coefficients(d_hat, d_var, ["magmatic"])
    np.testing.assert_allclose(result.d_pooled, d_hat, atol=1e-9)


# ---------------------------------------------------------------------------
# Multi-dimensional coefficients (R, p)
# ---------------------------------------------------------------------------


def test_multivariate_coefficients():
    rng = np.random.default_rng(42)
    R, p = 5, 3
    d_hat = rng.standard_normal((R, p))
    d_var = np.full((R, p), 0.1)
    play_types = [
        "extensional",
        "extensional",
        "magmatic",
        "magmatic",
        "convective",
    ]
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    assert result.d_pooled.shape == (R, p)
    assert result.group_mean.shape[1] == p


# ---------------------------------------------------------------------------
# Multiple play types become separate groups
# ---------------------------------------------------------------------------


def test_play_type_groups_are_distinct():
    d_hat = np.array([1.0, 2.0, 10.0, 11.0])
    d_var = np.full(4, 0.01)
    play_types = ["extensional", "extensional", "magmatic", "magmatic"]
    result = pool_regional_coefficients(d_hat, d_var, play_types)
    assert set(result.group_labels.tolist()) == {"extensional", "magmatic"}
    assert result.group_mean.shape[0] == 2


def test_single_play_type_produces_one_group():
    d_hat = np.array([1.0, 2.0, 3.0])
    d_var = np.array([0.1, 0.2, 0.3])
    result = pool_regional_coefficients(d_hat, d_var, ["convective"] * 3)
    assert result.group_mean.shape[0] == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_length_mismatch_raises():
    d_hat = np.array([1.0, 2.0])
    d_var = np.array([0.1, 0.1])
    with pytest.raises(GEOPFAValueError, match="play_type_per_region length"):
        pool_regional_coefficients(d_hat, d_var, ["extensional"])


def test_negative_variance_raises():
    d_hat = np.array([1.0, 2.0])
    d_var = np.array([0.1, -0.1])
    with pytest.raises(GEOPFAValueError, match="non-negative"):
        pool_regional_coefficients(d_hat, d_var, ["extensional", "magmatic"])
