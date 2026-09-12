"""Tests for :mod:`geopfa.prob.variogram` — empirical variogram and range estimation."""

from __future__ import annotations

import numpy as np
import pytest

import geopfa.prob.variogram as variogram_module
from geopfa.prob.variogram import (
    empirical_variogram,
    estimate_variogram_range,
    recommend_block_size_km,
)

_SEED = 0


def _grid_coords(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(_SEED)
    return rng.uniform(0.0, 100.0, size=(n, 2))


def _smooth_values(coords: np.ndarray) -> np.ndarray:
    """Linear spatial trend — variogram should increase monotonically."""
    return coords[:, 0] * 0.01 + np.random.default_rng(_SEED).normal(
        0, 0.1, len(coords)
    )


def test_empirical_variogram_returns_midpoints_and_gamma() -> None:
    coords = _grid_coords(40)
    lags, gamma = empirical_variogram(coords, _smooth_values(coords))
    assert lags.shape == gamma.shape
    assert len(lags) > 0
    assert np.all(lags > 0)


def test_empirical_variogram_downsamples_large_inputs() -> None:
    """More than 300 points triggers the sub-sampling path."""
    rng = np.random.default_rng(_SEED)
    coords = rng.uniform(0.0, 100.0, size=(500, 2))
    values = rng.standard_normal(500)
    lags, gamma = empirical_variogram(coords, values)
    assert lags.shape == gamma.shape
    assert len(lags) > 0


def test_empirical_variogram_too_few_points_raises() -> None:
    coords = np.array([[0.0, 0.0]])
    values = np.array([1.0])
    with pytest.raises(ValueError):
        empirical_variogram(coords, values)


def test_empirical_variogram_empty_bin_filled_with_nan() -> None:
    """Very small input where some bins may have no pairs."""
    rng = np.random.default_rng(_SEED)
    coords = rng.uniform(0.0, 1.0, size=(6, 2))
    values = rng.standard_normal(6)
    lags, gamma = empirical_variogram(coords, values, n_bins=20)
    # Some bins may be NaN — that's expected and valid
    assert np.isfinite(lags).all()
    assert len(gamma) == 20


def test_estimate_variogram_range_returns_positive_float() -> None:
    coords = _grid_coords(60)
    values = _smooth_values(coords)
    r = estimate_variogram_range(coords, values)
    assert isinstance(r, float)
    assert r > 0


def test_estimate_variogram_range_rejects_too_few_points() -> None:
    """Automatic CV geometry must fail closed when range is unidentified."""
    coords = np.array([[0.0, 0.0], [100.0, 0.0]])
    values = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="at least 4"):
        estimate_variogram_range(coords, values)


def test_estimate_variogram_range_all_nan_gamma_fallback() -> None:
    """Constant values → zero variance → all NaN gamma bins → fallback to last lag."""
    rng = np.random.default_rng(_SEED)
    coords = rng.uniform(0.0, 100.0, size=(20, 2))
    values = np.ones(20)
    r = estimate_variogram_range(coords, values)
    assert r > 0


def test_recommend_block_size_km_returns_positive() -> None:
    rng = np.random.default_rng(_SEED)
    coords = rng.uniform(0.0, 50.0, size=(80, 2))
    values = rng.standard_normal(80)
    block_km = recommend_block_size_km(coords, values)
    assert block_km > 0


def test_recommend_block_size_km_clamps_within_bounds() -> None:
    """Result should be between min_km and max_km."""
    rng = np.random.default_rng(_SEED)
    coords = rng.uniform(0.0, 50.0, size=(80, 2))
    values = rng.standard_normal(80)
    block_km = recommend_block_size_km(coords, values, min_km=5.0, max_km=50.0)
    assert 5.0 <= block_km <= 50.0


def test_recommend_block_size_propagates_variogram_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> float:
        raise RuntimeError("forced variogram failure")

    monkeypatch.setattr(variogram_module, "estimate_variogram_range", fail)
    with pytest.raises(RuntimeError, match="forced variogram failure"):
        recommend_block_size_km(_grid_coords(20), np.arange(20.0))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"multiplier": 0.0}, ValueError, "multiplier"),
        ({"min_km": 20.0, "max_km": 10.0}, ValueError, "min_km"),
        ({"assumed_crs_unit_m": "yes"}, TypeError, "assumed_crs_unit_m"),
    ],
)
def test_recommend_block_size_rejects_invalid_policy(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        recommend_block_size_km(_grid_coords(20), np.arange(20.0), **kwargs)
