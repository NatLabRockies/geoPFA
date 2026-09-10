"""Tests for variogram estimation, PU correction, and multi-region helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from geopfa.prob.variogram import (
    empirical_variogram,
    estimate_variogram_range,
    recommend_block_size_km,
)
from geopfa.prob.pu import NNPULogitResult, fit_nnpu_logistic
from geopfa.prob.inference import ComponentFitResult
from geopfa.prob.inference.sequential import SequentialFitter
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------


def test_empirical_variogram_returns_positive_gamma() -> None:
    rng = np.random.default_rng(0)
    n = 80
    coords = rng.uniform(0.0, 100_000.0, size=(n, 2))
    values = rng.normal(0.0, 1.0, size=n)
    lags, gamma = empirical_variogram(coords, values)
    assert lags.shape == gamma.shape
    assert np.all(lags >= 0)
    assert np.all(np.isfinite(gamma[np.isfinite(gamma)]))


def test_estimate_variogram_range_returns_positive() -> None:
    rng = np.random.default_rng(1)
    coords = rng.uniform(0.0, 100_000.0, size=(50, 2))
    values = rng.normal(0.0, 1.0, size=50)
    r = estimate_variogram_range(coords, values)
    assert r > 0


def test_recommend_block_size_km_is_within_bounds() -> None:
    rng = np.random.default_rng(2)
    coords = rng.uniform(0.0, 100_000.0, size=(60, 2))  # 100 km span in metres
    values = rng.normal(0.0, 1.0, size=60)
    rec = recommend_block_size_km(coords, values, min_km=5.0, max_km=150.0)
    assert 5.0 <= rec <= 150.0


def test_empirical_variogram_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="4"):
        empirical_variogram(np.zeros((3, 2)), np.zeros(3))


# ---------------------------------------------------------------------------
# PU correction
# ---------------------------------------------------------------------------


def test_nnpu_returns_nonnegative_empirical_risk() -> None:
    x = np.column_stack([np.ones(12), np.linspace(-2.0, 2.0, 12)])
    observed_positive = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    result = fit_nnpu_logistic(
        x,
        observed_positive,
        class_prior=0.45,
        penalty_weights=np.array([0.0, 1e-3]),
    )
    assert isinstance(result, NNPULogitResult)
    assert result.negative_risk >= 0.0
    assert result.success


def test_nnpu_rejects_missing_class_prior() -> None:
    with pytest.raises(ValueError, match="class_prior"):
        fit_nnpu_logistic(
            np.ones((4, 1)),
            np.array([1, 0, 1, 0]),
            class_prior=None,
        )


# ---------------------------------------------------------------------------
# SequentialFitter protocol
# ---------------------------------------------------------------------------


def test_sequential_fitter_returns_component_fit_result() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=0)
    from geopfa.prob.alpha import build_alpha_c
    from geopfa.prob.config import (
        AlphaModeConfig,
        EvidenceConfig,
        SpatialFieldConfig,
    )

    comp_data = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]
    alpha_result = build_alpha_c(
        comp_data,
        AlphaModeConfig(mode="layer_logit", layer="prior_layer_a"),
        grid_gdf=comp_data["pr_norm"],
    )
    fitter = SequentialFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fitter.fit(
            comp_data,
            alpha_result,
            EvidenceConfig(),
            fixture.wells,
            "heat_label",
            SpatialFieldConfig(enabled=False),
        )
    assert isinstance(result, ComponentFitResult)
    probs = result.probability["probability"].to_numpy()
    assert (probs >= 0).all() and (probs <= 1).all()
    assert result.n_train == len(fixture.wells)
    assert result.diagnostics["n_train"] == len(fixture.wells)
    assert "mean" in result.beta_posterior
    assert result.diagnostics.get("backend") == "sequential"
