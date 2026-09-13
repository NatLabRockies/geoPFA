"""Tests for explicit per-feature regularization parameters."""

from __future__ import annotations

import warnings

import numpy as np

from geopfa.prob.fitting import _fit_offset_logit, fit_component_probability
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def test_per_feature_weights_zero_recovers_unregularized_fit() -> None:
    """A zero weight per feature should not shrink that coefficient."""
    rng = np.random.default_rng(0)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    beta_true = np.array([1.5, -0.8, 0.2])
    eta = X @ beta_true
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    result = _fit_offset_logit(
        X,
        y,
        np.zeros(n),
        per_feature_weights=np.zeros(k),
    )
    assert np.allclose(result.x, beta_true, atol=0.5)


def test_per_feature_weights_large_shrinks_to_prior_mean() -> None:
    """A large weight should shrink its coefficient toward its prior mean."""
    rng = np.random.default_rng(1)
    n, k = 400, 2
    X = rng.normal(size=(n, k))
    eta = X @ np.array([1.0, -1.0])
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    result_penalized = _fit_offset_logit(
        X,
        y,
        np.zeros(n),
        per_feature_weights=np.array([1e-4, 50.0]),
        prior_means=np.array([0.0, 3.0]),
    )
    result_free = _fit_offset_logit(X, y, np.zeros(n))
    assert abs(result_penalized.x[1] - 3.0) < abs(result_free.x[1] - 3.0)


def test_fit_component_probability_uses_explicit_regularization() -> None:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=2)
    component = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_component_probability(
            component,
            prior_probability=0.5,
            labeled_wells=fixture.wells,
            label_column="heat_label",
            per_feature_weights={"gradient:value_interpolated": 100.0},
            prior_means={"gradient:value_interpolated": 0.0},
        )
    assert "probability" in result.probability.columns
