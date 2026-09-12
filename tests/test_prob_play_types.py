"""Tests for per-feature regularization weights + play-type defaults registry."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from geopfa.prob.fitting import _fit_offset_logit, fit_component_probability
from geopfa.prob.play_types import (
    PLAY_TYPE_REGISTRY,
    available_play_types,
    play_type_defaults,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# ---------------------------------------------------------------------------
# C.1 per-feature regularization
# ---------------------------------------------------------------------------


def test_per_feature_weights_zero_recovers_unregularized_fit() -> None:
    """A zero weight per feature should not shrink that coefficient."""
    rng = np.random.default_rng(0)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    beta_true = np.array([1.5, -0.8, 0.2])
    eta = X @ beta_true
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    offset = np.zeros(n)
    # Zero weights on all features means very weak regularization
    result = _fit_offset_logit(
        X,
        y,
        offset,
        per_feature_weights=np.zeros(k),
    )
    # Coefficients should be close to truth (within sample noise)
    assert np.allclose(result.x, beta_true, atol=0.5)


def test_per_feature_weights_large_shrinks_to_prior_mean() -> None:
    """A large weight on a feature should shrink its coefficient toward its prior mean."""
    rng = np.random.default_rng(1)
    n, k = 400, 2
    X = rng.normal(size=(n, k))
    eta = X @ np.array([1.0, -1.0])
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    offset = np.zeros(n)
    # Compare: large weight on feature 1 toward prior mean 3.0 vs no weight.
    result_penalised = _fit_offset_logit(
        X,
        y,
        offset,
        per_feature_weights=np.array([1e-4, 50.0]),
        prior_means=np.array([0.0, 3.0]),
    )
    result_free = _fit_offset_logit(X, y, offset)
    # With heavy penalization toward prior_mean=3.0, feature 1's coefficient
    # should be pulled closer to 3.0 compared to the free fit (≈ -1.0).
    dist_penalised = abs(result_penalised.x[1] - 3.0)
    dist_free = abs(result_free.x[1] - 3.0)
    assert dist_penalised < dist_free


# ---------------------------------------------------------------------------
# C.2 play-type defaults registry
# ---------------------------------------------------------------------------


def test_available_play_types_lists_registered_types() -> None:
    types = available_play_types()
    assert "extensional" in types
    assert "magmatic" in types
    assert "convective" in types


def test_play_type_defaults_returns_known_signature() -> None:
    defaults = play_type_defaults(
        "extensional", layer_names=("fault_slip", "thermal_gradient", "random")
    )
    assert "per_feature_weights" in defaults
    assert "prior_means" in defaults
    # The known fault layer should pick up an entry from the registry
    assert "fault_slip" in defaults["per_feature_weights"]


def test_play_type_defaults_unknown_warns_returns_empty() -> None:
    with pytest.warns(UserWarning, match="unknown play type"):
        defaults = play_type_defaults("imaginary", layer_names=("layer_a",))
    assert defaults["per_feature_weights"] == {}
    assert defaults["prior_means"] == {}


def test_play_type_defaults_pattern_match_layer_names() -> None:
    """Pattern matching should identify layers by substring, not exact match."""
    defaults = play_type_defaults(
        "extensional",
        layer_names=("quaternary_fault_density", "fault_slip_dilation"),
    )
    # Both should be matched by 'fault' substring
    assert "quaternary_fault_density" in defaults["per_feature_weights"]
    assert "fault_slip_dilation" in defaults["per_feature_weights"]


def test_play_type_defaults_no_matching_layers_returns_empty_weights() -> None:
    defaults = play_type_defaults(
        "extensional",
        layer_names=("totally_unknown_layer", "random_evidence"),
    )
    assert defaults["per_feature_weights"] == {}


# ---------------------------------------------------------------------------
# Integration: play-type config drives the fit
# ---------------------------------------------------------------------------


def test_fit_component_probability_uses_per_feature_weights_from_config() -> (
    None
):
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=2)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_component_probability(
            comp,
            prior_probability=0.5,
            labeled_wells=fixture.wells,
            label_column="heat_label",
            per_feature_weights={"gradient:value_interpolated": 100.0},
            prior_means={"gradient:value_interpolated": 0.0},
        )
    assert "probability" in result.probability.columns
