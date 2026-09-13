"""Tests for componentwise blocked predictive-stacking weights."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.prob.predictive_stacking import (
    apply_predictive_stacking,
    select_predictive_density_stacking_weight,
    select_predictive_stacking_weight,
)


def test_stacking_selects_prior_when_update_is_harmful() -> None:
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])
    prior = np.array([0.1, 0.2, 0.8, 0.9])
    full = 1.0 - prior

    result = select_predictive_stacking_weight(outcomes, prior, full)

    assert result.weight == pytest.approx(0.0)
    assert result.selected_log_score == pytest.approx(result.prior_log_score)
    assert result.full_log_score > result.prior_log_score


def test_stacking_selects_update_when_it_improves_proper_score() -> None:
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])
    prior = np.full(4, 0.5)
    full = np.array([0.1, 0.2, 0.8, 0.9])

    result = select_predictive_stacking_weight(outcomes, prior, full)

    assert result.weight == pytest.approx(1.0)
    assert result.status == "estimated"
    assert result.selected_log_score == pytest.approx(result.full_log_score)
    assert result.full_log_score < result.prior_log_score


def test_stacking_can_select_interior_mixture() -> None:
    outcomes = np.array([0.0, 1.0])
    prior = np.array([0.1, 0.1])
    full = np.array([0.9, 0.9])

    result = select_predictive_stacking_weight(outcomes, prior, full)

    assert result.weight == pytest.approx(0.5, abs=1e-5)


def test_density_stacking_uses_continuous_predictive_log_score() -> None:
    prior_log_density = np.log(np.array([0.8, 0.7, 0.9]))
    full_log_density = np.log(np.array([0.2, 0.3, 0.1]))

    result = select_predictive_density_stacking_weight(
        prior_log_density,
        full_log_density,
    )

    assert result.weight == pytest.approx(0.0)
    assert result.selected_log_score == pytest.approx(result.prior_log_score)
    assert result.full_log_score > result.prior_log_score


def test_apply_stacking_mixes_each_draw_with_prior_probability() -> None:
    prior = np.array([0.2, 0.8])
    full_draws = np.array([[0.0, 1.0], [1.0, 0.0]])

    mixed = apply_predictive_stacking(prior, full_draws, weight=0.25)

    np.testing.assert_allclose(
        mixed,
        0.75 * prior[np.newaxis, :] + 0.25 * full_draws,
    )


@pytest.mark.parametrize("weight", [-0.01, 1.01, np.nan])
def test_apply_stacking_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        apply_predictive_stacking(
            np.array([0.5]), np.array([[0.5]]), weight=weight
        )
