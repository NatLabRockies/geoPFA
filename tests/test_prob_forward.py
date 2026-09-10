"""Contract tests for frozen deterministic GBLK forward models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geopfa.prob.forward import (
    FrozenGBLKForwardState,
    evaluate_frozen_gblk,
    load_frozen_gblk_forward_state,
    save_frozen_gblk_forward_state,
)


def _state() -> FrozenGBLKForwardState:
    prior = np.array([[-0.4, 0.2], [0.1, 0.2], [0.6, 0.2]])
    evidence = np.array([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.4]])
    spatial = np.array([[0.1, 0.0], [-0.2, 0.0], [0.3, 0.0]])
    eta = prior + spatial
    eta[:, 0] += evidence.sum(axis=1)
    probability = 1.0 / (1.0 + np.exp(-eta))
    return FrozenGBLKForwardState(
        coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        coordinate_names=("x", "y"),
        coordinate_units="metres",
        component_names=("heat", "reservoir"),
        prior_only_components=("reservoir",),
        prior_logit=prior,
        evidence_logit_contribution=evidence,
        evidence_term_names=("heat:temperature", "heat:resistivity"),
        evidence_term_component=np.array([0, 0]),
        spatial_logit=spatial,
        baseline_component_probability=probability,
        structural_scenario="axis_range",
    )


def test_frozen_forward_baseline_round_trips_and_combines_components() -> None:
    state = _state()

    result = evaluate_frozen_gblk(state)

    assert result.component_probability.shape == (1, 3, 2)
    np.testing.assert_allclose(
        result.component_probability[0],
        state.baseline_component_probability,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.combined_probability[0],
        np.prod(state.baseline_component_probability, axis=1),
        rtol=0.0,
        atol=1e-14,
    )


def test_frozen_forward_accepts_vectorized_independent_factor_blocks() -> None:
    state = _state()

    result = evaluate_frozen_gblk(
        state,
        evidence_multipliers=np.array([[1.0, 1.0], [0.0, 0.0]]),
        spatial_multipliers=np.array([[1.0, 1.0], [0.0, 1.0]]),
        component_logit_shift=np.array([[0.0, 0.0], [0.5, -0.5]]),
    )

    assert result.component_probability.shape == (2, 3, 2)
    np.testing.assert_allclose(
        result.component_probability[0], state.baseline_component_probability
    )
    expected_eta = state.prior_logit + np.array([0.5, -0.5])
    np.testing.assert_allclose(
        result.component_probability[1],
        1.0 / (1.0 + np.exp(-expected_eta)),
    )


def test_frozen_forward_accepts_cellwise_spatial_innovations() -> None:
    state = _state()
    innovation = np.zeros((2, 3, 2))
    innovation[1, :, 0] = np.array([-0.3, 0.0, 0.3])

    result = evaluate_frozen_gblk(state, spatial_logit_delta=innovation)

    baseline_logit = np.log(state.baseline_component_probability) - np.log1p(
        -state.baseline_component_probability
    )
    expected = 1.0 / (1.0 + np.exp(-(baseline_logit + innovation[1])))
    np.testing.assert_allclose(result.component_probability[1], expected)


def test_frozen_forward_archive_round_trips_exactly(tmp_path: Path) -> None:
    state = _state()
    path = tmp_path / "forward_state.npz"

    save_frozen_gblk_forward_state(path, state)
    restored = load_frozen_gblk_forward_state(path)

    assert restored.component_names == state.component_names
    assert restored.prior_only_components == state.prior_only_components
    assert restored.evidence_term_names == state.evidence_term_names
    assert restored.structural_scenario == state.structural_scenario
    np.testing.assert_array_equal(restored.coordinates, state.coordinates)
    np.testing.assert_array_equal(restored.prior_logit, state.prior_logit)
    np.testing.assert_array_equal(
        restored.evidence_logit_contribution,
        state.evidence_logit_contribution,
    )
    np.testing.assert_array_equal(restored.spatial_logit, state.spatial_logit)


def test_frozen_forward_rejects_invalid_prior_only_contributions() -> None:
    state = _state()

    with pytest.raises(ValueError, match="prior-only.*spatial"):
        FrozenGBLKForwardState(
            **{
                **state.__dict__,
                "spatial_logit": np.array(
                    [[0.1, 0.1], [-0.2, 0.0], [0.3, 0.0]]
                ),
            }
        )


def test_frozen_forward_rejects_incompatible_evaluation_counts() -> None:
    state = _state()

    with pytest.raises(ValueError, match="evaluation count"):
        evaluate_frozen_gblk(
            state,
            evidence_multipliers=np.ones((2, 2)),
            spatial_multipliers=np.ones((3, 2)),
        )
