"""Tests for handing generated evidence priors to existing geoPFA config."""

from __future__ import annotations

import math

import numpy as np

from geopfa.prob.evidence_priors import (
    EvidencePriorGenerationConfig,
    EvidencePriorRule,
    derive_evidence_coefficient_priors,
)


def test_recommendation_exports_component_qualified_regularization_maps() -> (
    None
):
    recommendation = derive_evidence_coefficient_priors(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        ("fault_distance_inverted",),
        component="reservoir",
        config=EvidencePriorGenerationConfig(
            rules={
                "reservoir:fault_distance_inverted": EvidencePriorRule(
                    mean_log_odds=math.log(2.0),
                    sd_log_odds=0.25,
                )
            }
        ),
    )

    assert recommendation.regularization_values() == {
        "prior_means": {"reservoir:fault_distance_inverted": math.log(2.0)},
        "prior_precisions": {"reservoir:fault_distance_inverted": 16.0},
    }
