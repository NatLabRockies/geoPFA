"""Tests for auditable evidence-coefficient prior recommendations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geopfa.prob.evidence_priors import (
    EvidencePriorGenerationConfig,
    EvidencePriorRule,
    derive_evidence_coefficient_priors,
)


def test_neutral_generator_returns_zero_centered_priors() -> None:
    evidence = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, -1.0],
        ]
    )

    recommendation = derive_evidence_coefficient_priors(
        evidence,
        ("density", "resistivity"),
        component="reservoir",
    )

    assert recommendation.feature_names == ("density", "resistivity")
    np.testing.assert_allclose(recommendation.mean_log_odds, 0.0)
    np.testing.assert_allclose(recommendation.sd_log_odds, 0.5)
    assert recommendation.status == "neutral_unlabeled"
    assert recommendation.references == {}


def test_rule_records_literature_provenance_and_overrides_neutral_mean() -> (
    None
):
    evidence = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
        ]
    )
    rule = EvidencePriorRule(
        mean_log_odds=math.log(2.0),
        sd_log_odds=0.4,
        references=("https://doi.org/10.15121/1493758",),
        rationale="Mapped fault proximity is a reservoir-permeability proxy.",
    )

    recommendation = derive_evidence_coefficient_priors(
        evidence,
        ("fault_distance_inverted", "unmatched"),
        component="reservoir",
        config=EvidencePriorGenerationConfig(
            rules={"reservoir:fault_distance_inverted": rule}
        ),
    )

    assert recommendation.mean_log_odds[0] == pytest.approx(math.log(2.0))
    assert recommendation.sd_log_odds[0] == pytest.approx(0.4)
    assert recommendation.references["fault_distance_inverted"] == (
        "https://doi.org/10.15121/1493758",
    )
    assert recommendation.rationales["fault_distance_inverted"] == (
        "Mapped fault proximity is a reservoir-permeability proxy.",
    )
    assert recommendation.mean_log_odds[1] == pytest.approx(0.0)


def test_redundant_layers_share_a_rule_effect_budget() -> None:
    evidence = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 3.0, 0.0],
        ]
    )
    rule = EvidencePriorRule(
        mean_log_odds=math.log(2.0),
        sd_log_odds=0.5,
    )

    recommendation = derive_evidence_coefficient_priors(
        evidence,
        ("gravity", "density", "independent"),
        component="reservoir",
        config=EvidencePriorGenerationConfig(
            redundancy_correlation=0.95,
            rules={"gravity": rule, "density": rule},
        ),
    )

    np.testing.assert_allclose(
        recommendation.mean_log_odds[:2], math.log(2.0) / 2.0
    )
    np.testing.assert_allclose(
        recommendation.sd_log_odds[:2], 0.5 / math.sqrt(2.0)
    )
    assert recommendation.redundancy_groups == (("density", "gravity"),)
    assert recommendation.mean_log_odds[2] == pytest.approx(0.0)
    assert recommendation.sd_log_odds[2] == pytest.approx(0.5)


def test_component_qualified_rule_precedes_global_rule() -> None:
    evidence = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    recommendation = derive_evidence_coefficient_priors(
        evidence,
        ("faults",),
        component="reservoir",
        config=EvidencePriorGenerationConfig(
            rules={
                "faults": EvidencePriorRule(
                    mean_log_odds=math.log(1.5), sd_log_odds=0.4
                ),
                "reservoir:faults": EvidencePriorRule(
                    mean_log_odds=math.log(3.0), sd_log_odds=0.3
                ),
            }
        ),
    )

    assert recommendation.mean_log_odds == pytest.approx((math.log(3.0),))
    assert recommendation.sd_log_odds == pytest.approx((0.3,))


def test_programmatic_heuristic_can_set_a_simple_component_rule() -> None:
    evidence = np.asarray([[0.0], [1.0], [2.0], [3.0]])

    def fault_rule(
        component: str, feature_name: object
    ) -> EvidencePriorRule | None:
        assert component == "reservoir"
        assert feature_name.feature_name == "faults"
        return EvidencePriorRule(mean_log_odds=math.log(2.0), sd_log_odds=0.35)

    recommendation = derive_evidence_coefficient_priors(
        evidence,
        ("faults",),
        component="reservoir",
        heuristic=fault_rule,
    )

    assert recommendation.mean_log_odds == pytest.approx((math.log(2.0),))
    assert recommendation.sd_log_odds == pytest.approx((0.35,))


@pytest.mark.parametrize(
    ("evidence", "names", "message"),
    [
        (np.asarray([[0.0], [0.0]]), ("constant",), "constant"),
        (np.asarray([[0.0], [np.nan]]), ("missing",), "finite"),
        (np.asarray([[0.0], [1.0]]), ("duplicate", "duplicate"), "duplicate"),
    ],
)
def test_generator_rejects_invalid_evidence(
    evidence: np.ndarray, names: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_evidence_coefficient_priors(
            evidence,
            names,
            component="reservoir",
        )
