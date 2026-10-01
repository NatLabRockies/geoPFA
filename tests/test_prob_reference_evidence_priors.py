"""Tests for labelled reference-domain evidence-prior learning."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.prob.reference_evidence_priors import (
    ReferenceDomainPriorConfig,
    ReferenceEvidenceDomain,
    learn_reference_domain_evidence_priors,
)


def _reference_domain(
    domain_id: str,
    *,
    seed: int,
    feature_names: tuple[str, ...] = ("fault_proximity", "resistivity"),
) -> ReferenceEvidenceDomain:
    """Return one reproducible labelled reservoir reference domain."""
    rng = np.random.default_rng(seed)
    evidence = rng.normal(size=(500, len(feature_names)))
    logit = -0.2 + 1.1 * evidence[:, 0] - 0.45 * evidence[:, 1]
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return ReferenceEvidenceDomain(
        domain_id=domain_id,
        component="reservoir",
        evidence=evidence,
        labels=labels,
        feature_names=feature_names,
        source_id=f"doi:example/{domain_id}",
        target_definition="Observed productive reservoir indicator.",
    )


def test_reference_domains_learn_component_specific_prior_rules() -> None:
    result = learn_reference_domain_evidence_priors(
        (
            _reference_domain("domain-a", seed=2),
            _reference_domain("domain-b", seed=3),
        ),
        config=ReferenceDomainPriorConfig(ridge_penalty=1.0),
    )

    assert result.component == "reservoir"
    assert result.rules["reservoir:fault_proximity"].mean_log_odds > 0.0
    assert result.rules["reservoir:resistivity"].mean_log_odds < 0.0
    assert result.rules["reservoir:fault_proximity"].sd_log_odds > 0.0
    assert result.feature_estimates["fault_proximity"].n_domains == 2
    assert result.feature_estimates["fault_proximity"].source_ids == (
        "doi:example/domain-a",
        "doi:example/domain-b",
    )


def test_reference_learner_requires_multiple_independent_domains() -> None:
    with pytest.raises(ValueError, match="at least two"):
        learn_reference_domain_evidence_priors(
            (_reference_domain("domain-a", seed=2),)
        )


def test_reference_learner_excludes_features_without_cross_domain_support() -> (
    None
):
    result = learn_reference_domain_evidence_priors(
        (
            _reference_domain("domain-a", seed=2),
            _reference_domain(
                "domain-b",
                seed=3,
                feature_names=("fault_proximity", "geochemistry"),
            ),
        )
    )

    assert tuple(result.feature_estimates) == ("fault_proximity",)
    assert "reservoir:resistivity" not in result.rules
    assert "reservoir:geochemistry" not in result.rules


def test_reference_domain_rejects_unbalanced_or_mismatched_inputs() -> None:
    domain = _reference_domain("domain-a", seed=2)
    unbalanced = ReferenceEvidenceDomain(
        domain_id="unbalanced",
        component="reservoir",
        evidence=domain.evidence,
        labels=np.ones(domain.evidence.shape[0]),
        feature_names=domain.feature_names,
        source_id="doi:example/unbalanced",
        target_definition="Observed productive reservoir indicator.",
    )
    mismatch = ReferenceEvidenceDomain(
        domain_id="insulation",
        component="insulation",
        evidence=domain.evidence,
        labels=domain.labels,
        feature_names=domain.feature_names,
        source_id="doi:example/insulation",
        target_definition="Observed insulation indicator.",
    )

    with pytest.raises(ValueError, match="both binary classes"):
        learn_reference_domain_evidence_priors(
            (unbalanced, _reference_domain("b", seed=3))
        )
    with pytest.raises(ValueError, match="same component"):
        learn_reference_domain_evidence_priors((domain, mismatch))
