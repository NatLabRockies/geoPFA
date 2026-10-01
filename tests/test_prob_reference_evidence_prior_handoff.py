"""Tests the handoff from labelled-reference learning to target priors."""

from __future__ import annotations

import numpy as np

from geopfa.prob.evidence_priors import derive_evidence_coefficient_priors
from geopfa.prob.reference_evidence_priors import (
    ReferenceEvidenceDomain,
    learn_reference_domain_evidence_priors,
)


def _domain(domain_id: str, seed: int) -> ReferenceEvidenceDomain:
    """Create one independent labelled reservoir reference domain."""
    rng = np.random.default_rng(seed)
    evidence = rng.normal(size=(400, 1))
    labels = rng.binomial(
        1,
        1.0 / (1.0 + np.exp(-0.8 * evidence[:, 0])),
    )
    return ReferenceEvidenceDomain(
        domain_id=domain_id,
        component="reservoir",
        evidence=evidence,
        labels=labels,
        feature_names=("fault_proximity_inverted",),
        source_id=f"doi:example/{domain_id}",
        target_definition="Observed productive-reservoir indicator.",
    )


def test_learned_rule_and_neutral_target_only_layer_share_one_handoff() -> (
    None
):
    learned = learn_reference_domain_evidence_priors(
        (_domain("reference-a", 11), _domain("reference-b", 12))
    )

    target = derive_evidence_coefficient_priors(
        np.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0]]),
        ("fault_proximity_inverted", "target_only_geochemistry"),
        component="reservoir",
        config=learned.generation_config(),
    )

    assert target.mean_log_odds[0] > 0.0
    assert target.mean_log_odds[1] == 0.0
    assert target.status == "rule_informed_unlabeled"
    assert target.references["fault_proximity_inverted"] == (
        "doi:example/reference-a",
        "doi:example/reference-b",
    )
