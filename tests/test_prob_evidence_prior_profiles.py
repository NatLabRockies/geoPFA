"""Tests for portable evidence-prior profile loading."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from geopfa.prob.evidence_prior_profiles import (
    evidence_prior_generation_config_from_dict,
    evidence_prior_profile_from_dict,
    load_evidence_prior_profile,
)
from geopfa.prob.evidence_priors import derive_evidence_coefficient_priors


def _profile() -> dict[str, object]:
    return {
        "default_mean_log_odds": 0.0,
        "default_sd_log_odds": 0.5,
        "redundancy_correlation": 0.7,
        "rules": {
            "reservoir:fault_distance_inverted": {
                "mean_log_odds": math.log(2.0),
                "sd_log_odds": 0.4,
                "references": ["https://doi.org/10.15121/1493758"],
                "rationale": "Declared structural-permeability prior.",
            }
        },
    }


def test_profile_parser_retains_rule_provenance() -> None:
    config = evidence_prior_generation_config_from_dict(_profile())

    assert config.redundancy_correlation == pytest.approx(0.7)
    rule = config.rules["reservoir:fault_distance_inverted"]
    assert rule.mean_log_odds == pytest.approx(math.log(2.0))
    assert rule.references == ("https://doi.org/10.15121/1493758",)


def test_profile_loader_reads_json_file(tmp_path: Path) -> None:
    path = tmp_path / "evidence_priors.json"
    path.write_text(json.dumps(_profile()))

    config = load_evidence_prior_profile(path)

    assert config.default_sd_log_odds == pytest.approx(0.5)
    assert "reservoir:fault_distance_inverted" in config.rules


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda profile: profile.update({"unknown": 1}),
            "unknown evidence-prior profile key",
        ),
        (
            lambda profile: profile["rules"].update(
                {"reservoir:faults": {"mean_log_odds": 0.1}}
            ),
            "requires.*sd_log_odds",
        ),
        (
            lambda profile: profile.update({"default_sd_log_odds": 0.0}),
            "strictly positive",
        ),
    ],
)
def test_profile_parser_rejects_ambiguous_or_invalid_input(
    mutate: object, message: str
) -> None:
    profile = _profile()
    mutate(profile)

    with pytest.raises(ValueError, match=message):
        evidence_prior_generation_config_from_dict(profile)


def test_study_profile_resolves_cited_and_fixed_rules() -> None:
    """The explicit profile pathway preserves cited and fixed assignments."""
    profile = evidence_prior_profile_from_dict(
        {
            "profile_id": "documented-example-v1",
            "default": {"mean_log_odds": 0.0, "sd_log_odds": 0.5},
            "components": {
                "reservoir": {
                    "ring_faults": {
                        "mean_log_odds": 0.4,
                        "sd_log_odds": 0.6,
                        "references": ["https://doi.org/10.3133/pp1578"],
                        "rationale": "Example directional rule.",
                    }
                },
                "insulation": {
                    "temperature_model_500m": {
                        "fixed_value": 0.0,
                        "rationale": "Example exclusion.",
                    }
                },
            },
        }
    )

    resolved = derive_evidence_coefficient_priors(
        profile,
        {
            "reservoir": ("ring_faults",),
            "insulation": ("temperature_model_500m",),
        },
    )

    assert resolved.prior_means == {"reservoir:ring_faults": 0.4}
    assert resolved.fixed_coefficients == {
        "insulation:temperature_model_500m": 0.0
    }


@pytest.mark.parametrize(
    "rule", [{"fixed_value": 0.0, "sd_log_odds": 0.5}, {"sd_log_odds": 0.5}]
)
def test_study_profile_rejects_ambiguous_or_incomplete_rule(
    rule: dict[str, object],
) -> None:
    """Invalid explicit rules must fail rather than acquire defaults."""
    with pytest.raises(ValueError):
        evidence_prior_profile_from_dict(
            {
                "profile_id": "bad",
                "default": {"mean_log_odds": 0.0, "sd_log_odds": 0.5},
                "components": {"reservoir": {"ring_faults": rule}},
            }
        )
