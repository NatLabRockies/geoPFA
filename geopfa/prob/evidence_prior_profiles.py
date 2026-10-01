"""Load strict, portable evidence-coefficient prior profiles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.evidence_priors import (
    EvidenceCoefficientPrior,
    EvidencePriorGenerationConfig,
    EvidencePriorProfile,
    EvidencePriorRule,
)


def _finite_real(value: object, *, context: str) -> float:
    """Return one finite non-boolean value suitable for a profile field."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    result = float(value)
    if not (float("-inf") < result < float("inf")):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    return result


def _rule_from_dict(name: str, raw: Mapping[str, Any]) -> EvidencePriorRule:
    """Parse one named profile rule without tolerating ambiguous keys."""
    allowed = {
        "mean_log_odds",
        "sd_log_odds",
        "references",
        "rationale",
    }
    if unknown := set(raw) - allowed:
        raise GEOPFAValueError(
            f"evidence-prior rule {name!r} has unknown key(s): "
            + ", ".join(sorted(unknown))
        )
    required = {"mean_log_odds", "sd_log_odds"}
    if missing := required - set(raw):
        raise GEOPFAValueError(
            f"evidence-prior rule {name!r} requires "
            + ", ".join(sorted(missing))
        )
    references = raw.get("references", [])
    if not isinstance(references, list) or any(
        not isinstance(reference, str) or not reference.strip()
        for reference in references
    ):
        raise GEOPFAValueError(
            f"evidence-prior rule {name!r} references must be a JSON array "
            "of non-empty strings"
        )
    rationale = raw.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise GEOPFAValueError(
            f"evidence-prior rule {name!r} rationale must be a string or null"
        )
    return EvidencePriorRule(
        mean_log_odds=_finite_real(
            raw["mean_log_odds"],
            context=f"evidence-prior rule {name!r}.mean_log_odds",
        ),
        sd_log_odds=_finite_real(
            raw["sd_log_odds"],
            context=f"evidence-prior rule {name!r}.sd_log_odds",
        ),
        references=tuple(references),
        rationale=rationale,
    )


def evidence_prior_generation_config_from_dict(
    raw: Mapping[str, Any],
) -> EvidencePriorGenerationConfig:
    """Build an auditable profile configuration from strict JSON data."""
    allowed = {
        "default_mean_log_odds",
        "default_sd_log_odds",
        "redundancy_correlation",
        "rules",
    }
    if unknown := set(raw) - allowed:
        raise GEOPFAValueError(
            "unknown evidence-prior profile key(s): "
            + ", ".join(sorted(unknown))
        )
    rules_raw = raw.get("rules", {})
    if not isinstance(rules_raw, Mapping):
        raise GEOPFAValueError(
            "evidence-prior profile rules must be an object"
        )
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(rule, Mapping)
        for name, rule in rules_raw.items()
    ):
        raise GEOPFAValueError(
            "evidence-prior profile rules must map non-empty names to objects"
        )
    correlation = raw.get("redundancy_correlation")
    return EvidencePriorGenerationConfig(
        default_mean_log_odds=_finite_real(
            raw.get("default_mean_log_odds", 0.0),
            context="default_mean_log_odds",
        ),
        default_sd_log_odds=_finite_real(
            raw.get("default_sd_log_odds", 0.5),
            context="default_sd_log_odds",
        ),
        redundancy_correlation=(
            None
            if correlation is None
            else _finite_real(
                correlation,
                context="redundancy_correlation",
            )
        ),
        rules={
            name: _rule_from_dict(name, rule)
            for name, rule in rules_raw.items()
        },
    )


def load_evidence_prior_profile(
    path: str | Path,
) -> EvidencePriorGenerationConfig:
    """Load one JSON evidence-prior profile from disk."""
    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GEOPFAValueError(
            f"could not read evidence-prior profile {profile_path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise GEOPFAValueError(
            f"evidence-prior profile {profile_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise GEOPFAValueError("evidence-prior profile must be a JSON object")
    return evidence_prior_generation_config_from_dict(raw)


def _profile_coefficient_prior_from_dict(
    raw: Any,
    *,
    context: str,
) -> EvidenceCoefficientPrior:
    """Parse one explicit Gaussian or fixed coefficient-profile rule."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"evidence-prior rule {context!r} must be a mapping")
    allowed = {
        "mean_log_odds",
        "sd_log_odds",
        "fixed_value",
        "references",
        "rationale",
    }
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise GEOPFAValueError(
            f"evidence-prior rule {context!r} has unknown key(s): "
            + ", ".join(unexpected)
        )
    references = raw.get("references", ())
    if not isinstance(references, list | tuple) or not all(
        isinstance(reference, str) and reference.strip()
        for reference in references
    ):
        raise GEOPFAValueError(
            f"evidence-prior rule {context!r} references must be strings"
        )
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str):
        raise TypeError(
            f"evidence-prior rule {context!r} rationale must be a string"
        )
    return EvidenceCoefficientPrior(
        mean_log_odds=raw.get("mean_log_odds"),
        sd_log_odds=raw.get("sd_log_odds"),
        fixed_value=raw.get("fixed_value"),
        references=tuple(references),
        rationale=rationale,
    )


def evidence_prior_profile_from_dict(
    raw: Mapping[str, Any],
) -> EvidencePriorProfile:
    """Deserialize a self-contained profile with fixed-rule support.

    This schema is distinct from the generic generation-policy schema above:
    its explicit top-level default and component-qualified rules are intended
    for a study-specific, human-reviewable literature profile.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("evidence-prior profile must be a mapping")
    allowed = {"profile_id", "default", "components"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise GEOPFAValueError(
            "evidence-prior profile has unknown key(s): "
            + ", ".join(unexpected)
        )
    try:
        profile_id = raw["profile_id"]
        default_raw = raw["default"]
    except KeyError as error:
        raise GEOPFAValueError(
            f"evidence-prior profile requires {error.args[0]!r}"
        ) from error
    if not isinstance(profile_id, str):
        raise TypeError("evidence-prior profile_id must be a string")
    components_raw = raw.get("components", {})
    if not isinstance(components_raw, Mapping):
        raise TypeError("evidence-prior components must be a mapping")
    component_rules: dict[str, dict[str, EvidenceCoefficientPrior]] = {}
    for component, rules_raw in components_raw.items():
        if not isinstance(component, str) or not isinstance(
            rules_raw, Mapping
        ):
            raise TypeError(
                "evidence-prior component rules must be string mappings"
            )
        component_rules[component] = {
            feature: _profile_coefficient_prior_from_dict(
                rule_raw,
                context=f"{component}:{feature}",
            )
            for feature, rule_raw in rules_raw.items()
        }
    return EvidencePriorProfile(
        profile_id=profile_id,
        default=_profile_coefficient_prior_from_dict(
            default_raw,
            context="default",
        ),
        component_rules=component_rules,
    )


__all__ = [
    "evidence_prior_generation_config_from_dict",
    "evidence_prior_profile_from_dict",
    "load_evidence_prior_profile",
]
