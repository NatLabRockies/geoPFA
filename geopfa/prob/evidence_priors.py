"""Auditable coefficient-prior generation for unlabeled evidence layers.

Raster diagnostics can standardise values and identify redundant layers, but
cannot identify the sign or size of an unobserved component effect. Unmatched
layers therefore receive neutral zero-centred priors; directional rules must
come from an external source and retain their provenance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from geopfa.exceptions import GEOPFAValueError


_MATRIX_DIMENSIONS = 2
_MINIMUM_CELLS = 2
_MINIMUM_GROUP_SIZE = 2


def _finite_real(value: object, *, context: str) -> float:
    """Return a finite, non-boolean real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    return result


def _string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    """Return a validated immutable collection of strings."""
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise GEOPFAValueError(
            f"{context} must be a tuple of non-empty strings"
        )
    return value


@dataclass(frozen=True)
class EvidencePriorRule:
    """One externally justified distribution for a standardised coefficient."""

    mean_log_odds: float
    sd_log_odds: float
    references: tuple[str, ...] = ()
    rationale: str | None = None

    def __post_init__(self) -> None:
        """Validate the distribution and its provenance fields."""
        _finite_real(self.mean_log_odds, context="mean_log_odds")
        if _finite_real(self.sd_log_odds, context="sd_log_odds") <= 0.0:
            raise GEOPFAValueError("sd_log_odds must be strictly positive")
        _string_tuple(self.references, context="references")
        if self.rationale is not None and (
            not isinstance(self.rationale, str) or not self.rationale.strip()
        ):
            raise GEOPFAValueError(
                "rationale must be a non-empty string or None"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        result: dict[str, Any] = {
            "mean_log_odds": self.mean_log_odds,
            "sd_log_odds": self.sd_log_odds,
        }
        if self.references:
            result["references"] = list(self.references)
        if self.rationale is not None:
            result["rationale"] = self.rationale
        return result


@dataclass(frozen=True)
class EvidenceCoefficientPrior:
    """One Gaussian or fixed rule on the standardised log-odds scale."""

    mean_log_odds: float | None = None
    sd_log_odds: float | None = None
    fixed_value: float | None = None
    references: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        """Reject ambiguous or improper coefficient rules."""
        is_fixed = self.fixed_value is not None
        is_gaussian = (
            self.mean_log_odds is not None or self.sd_log_odds is not None
        )
        if is_fixed == is_gaussian:
            raise GEOPFAValueError(
                "an evidence coefficient prior must be either Gaussian or fixed"
            )
        if is_fixed:
            _finite_real(self.fixed_value, context="fixed_value")
        else:
            if self.mean_log_odds is None or self.sd_log_odds is None:
                raise GEOPFAValueError(
                    "Gaussian coefficient priors require mean and SD"
                )
            _finite_real(self.mean_log_odds, context="mean_log_odds")
            if _finite_real(self.sd_log_odds, context="sd_log_odds") <= 0.0:
                raise GEOPFAValueError(
                    "coefficient-prior SDs must be finite and positive"
                )
        _string_tuple(self.references, context="references")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")

    @classmethod
    def neutral(cls, *, sd_log_odds: float = 0.5) -> EvidenceCoefficientPrior:
        """Return a neutral zero-centred proper Gaussian coefficient prior."""
        return cls(mean_log_odds=0.0, sd_log_odds=sd_log_odds)

    @classmethod
    def fixed(
        cls, value: float, *, rationale: str = ""
    ) -> EvidenceCoefficientPrior:
        """Return an auditable fixed coefficient rule."""
        return cls(fixed_value=value, rationale=rationale)

    @property
    def precision(self) -> float:
        """Return Gaussian precision, rejecting fixed rules."""
        if self.sd_log_odds is None:
            raise GEOPFAValueError(
                "fixed coefficient rules do not have a precision"
            )
        return 1.0 / self.sd_log_odds**2


@dataclass(frozen=True)
class EvidencePriorProfile:
    """A named default rule plus component-qualified overrides."""

    profile_id: str
    default: EvidenceCoefficientPrior
    component_rules: Mapping[str, Mapping[str, EvidenceCoefficientPrior]] = (
        field(default_factory=dict)
    )

    def __post_init__(self) -> None:
        """Validate profile structure before it reaches a model configuration."""
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise GEOPFAValueError("profile_id must be a non-empty string")
        if not isinstance(self.default, EvidenceCoefficientPrior):
            raise TypeError("default must be an EvidenceCoefficientPrior")
        for component, rules in self.component_rules.items():
            if not isinstance(component, str) or not component.strip():
                raise GEOPFAValueError(
                    "evidence-prior component names must be non-empty strings"
                )
            if not isinstance(rules, Mapping):
                raise TypeError(
                    "component evidence-prior rules must be mappings"
                )
            for feature, rule in rules.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise GEOPFAValueError(
                        "evidence-prior feature names must be non-empty strings"
                    )
                if not isinstance(rule, EvidenceCoefficientPrior):
                    raise TypeError(
                        "evidence-prior rules must be EvidenceCoefficientPrior"
                    )

    @classmethod
    def neutral(
        cls, *, sd_log_odds: float = 0.5, profile_id: str = "neutral-v1"
    ) -> EvidencePriorProfile:
        """Return a neutral profile that makes no directional claim."""
        return cls(
            profile_id=profile_id,
            default=EvidenceCoefficientPrior.neutral(sd_log_odds=sd_log_odds),
        )

    def resolve(
        self, component: str, feature: str
    ) -> EvidenceCoefficientPrior:
        """Return the component-qualified override or the profile default."""
        return self.component_rules.get(component, {}).get(
            feature, self.default
        )


@dataclass(frozen=True)
class ResolvedEvidencePriors:
    """Configuration-ready coefficient priors and their source assignments."""

    profile_id: str
    prior_means: Mapping[str, float]
    prior_precisions: Mapping[str, float]
    fixed_coefficients: Mapping[str, float]
    assignments: Mapping[str, EvidenceCoefficientPrior]


def resolve_evidence_prior_profile(
    profile: EvidencePriorProfile,
    active_features: Mapping[str, Sequence[str]],
) -> ResolvedEvidencePriors:
    """Resolve a strict profile against exact active component features."""
    if not isinstance(profile, EvidencePriorProfile):
        raise TypeError("profile must be an EvidencePriorProfile")
    normalized: dict[str, tuple[str, ...]] = {}
    for component, features in active_features.items():
        if not isinstance(component, str) or not component.strip():
            raise GEOPFAValueError(
                "active component names must be non-empty strings"
            )
        names = tuple(features)
        if len(names) != len(set(names)):
            raise GEOPFAValueError(
                f"active evidence features repeat in {component!r}"
            )
        if any(
            not isinstance(feature, str) or not feature.strip()
            for feature in names
        ):
            raise GEOPFAValueError(
                "active evidence feature names must be non-empty strings"
            )
        normalized[component] = names
    for component, rules in profile.component_rules.items():
        stale = sorted(set(rules) - set(normalized.get(component, ())))
        if stale:
            raise GEOPFAValueError(
                f"evidence-prior profile {profile.profile_id!r} assigns inactive "
                f"feature(s) for {component!r}: {', '.join(stale)}"
            )
    prior_means: dict[str, float] = {}
    prior_precisions: dict[str, float] = {}
    fixed_coefficients: dict[str, float] = {}
    assignments: dict[str, EvidenceCoefficientPrior] = {}
    for component, features in normalized.items():
        for feature in features:
            key = f"{component}:{feature}"
            rule = profile.resolve(component, feature)
            assignments[key] = rule
            if rule.fixed_value is None:
                prior_means[key] = float(rule.mean_log_odds)
                prior_precisions[key] = rule.precision
            else:
                fixed_coefficients[key] = float(rule.fixed_value)
    return ResolvedEvidencePriors(
        profile_id=profile.profile_id,
        prior_means=prior_means,
        prior_precisions=prior_precisions,
        fixed_coefficients=fixed_coefficients,
        assignments=assignments,
    )


@dataclass(frozen=True)
class EvidencePriorGenerationConfig:
    """Generic policy and optional exact component/layer rules.

    An unmatched layer receives ``Normal(0, default_sd_log_odds**2)``. Setting
    ``redundancy_correlation`` is opt-in and makes each correlated layer group
    share its total directional and uncertainty budget.
    """

    default_mean_log_odds: float = 0.0
    default_sd_log_odds: float = 0.5
    redundancy_correlation: float | None = None
    rules: Mapping[str, EvidencePriorRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate policy parameters and named rules."""
        _finite_real(
            self.default_mean_log_odds,
            context="default_mean_log_odds",
        )
        if (
            _finite_real(
                self.default_sd_log_odds,
                context="default_sd_log_odds",
            )
            <= 0.0
        ):
            raise GEOPFAValueError(
                "default_sd_log_odds must be strictly positive"
            )
        if self.redundancy_correlation is not None:
            threshold = _finite_real(
                self.redundancy_correlation,
                context="redundancy_correlation",
            )
            if not 0.0 < threshold <= 1.0:
                raise GEOPFAValueError(
                    "redundancy_correlation must be in (0, 1]"
                )
        if not isinstance(self.rules, Mapping):
            raise TypeError("rules must map layer names to EvidencePriorRule")
        for name, rule in self.rules.items():
            if not isinstance(name, str) or not name.strip():
                raise GEOPFAValueError(
                    "evidence-prior rule keys must be non-empty strings"
                )
            if not isinstance(rule, EvidencePriorRule):
                raise TypeError(
                    "evidence-prior rules must contain EvidencePriorRule "
                    "instances"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy."""
        result: dict[str, Any] = {
            "default_mean_log_odds": self.default_mean_log_odds,
            "default_sd_log_odds": self.default_sd_log_odds,
            "rules": {
                name: rule.to_dict() for name, rule in self.rules.items()
            },
        }
        if self.redundancy_correlation is not None:
            result["redundancy_correlation"] = self.redundancy_correlation
        return result


@dataclass(frozen=True)
class EvidenceLayerSummary:
    """Data-only diagnostics for one layer on the prediction support."""

    feature_name: str
    center: float
    scale: float
    minimum: float
    maximum: float
    n_cells: int


@dataclass(frozen=True)
class EvidencePriorRecommendation:
    """An auditable prior recommendation on the standardised logit scale."""

    component: str
    feature_names: tuple[str, ...]
    mean_log_odds: tuple[float, ...]
    sd_log_odds: tuple[float, ...]
    summaries: tuple[EvidenceLayerSummary, ...]
    redundancy_groups: tuple[tuple[str, ...], ...]
    references: Mapping[str, tuple[str, ...]]
    rationales: Mapping[str, tuple[str, ...]]
    status: str

    def regularization_values(self) -> dict[str, dict[str, float]]:
        """Return existing geoPFA regularization maps with qualified keys."""
        names = tuple(
            f"{self.component}:{feature}" for feature in self.feature_names
        )
        return {
            "prior_means": dict(zip(names, self.mean_log_odds, strict=True)),
            "prior_precisions": {
                name: 1.0 / sd**2
                for name, sd in zip(names, self.sd_log_odds, strict=True)
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return compact recommendation provenance suitable for a manifest."""
        return {
            "component": self.component,
            "feature_names": list(self.feature_names),
            "mean_log_odds": list(self.mean_log_odds),
            "sd_log_odds": list(self.sd_log_odds),
            "summaries": [summary.__dict__ for summary in self.summaries],
            "redundancy_groups": [
                list(group) for group in self.redundancy_groups
            ],
            "references": {
                name: list(values) for name, values in self.references.items()
            },
            "rationales": {
                name: list(values) for name, values in self.rationales.items()
            },
            "status": self.status,
        }


EvidencePriorHeuristic = Callable[
    [str, EvidenceLayerSummary], EvidencePriorRule | None
]


def _validated_evidence(
    evidence: NDArray[np.float64] | Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Validate a finite, nonconstant matrix with named columns."""
    names = tuple(feature_names)
    if not names or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise GEOPFAValueError("feature_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise GEOPFAValueError("feature_names must not contain duplicates")
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != _MATRIX_DIMENSIONS or values.shape[0] < _MINIMUM_CELLS:
        raise GEOPFAValueError(
            "evidence must be a two-dimensional matrix with at least two cells"
        )
    if values.shape[1] != len(names):
        raise GEOPFAValueError(
            "evidence column count must match feature_names"
        )
    if not np.all(np.isfinite(values)):
        raise GEOPFAValueError(
            "evidence must be finite on every prediction cell"
        )
    constant = np.flatnonzero(values.std(axis=0) <= 0.0)
    if constant.size:
        listed = ", ".join(names[index] for index in constant)
        raise GEOPFAValueError(
            "evidence layer(s) are constant over prediction support: " + listed
        )
    return values, names


def _redundancy_groups(
    correlation: NDArray[np.float64],
    names: tuple[str, ...],
    threshold: float | None,
) -> tuple[tuple[str, ...], ...]:
    """Return connected absolute-correlation groups of multiple layers."""
    if threshold is None or len(names) < _MINIMUM_GROUP_SIZE:
        return ()
    adjacent = np.abs(correlation) >= threshold
    np.fill_diagonal(adjacent, False)
    seen = np.zeros(len(names), dtype=bool)
    groups: list[tuple[str, ...]] = []
    for start in range(len(names)):
        if seen[start]:
            continue
        pending, members = [start], set()
        seen[start] = True
        while pending:
            current = pending.pop()
            members.add(current)
            for neighbor in np.flatnonzero(adjacent[current]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    pending.append(int(neighbor))
        if len(members) >= _MINIMUM_GROUP_SIZE:
            groups.append(tuple(sorted(names[index] for index in members)))
    return tuple(sorted(groups))


def _rule_for(
    component: str,
    summary: EvidenceLayerSummary,
    config: EvidencePriorGenerationConfig,
    heuristic: EvidencePriorHeuristic | None,
) -> EvidencePriorRule | None:
    """Resolve qualified, global, or programmatic rule precedence."""
    name = summary.feature_name
    rule = config.rules.get(f"{component}:{name}", config.rules.get(name))
    if rule is not None or heuristic is None:
        return rule
    resolved = heuristic(component, summary)
    if resolved is not None and not isinstance(resolved, EvidencePriorRule):
        raise TypeError(
            "evidence-prior heuristic must return EvidencePriorRule or None"
        )
    return resolved


def _recommend_evidence_coefficient_priors(  # noqa: PLR0914
    evidence: NDArray[np.float64] | Sequence[Sequence[float]],
    feature_names: Sequence[str],
    *,
    component: str,
    config: EvidencePriorGenerationConfig | None = None,
    heuristic: EvidencePriorHeuristic | None = None,
) -> EvidencePriorRecommendation:
    """Recommend auditable priors without inventing an unlabeled direction.

    Rules may be declared with exact component-qualified keys (which take
    precedence) or global layer names, or supplied as a small callable. With no
    matching rule, the mean stays zero. Correlation-based sharing is a declared
    duplicate-evidence safeguard, not a learned dependence model.
    """
    if not isinstance(component, str) or not component.strip():
        raise GEOPFAValueError("component must be a non-empty string")
    values, names = _validated_evidence(evidence, feature_names)
    policy = EvidencePriorGenerationConfig() if config is None else config
    if not isinstance(policy, EvidencePriorGenerationConfig):
        raise TypeError(
            "config must be an EvidencePriorGenerationConfig or None"
        )
    center, scale = values.mean(axis=0), values.std(axis=0)
    standardised = (values - center) / scale
    correlation = np.atleast_2d(np.corrcoef(standardised, rowvar=False))
    summaries = tuple(
        EvidenceLayerSummary(
            feature_name=name,
            center=float(center[index]),
            scale=float(scale[index]),
            minimum=float(values[:, index].min()),
            maximum=float(values[:, index].max()),
            n_cells=values.shape[0],
        )
        for index, name in enumerate(names)
    )
    rules = tuple(
        _rule_for(component, summary, policy, heuristic)
        for summary in summaries
    )
    mean = np.asarray(
        [
            policy.default_mean_log_odds
            if rule is None
            else rule.mean_log_odds
            for rule in rules
        ],
        dtype=np.float64,
    )
    sd = np.asarray(
        [
            policy.default_sd_log_odds if rule is None else rule.sd_log_odds
            for rule in rules
        ],
        dtype=np.float64,
    )
    groups = _redundancy_groups(
        correlation,
        names,
        policy.redundancy_correlation,
    )
    for group in groups:
        indices = np.asarray([names.index(name) for name in group])
        mean[indices] /= len(indices)
        sd[indices] /= math.sqrt(len(indices))
    references = {
        name: rule.references
        for name, rule in zip(names, rules, strict=True)
        if rule is not None and rule.references
    }
    rationales = {
        name: (rule.rationale,)
        for name, rule in zip(names, rules, strict=True)
        if rule is not None and rule.rationale is not None
    }
    status = (
        "neutral_unlabeled"
        if all(rule is None for rule in rules)
        else "rule_informed_unlabeled"
    )
    if groups:
        status += "_redundancy_adjusted"
    return EvidencePriorRecommendation(
        component=component,
        feature_names=names,
        mean_log_odds=tuple(float(value) for value in mean),
        sd_log_odds=tuple(float(value) for value in sd),
        summaries=summaries,
        redundancy_groups=groups,
        references=references,
        rationales=rationales,
        status=status,
    )


def derive_evidence_coefficient_priors(
    evidence_or_profile: (
        EvidencePriorProfile | NDArray[np.float64] | Sequence[Sequence[float]]
    ),
    feature_names_or_active_features: (
        Sequence[str] | Mapping[str, Sequence[str]]
    ),
    *,
    component: str | None = None,
    config: EvidencePriorGenerationConfig | None = None,
    heuristic: EvidencePriorHeuristic | None = None,
) -> EvidencePriorRecommendation | ResolvedEvidencePriors:
    """Generate generic priors or resolve an explicit component profile.

    Matrix input requires ``component=...`` and produces a data-diagnostic
    recommendation. Passing an :class:`EvidencePriorProfile` resolves its
    explicit Gaussian or fixed rules against active component feature names.
    The profile pathway is for reviewable study-specific rules; unlabeled data
    alone never supplies a directional effect.
    """
    if isinstance(evidence_or_profile, EvidencePriorProfile):
        if (
            component is not None
            or config is not None
            or heuristic is not None
        ):
            raise TypeError(
                "profile resolution does not accept component, config, or heuristic"
            )
        if not isinstance(feature_names_or_active_features, Mapping):
            raise TypeError(
                "profile resolution requires active feature mappings"
            )
        return resolve_evidence_prior_profile(
            evidence_or_profile,
            feature_names_or_active_features,
        )
    if component is None:
        raise TypeError("matrix prior generation requires component=...")
    if isinstance(feature_names_or_active_features, Mapping):
        raise TypeError(
            "matrix prior generation requires ordered feature names"
        )
    return _recommend_evidence_coefficient_priors(
        evidence_or_profile,
        feature_names_or_active_features,
        component=component,
        config=config,
        heuristic=heuristic,
    )


__all__ = [
    "EvidenceCoefficientPrior",
    "EvidenceLayerSummary",
    "EvidencePriorGenerationConfig",
    "EvidencePriorHeuristic",
    "EvidencePriorProfile",
    "EvidencePriorRecommendation",
    "EvidencePriorRule",
    "ResolvedEvidencePriors",
    "derive_evidence_coefficient_priors",
    "resolve_evidence_prior_profile",
]
