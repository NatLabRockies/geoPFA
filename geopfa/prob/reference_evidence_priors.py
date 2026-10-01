"""Learn coefficient-prior rules from labelled reference domains.

Unlabelled target rasters can be diagnosed for redundancy, but an evidence
effect distribution can only be estimated from component labels in independent
reference domains.  This module fits regularised logistic models within those
domains and pools shared standardised-layer effects with random effects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.evidence_priors import (
    EvidencePriorGenerationConfig,
    EvidencePriorRule,
)


_MATRIX_DIMENSIONS = 2
_MINIMUM_REFERENCE_DOMAINS = 2
_MINIMUM_CLASS_COUNT = 2
_MINIMUM_CELLS = 4
_MAX_BACKTRACKS = 24
_VARIANCE_EPSILON = 1.0e-12


def _finite_real(value: object, *, context: str) -> float:
    """Return one finite non-boolean real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    return result


def _nonempty_string(value: object, *, context: str) -> str:
    """Validate one required provenance string."""
    if not isinstance(value, str) or not value.strip():
        raise GEOPFAValueError(f"{context} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ReferenceEvidenceDomain:
    """One independently labelled component dataset used to learn a prior."""

    domain_id: str
    component: str
    evidence: NDArray[np.float64] | Sequence[Sequence[float]]
    labels: NDArray[np.float64] | Sequence[float]
    feature_names: Sequence[str]
    source_id: str
    target_definition: str


@dataclass(frozen=True)
class ReferenceDomainPriorConfig:
    """Controls for transparent reference-domain prior learning."""

    ridge_penalty: float = 1.0
    minimum_domains_per_feature: int = _MINIMUM_REFERENCE_DOMAINS
    minimum_class_count: int = _MINIMUM_CLASS_COUNT
    uncertainty_floor: float = 0.25
    maximum_iterations: int = 100
    convergence_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate numerical controls before fitting any reference domain."""
        if _finite_real(self.ridge_penalty, context="ridge_penalty") <= 0.0:
            raise GEOPFAValueError("ridge_penalty must be strictly positive")
        if not isinstance(self.minimum_domains_per_feature, int) or isinstance(
            self.minimum_domains_per_feature, bool
        ):
            raise GEOPFAValueError(
                "minimum_domains_per_feature must be an integer"
            )
        if self.minimum_domains_per_feature < _MINIMUM_REFERENCE_DOMAINS:
            raise GEOPFAValueError(
                "minimum_domains_per_feature must be at least two"
            )
        if not isinstance(self.minimum_class_count, int) or isinstance(
            self.minimum_class_count, bool
        ):
            raise GEOPFAValueError("minimum_class_count must be an integer")
        if self.minimum_class_count < _MINIMUM_CLASS_COUNT:
            raise GEOPFAValueError("minimum_class_count must be at least two")
        if (
            _finite_real(self.uncertainty_floor, context="uncertainty_floor")
            <= 0
        ):
            raise GEOPFAValueError(
                "uncertainty_floor must be strictly positive"
            )
        if not isinstance(self.maximum_iterations, int) or isinstance(
            self.maximum_iterations, bool
        ):
            raise GEOPFAValueError("maximum_iterations must be an integer")
        if self.maximum_iterations <= 0:
            raise GEOPFAValueError(
                "maximum_iterations must be strictly positive"
            )
        if (
            _finite_real(
                self.convergence_tolerance,
                context="convergence_tolerance",
            )
            <= 0.0
        ):
            raise GEOPFAValueError(
                "convergence_tolerance must be strictly positive"
            )


@dataclass(frozen=True)
class ReferenceDomainFeatureEstimate:
    """One random-effects estimate with training-domain provenance."""

    feature_name: str
    mean_log_odds: float
    sd_log_odds: float
    between_domain_sd: float
    n_domains: int
    domain_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    target_definitions: tuple[str, ...]
    domain_mean_log_odds: tuple[float, ...]
    domain_standard_errors: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record for one learned layer."""
        return {
            "feature_name": self.feature_name,
            "mean_log_odds": self.mean_log_odds,
            "sd_log_odds": self.sd_log_odds,
            "between_domain_sd": self.between_domain_sd,
            "n_domains": self.n_domains,
            "domain_ids": list(self.domain_ids),
            "source_ids": list(self.source_ids),
            "target_definitions": list(self.target_definitions),
            "domain_mean_log_odds": list(self.domain_mean_log_odds),
            "domain_standard_errors": list(self.domain_standard_errors),
        }


@dataclass(frozen=True)
class ReferenceDomainPriorResult:
    """Portable rules learned from independently labelled domains."""

    component: str
    rules: Mapping[str, EvidencePriorRule]
    feature_estimates: Mapping[str, ReferenceDomainFeatureEstimate]
    config: ReferenceDomainPriorConfig
    status: str = "reference_domain_random_effects"

    def generation_config(
        self,
        *,
        default_mean_log_odds: float = 0.0,
        default_sd_log_odds: float = 0.5,
        redundancy_correlation: float | None = None,
    ) -> EvidencePriorGenerationConfig:
        """Return rules ready for target-raster prior generation.

        A target-only layer retains the supplied neutral default.
        """
        return EvidencePriorGenerationConfig(
            default_mean_log_odds=default_mean_log_odds,
            default_sd_log_odds=default_sd_log_odds,
            redundancy_correlation=redundancy_correlation,
            rules=self.rules,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record suitable for provenance."""
        return {
            "component": self.component,
            "status": self.status,
            "config": {
                "ridge_penalty": self.config.ridge_penalty,
                "minimum_domains_per_feature": (
                    self.config.minimum_domains_per_feature
                ),
                "minimum_class_count": self.config.minimum_class_count,
                "uncertainty_floor": self.config.uncertainty_floor,
            },
            "rules": {
                name: rule.to_dict() for name, rule in self.rules.items()
            },
            "feature_estimates": {
                name: estimate.to_dict()
                for name, estimate in self.feature_estimates.items()
            },
        }


@dataclass(frozen=True)
class _ValidatedReferenceDomain:
    """Canonical finite, class-balanced data for one fitted domain."""

    domain_id: str
    component: str
    evidence: NDArray[np.float64]
    labels: NDArray[np.float64]
    feature_names: tuple[str, ...]
    source_id: str
    target_definition: str


@dataclass(frozen=True)
class _DomainFit:
    """Estimated effects and local uncertainty for one reference domain."""

    domain: _ValidatedReferenceDomain
    effects: Mapping[str, float]
    variances: Mapping[str, float]


def _validate_reference_domain(
    domain: ReferenceEvidenceDomain,
    *,
    minimum_class_count: int,
) -> _ValidatedReferenceDomain:
    """Validate one domain without silently dropping invalid observations."""
    if not isinstance(domain, ReferenceEvidenceDomain):
        raise TypeError(
            "domains must contain ReferenceEvidenceDomain instances"
        )
    domain_id = _nonempty_string(domain.domain_id, context="domain_id")
    component = _nonempty_string(domain.component, context="component")
    source_id = _nonempty_string(domain.source_id, context="source_id")
    target_definition = _nonempty_string(
        domain.target_definition,
        context="target_definition",
    )
    names = tuple(domain.feature_names)
    if not names or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise GEOPFAValueError("feature_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise GEOPFAValueError("feature_names must not contain duplicates")
    evidence = np.asarray(domain.evidence, dtype=np.float64)
    labels = np.asarray(domain.labels, dtype=np.float64)
    if (
        evidence.ndim != _MATRIX_DIMENSIONS
        or evidence.shape[0] < _MINIMUM_CELLS
    ):
        raise GEOPFAValueError(
            "reference-domain evidence must be a two-dimensional matrix "
            "with at least four cells"
        )
    if evidence.shape[1] != len(names):
        raise GEOPFAValueError(
            "reference-domain evidence column count must match feature_names"
        )
    if labels.ndim != 1 or labels.shape[0] != evidence.shape[0]:
        raise GEOPFAValueError(
            "reference-domain labels must be one-dimensional with one value "
            "per evidence row"
        )
    if not np.all(np.isfinite(evidence)) or not np.all(np.isfinite(labels)):
        raise GEOPFAValueError(
            "reference-domain evidence and labels must be finite"
        )
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise GEOPFAValueError("reference-domain labels must be binary 0 or 1")
    class_counts = np.bincount(labels.astype(np.intp), minlength=2)
    if np.any(class_counts < minimum_class_count):
        raise GEOPFAValueError(
            "reference-domain labels must contain both binary classes at "
            f"least {minimum_class_count} times"
        )
    return _ValidatedReferenceDomain(
        domain_id=domain_id,
        component=component,
        evidence=evidence,
        labels=labels,
        feature_names=names,
        source_id=source_id,
        target_definition=target_definition,
    )


def _negative_log_posterior(
    design: NDArray[np.float64],
    labels: NDArray[np.float64],
    parameters: NDArray[np.float64],
    ridge_penalty: float,
) -> float:
    """Evaluate logistic likelihood plus an intercept-free ridge penalty."""
    eta = design @ parameters
    likelihood = np.logaddexp(0.0, eta).sum() - labels @ eta
    penalty = 0.5 * ridge_penalty * np.dot(parameters[1:], parameters[1:])
    return float(likelihood + penalty)


def _ridge_logistic_fit(
    evidence: NDArray[np.float64],
    labels: NDArray[np.float64],
    config: ReferenceDomainPriorConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a local standardised logistic model with damped Newton updates."""
    design = np.column_stack((np.ones(evidence.shape[0]), evidence))
    penalty = np.zeros(design.shape[1], dtype=np.float64)
    penalty[1:] = config.ridge_penalty
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(config.maximum_iterations):
        probabilities = expit(design @ parameters)
        gradient = design.T @ (probabilities - labels) + penalty * parameters
        if np.max(np.abs(gradient)) <= config.convergence_tolerance:
            break
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, np.newaxis] * design)
        hessian += np.diag(penalty)
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise GEOPFAValueError(
                "reference-domain logistic Hessian is singular"
            ) from error
        if np.max(np.abs(direction)) <= config.convergence_tolerance:
            break
        objective = _negative_log_posterior(
            design,
            labels,
            parameters,
            config.ridge_penalty,
        )
        step_size = 1.0
        numerical_slack = np.finfo(np.float64).eps * max(
            1.0,
            abs(objective),
        )
        for _ in range(_MAX_BACKTRACKS):
            candidate = parameters - step_size * direction
            candidate_objective = _negative_log_posterior(
                design,
                labels,
                candidate,
                config.ridge_penalty,
            )
            if candidate_objective <= objective + numerical_slack:
                parameters = candidate
                break
            step_size /= 2.0
        else:
            raise GEOPFAValueError(
                "reference-domain ridge logistic fit could not improve "
                "its objective"
            )
    else:
        raise GEOPFAValueError(
            "reference-domain ridge logistic fit did not converge"
        )
    probabilities = expit(design @ parameters)
    curvature = probabilities * (1.0 - probabilities)
    hessian = design.T @ (curvature[:, np.newaxis] * design)
    hessian += np.diag(penalty)
    covariance = np.linalg.pinv(hessian)
    if not np.all(np.isfinite(parameters)) or not np.all(
        np.isfinite(covariance)
    ):
        raise GEOPFAValueError(
            "reference-domain ridge logistic fit produced non-finite values"
        )
    return parameters, covariance


def _fit_domain(
    domain: _ValidatedReferenceDomain,
    config: ReferenceDomainPriorConfig,
) -> _DomainFit:
    """Fit only variable layers and retain their named estimates."""
    scale = domain.evidence.std(axis=0)
    active = np.flatnonzero(np.isfinite(scale) & (scale > 0.0))
    if active.size == 0:
        raise GEOPFAValueError(
            f"reference domain {domain.domain_id!r} has no variable "
            "evidence layers"
        )
    center = domain.evidence[:, active].mean(axis=0)
    standardised = (domain.evidence[:, active] - center) / scale[active]
    parameters, covariance = _ridge_logistic_fit(
        standardised,
        domain.labels,
        config,
    )
    effects: dict[str, float] = {}
    variances: dict[str, float] = {}
    for local_index, original_index in enumerate(active):
        feature = domain.feature_names[int(original_index)]
        variance = float(covariance[local_index + 1, local_index + 1])
        effects[feature] = float(parameters[local_index + 1])
        variances[feature] = max(variance, _VARIANCE_EPSILON)
    return _DomainFit(domain=domain, effects=effects, variances=variances)


def _random_effects_summary(
    effects: NDArray[np.float64],
    variances: NDArray[np.float64],
    uncertainty_floor: float,
) -> tuple[float, float, float]:
    """Pool independent estimates with DerSimonian--Laird moments."""
    weights = 1.0 / variances
    fixed_mean = float(np.sum(weights * effects) / np.sum(weights))
    heterogeneity = float(np.sum(weights * (effects - fixed_mean) ** 2))
    denominator = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    between_variance = (
        0.0
        if denominator <= _VARIANCE_EPSILON
        else max(0.0, (heterogeneity - (len(effects) - 1)) / denominator)
    )
    random_weights = 1.0 / (variances + between_variance)
    mean = float(np.sum(random_weights * effects) / np.sum(random_weights))
    mean_variance = float(1.0 / np.sum(random_weights))
    predictive_sd = math.sqrt(mean_variance + between_variance)
    return (
        mean,
        max(predictive_sd, uncertainty_floor),
        math.sqrt(between_variance),
    )


def _ordered_feature_names(
    fits: Sequence[_DomainFit],
) -> tuple[str, ...]:
    """Return first-seen layer names for stable, human-readable output."""
    names: list[str] = []
    seen: set[str] = set()
    for fit in fits:
        for name in fit.domain.feature_names:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    """Preserve first occurrence while removing duplicate provenance values."""
    return tuple(dict.fromkeys(values))


def learn_reference_domain_evidence_priors(  # noqa: PLR0914
    domains: Sequence[ReferenceEvidenceDomain],
    *,
    config: ReferenceDomainPriorConfig | None = None,
) -> ReferenceDomainPriorResult:
    """Learn transferable priors from independently labelled reference domains.

    Domains must describe one component but may have different layer sets. A
    rule is emitted only for a variable layer estimated in at least the
    configured number of domains. Target-only layers consequently retain a
    neutral default when this result is used for target-raster generation.

    This estimates associations, not target calibration. The caller must keep
    the target domain out of training and perform spatial holdout validation
    before treating an exported rule as transferable.
    """
    source_domains = tuple(domains)
    if len(source_domains) < _MINIMUM_REFERENCE_DOMAINS:
        raise GEOPFAValueError(
            "reference-domain prior learning requires at least two "
            "independent domains"
        )
    policy = ReferenceDomainPriorConfig() if config is None else config
    if not isinstance(policy, ReferenceDomainPriorConfig):
        raise TypeError("config must be a ReferenceDomainPriorConfig or None")
    validated = tuple(
        _validate_reference_domain(
            domain,
            minimum_class_count=policy.minimum_class_count,
        )
        for domain in source_domains
    )
    domain_ids = tuple(domain.domain_id for domain in validated)
    if len(set(domain_ids)) != len(domain_ids):
        raise GEOPFAValueError(
            "reference-domain domain_id values must be unique"
        )
    components = {domain.component for domain in validated}
    if len(components) != 1:
        raise GEOPFAValueError(
            "reference domains must all describe the same component"
        )
    fitted = tuple(_fit_domain(domain, policy) for domain in validated)
    rules: dict[str, EvidencePriorRule] = {}
    estimates: dict[str, ReferenceDomainFeatureEstimate] = {}
    component = validated[0].component
    for feature_name in _ordered_feature_names(fitted):
        available = tuple(fit for fit in fitted if feature_name in fit.effects)
        if len(available) < policy.minimum_domains_per_feature:
            continue
        effects = np.asarray(
            [fit.effects[feature_name] for fit in available],
            dtype=np.float64,
        )
        variances = np.asarray(
            [fit.variances[feature_name] for fit in available],
            dtype=np.float64,
        )
        mean, sd, between_domain_sd = _random_effects_summary(
            effects,
            variances,
            policy.uncertainty_floor,
        )
        fitted_domain_ids = tuple(fit.domain.domain_id for fit in available)
        source_ids = _unique(tuple(fit.domain.source_id for fit in available))
        target_definitions = _unique(
            tuple(fit.domain.target_definition for fit in available)
        )
        rationale = (
            "Ridge-logistic estimates from independently labelled reference "
            f"domains ({len(available)}), pooled with a random-effects "
            "model on each domain's within-domain standard-deviation scale. "
            "Target definition(s): " + "; ".join(target_definitions)
        )
        rules[f"{component}:{feature_name}"] = EvidencePriorRule(
            mean_log_odds=mean,
            sd_log_odds=sd,
            references=source_ids,
            rationale=rationale,
        )
        estimates[feature_name] = ReferenceDomainFeatureEstimate(
            feature_name=feature_name,
            mean_log_odds=mean,
            sd_log_odds=sd,
            between_domain_sd=between_domain_sd,
            n_domains=len(available),
            domain_ids=fitted_domain_ids,
            source_ids=source_ids,
            target_definitions=target_definitions,
            domain_mean_log_odds=tuple(float(value) for value in effects),
            domain_standard_errors=tuple(
                float(math.sqrt(value)) for value in variances
            ),
        )
    if not rules:
        raise GEOPFAValueError(
            "no evidence layer was variable in enough labelled reference "
            "domains to learn a transferable prior"
        )
    return ReferenceDomainPriorResult(
        component=component,
        rules=rules,
        feature_estimates=estimates,
        config=policy,
    )


__all__ = [
    "ReferenceDomainFeatureEstimate",
    "ReferenceDomainPriorConfig",
    "ReferenceDomainPriorResult",
    "ReferenceEvidenceDomain",
    "learn_reference_domain_evidence_priors",
]
