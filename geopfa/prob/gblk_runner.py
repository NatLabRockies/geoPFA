"""Joint GBLK probabilistic runner for geoPFA.

Maps a ``ProbabilisticConfig`` onto the joint GBLK fitter, returning a
:class:`~geopfa.prob.runner.ProbabilisticResult` whose schema is identical
to the sequential-backend result so downstream consumers are unchanged.

The entry point :func:`run_gblk_probabilistic` replaces the per-component
sequential path for ``inference.backend == "gblk"``:

1. Build :class:`~geopfa.prob.pfa_grid.PFAGridAdapter` and load labels.
2. Build per-component ``alpha_c`` offsets.
3. Assemble joint inputs via
   :func:`~geopfa.prob.gblk_assemble.assemble_gblk_inputs`.
4. Fit evidence coefficients and all spatial components simultaneously via
   :func:`~geopfa.prob.gblk_backend.fit_gblk_joint`.
5. Package per-component ``ComponentProbability`` GeoDataFrames and the
   joint combined surface into a ``ProbabilisticResult``.

Calibration and output writing are deferred to later phases (P6, P5-S03).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, ndtr

from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.alpha import AlphaCResult, build_alpha_c
from geopfa.prob.calibration import calibration_intercept_slope, log_loss
from geopfa.prob.config import GBLKBayesianConfig, ProbabilisticConfig
from geopfa.prob.cv import spatial_block_cv
from geopfa.prob.fitting import ComponentProbability
from geopfa.prob.fitting import _fit_offset_logit
from geopfa.prob.forward import FrozenGBLKForwardState
from geopfa.prob.gblk_assemble import (
    AssembledInputs,
    assemble_gblk_inputs,
    build_component_grid_evidence,
    pool_regional_coefficients,
)
from geopfa.prob.gblk_backend import (
    fit_gblk_bayesian_joint,
    fit_gblk_bayesian_posterior_state,
    fit_gblk_gaussian_bayesian_joint,
    fit_gblk_joint,
    project_gblk_bayesian_draw_block,
)
from geopfa.prob.io import (
    PersistedPosteriorDrawState,
    PosteriorDrawBlockWriter,
    load_posterior_draw_state,
)
from geopfa.prob.labels import LoadedLabels, load_labels
from geopfa.prob.pfa_grid import PFAGridAdapter, validate_declared_components
from geopfa.prob.predictive_stacking import (
    PredictiveStackingResult,
    apply_predictive_stacking,
    select_predictive_stacking_weight,
)
from geopfa.prob.runner import ProbabilisticResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from latticekrigx.glk.calibration import CalibrationCVResult

_TWO_DIMENSIONS = 2
_THREE_DIMENSIONS = 3


def _spawn_child_seeds(seed: int, count: int) -> tuple[int, ...]:
    """Derive deterministic seeds accepted by pyINLA and legacy NumPy APIs."""
    children = np.random.SeedSequence(seed).spawn(count)
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    )


def _validate_gblk_config(cfg: ProbabilisticConfig) -> None:
    """Require a valid config that explicitly selects the GBLK backend."""
    cfg.validate_raise()
    if cfg.inference.backend != "gblk":
        raise GEOPFAValueError(
            "direct GBLK runners require inference.backend='gblk'"
        )


def _combine_probability_columns(
    probabilities: NDArray[np.float64], rule: str
) -> NDArray[np.float64]:
    """Combine an ``(n, q)`` probability matrix by the configured rule."""
    if rule == "product":
        return np.prod(probabilities, axis=1)
    if rule == "geometric_mean":
        return np.exp(
            np.mean(np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
        )
    raise AssertionError(f"unexpected combination rule {rule!r}")


def _resolve_a_wght(
    spatial_dimension: int,
    requested: float | None,
) -> float:
    """Return a dimension-appropriate SAR diagonal weight.

    A first-order 3-D lattice has six neighbors, so its diagonal weight must
    exceed six.  The former runner silently replaced every explicit 3-D value
    with 6.5, which discarded caller configuration and left little numerical
    separation from the positive-definiteness boundary.  Use 8.0 only as the
    3-D default and otherwise preserve the caller's value verbatim.
    """
    if spatial_dimension not in {_TWO_DIMENSIONS, _THREE_DIMENSIONS}:
        raise ValueError("spatial_dimension must equal 2 or 3")
    value = (
        (4.5 if spatial_dimension == _TWO_DIMENSIONS else 8.0)
        if requested is None
        else float(requested)
    )
    if not np.isfinite(value):
        raise ValueError("a_wght must be finite")
    if spatial_dimension == _TWO_DIMENSIONS and value <= 4.0:  # noqa: PLR2004
        raise ValueError("a_wght must be greater than 4 for a 2-D lattice")
    if spatial_dimension == _THREE_DIMENSIONS and value <= 6.0:  # noqa: PLR2004
        raise ValueError("a_wght must be greater than 6 for a 3-D lattice")
    return value


@dataclass(frozen=True)
class JointEvidenceDesign:
    """Standardized evidence arrays and their proper coefficient priors."""

    train: NDArray[np.float64] | None
    prediction: NDArray[np.float64] | None
    precision: NDArray[np.float64] | None
    prior_mean: NDArray[np.float64] | None
    diagnostics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PriorPredictiveEvidenceDraws:
    """Draw-level prediction from declared Gaussian evidence priors."""

    probability_draws: NDArray[np.float64]
    coefficient_draws: NDArray[np.float64]
    feature_names: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PriorPredictiveEvidenceState:
    """Immutable prior coefficient draws and standardized grid evidence."""

    standardized_evidence: NDArray[np.float64]
    coefficient_draws: NDArray[np.float64]
    feature_names: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PersistedBayesianFitReference:
    """Reference to a hash-verified INLA fit state reopened from disk."""

    state_fingerprint: str
    posterior_draw_index: Path | None
    inference: str = "inla_persisted_state"


def _qualified_prior_values(  # noqa: PLR0913
    values: Mapping[str, float],
    *,
    component: str,
    layer_names: list[str],
    prior_name: str,
    require_explicit: bool,
    default: float,
) -> NDArray[np.float64]:
    """Resolve component-qualified coefficient prior values."""
    resolved: list[float] = []
    missing: list[str] = []
    for layer_name in layer_names:
        qualified = f"{component}:{layer_name}"
        if qualified in values:
            value = values[qualified]
        elif layer_name in values:
            value = values[layer_name]
        elif require_explicit:
            missing.append(qualified)
            continue
        else:
            value = default
        resolved.append(float(value))
    if missing:
        raise GEOPFAValueError(
            f"component {component!r} requires an explicit Gaussian prior "
            f"{prior_name} for every active evidence layer; missing: "
            + ", ".join(missing)
        )
    return np.asarray(resolved, dtype=np.float64)


def _prior_predictive_evidence_state(
    adapter: PFAGridAdapter,
    component: str,
    alpha: AlphaCResult,
    cfg: ProbabilisticConfig,
    *,
    seed: int,
) -> PriorPredictiveEvidenceState:
    """Prepare one component's named Gaussian coefficient-prior draws.

    The covariates are standardized over the declared prediction support, so
    each coefficient is a log-odds effect per one support-standard-deviation
    change. Legacy PFA layer or component ``weight`` values are never read.
    """
    evidence, layer_names = build_component_grid_evidence(
        adapter,
        component,
        alpha,
        cfg.evidence,
    )
    if not layer_names:
        raise GEOPFAValueError(
            f"component {component!r} requested an evidence prior but has "
            "no active evidence layers"
        )
    evidence = np.asarray(evidence, dtype=np.float64)
    if not np.all(np.isfinite(evidence)):
        raise GEOPFAValueError(
            f"component {component!r} evidence must be finite on every "
            "prediction cell"
        )
    center = evidence.mean(axis=0)
    scale = evidence.std(axis=0)
    constant = np.flatnonzero(~np.isfinite(scale) | (scale <= 0.0))
    if constant.size:
        names = ", ".join(layer_names[index] for index in constant)
        raise GEOPFAValueError(
            f"component {component!r} evidence layer(s) are constant over "
            f"the prediction support: {names}"
        )
    standardized = (evidence - center) / scale
    regularization = cfg.evidence.regularization
    prior_mean = _qualified_prior_values(
        regularization.prior_means,
        component=component,
        layer_names=layer_names,
        prior_name="mean",
        require_explicit=True,
        default=0.0,
    )
    prior_precision = _qualified_prior_values(
        regularization.prior_precisions,
        component=component,
        layer_names=layer_names,
        prior_name="precision",
        require_explicit=True,
        default=1.0,
    )
    if (
        not np.all(np.isfinite(prior_mean))
        or not np.all(np.isfinite(prior_precision))
        or np.any(prior_precision <= 0.0)
    ):
        raise GEOPFAValueError(
            "evidence coefficient priors require finite means and strictly "
            "positive finite precisions"
        )
    prior_sd = 1.0 / np.sqrt(prior_precision)
    rng = np.random.default_rng(seed)
    coefficient_draws = rng.normal(
        loc=prior_mean,
        scale=prior_sd,
        size=(cfg.inference.gblk_bayesian.n_draws, len(layer_names)),
    )
    return PriorPredictiveEvidenceState(
        standardized_evidence=standardized,
        coefficient_draws=coefficient_draws,
        feature_names=tuple(layer_names),
        diagnostics={
            "inference_role": "evidence_coefficient_prior_predictive",
            "outcome_update": False,
            "spatial_field_included": False,
            "alpha_provenance": alpha.provenance,
            "evidence_standardization": (
                "declared_prediction_support_mean_and_standard_deviation"
            ),
            "evidence_center": center.tolist(),
            "evidence_scale": scale.tolist(),
            "coefficient_prior_mean": prior_mean.tolist(),
            "coefficient_prior_sd": prior_sd.tolist(),
            "n_draws": cfg.inference.gblk_bayesian.n_draws,
            "seed": seed,
        },
    )


def _prior_predictive_evidence_draws(
    adapter: PFAGridAdapter,
    component: str,
    alpha: AlphaCResult,
    cfg: ProbabilisticConfig,
    *,
    seed: int,
) -> PriorPredictiveEvidenceDraws:
    """Evaluate all draws through the existing in-memory API."""
    state = _prior_predictive_evidence_state(
        adapter, component, alpha, cfg, seed=seed
    )
    eta_draws = (
        alpha.grid_offset[np.newaxis, :]
        + state.coefficient_draws @ state.standardized_evidence.T
    )
    return PriorPredictiveEvidenceDraws(
        probability_draws=expit(eta_draws),
        coefficient_draws=state.coefficient_draws,
        feature_names=state.feature_names,
        diagnostics=state.diagnostics,
    )


def _resolved_evidence_prior(
    cfg: ProbabilisticConfig,
    layer_names: list[str],
    *,
    component: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resolve play-type defaults and explicit proper coefficient priors."""
    base_precision = 1.0 / cfg.evidence.regularization.C
    default_precision: dict[str, float] = {}
    default_means: dict[str, float] = {}
    play_type = cfg.evidence.regularization.play_type
    if play_type:
        from .play_types import play_type_defaults  # noqa: PLC0415

        defaults = play_type_defaults(play_type, layer_names=layer_names)
        default_precision.update(defaults["per_feature_weights"])
        default_means.update(defaults["prior_means"])
    default_precision.update(cfg.evidence.regularization.per_feature_weights)
    default_precision.update(cfg.evidence.regularization.prior_precisions)
    default_means.update(cfg.evidence.regularization.prior_means)
    precision = _qualified_prior_values(
        default_precision,
        component=component,
        layer_names=layer_names,
        prior_name="precision",
        require_explicit=False,
        default=base_precision,
    )
    prior_mean = _qualified_prior_values(
        default_means,
        component=component,
        layer_names=layer_names,
        prior_name="mean",
        require_explicit=False,
        default=0.0,
    )
    if (
        not np.all(np.isfinite(precision))
        or np.any(precision <= 0.0)
        or not np.all(np.isfinite(prior_mean))
    ):
        raise GEOPFAValueError(
            "GBLK evidence priors require strictly positive finite precisions "
            "and finite means"
        )
    return precision, prior_mean


def _standardize_partial_evidence(  # noqa: PLR0913
    x_train: NDArray[np.float64],
    x_prediction: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    component_name: str,
    layer_names: list[str],
    minimum_count: int,
    standardization: str = "observed_labels",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    int,
]:
    """Apply fold-local per-feature scaling with neutral mean imputation."""
    finite_observed = observed[:, np.newaxis] & np.isfinite(x_train)
    finite_counts = finite_observed.sum(axis=0).astype(np.int64)
    unsupported = np.flatnonzero(finite_counts < minimum_count)
    if unsupported.size:
        details = ", ".join(
            f"{layer_names[index]}={int(finite_counts[index])}"
            for index in unsupported
        )
        raise GEOPFAValueError(
            f"component {component_name!r} has fewer than {minimum_count} "
            f"finite observed values for evidence layer(s): {details}"
        )
    if standardization == "observed_labels":
        masked = np.where(finite_observed, x_train, np.nan)
    elif standardization == "prediction_support":
        masked = np.where(np.isfinite(x_prediction), x_prediction, np.nan)
        unsupported_prediction = np.flatnonzero(
            np.sum(np.isfinite(masked), axis=0) == 0
        )
        if unsupported_prediction.size:
            names = ", ".join(
                layer_names[index] for index in unsupported_prediction
            )
            raise GEOPFAValueError(
                f"component {component_name!r} has no finite prediction "
                f"support for evidence layer(s): {names}"
            )
    else:  # pragma: no cover - guarded by EvidenceConfig
        raise GEOPFAValueError(
            f"unknown evidence standardization {standardization!r}"
        )
    center = np.nanmean(masked, axis=0)
    scale = np.nanstd(masked, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    train_scaled = (
        np.where(np.isfinite(x_train), x_train, center) - center
    ) / scale
    prediction_scaled = (
        np.where(np.isfinite(x_prediction), x_prediction, center) - center
    ) / scale
    complete_count = int(
        np.sum(observed & np.all(np.isfinite(x_train), axis=1))
    )
    return (
        train_scaled,
        prediction_scaled,
        center,
        scale,
        finite_counts,
        complete_count,
    )


def _prepare_joint_evidence_arrays(  # noqa: PLR0913, PLR0914
    *,
    component_names: tuple[str, ...],
    train_evidence: dict[str, NDArray[np.float64]],
    prediction_evidence: dict[str, NDArray[np.float64]],
    layer_names: dict[str, list[str]],
    y_train: NDArray[np.float64],
    observed_train: NDArray[np.bool_],
    cfg: ProbabilisticConfig,
) -> JointEvidenceDesign:
    """Prepare leakage-safe component designs for one joint model fit."""
    n_train, n_components = y_train.shape
    if component_names != tuple(train_evidence) or n_components != len(
        component_names
    ):
        raise GEOPFAValueError(
            "joint evidence component ordering is inconsistent"
        )
    n_prediction = len(prediction_evidence[component_names[0]])
    widths = [len(layer_names[name]) for name in component_names]
    max_width = max(widths, default=0)
    diagnostics: dict[str, dict[str, Any]] = {}
    for q_idx, name in enumerate(component_names):
        observed = observed_train[:, q_idx]
        if observed.sum() < cfg.labels.min_wells_for_fit:
            raise GEOPFAValueError(
                f"component {name!r} has fewer than "
                f"{cfg.labels.min_wells_for_fit} observed labels"
            )
        if (
            not cfg.inference.gblk_bayesian.enabled
            and np.unique(y_train[observed, q_idx]).size < 2  # noqa: PLR2004
        ):
            raise GEOPFAValueError(
                f"component {name!r} has only one observed outcome class"
            )
    if max_width == 0:
        for name in component_names:
            diagnostics[name] = {
                "evidence_stage": "offset_only_no_evidence",
                "evidence_beta": [],
                "evidence_coefficients_frozen_before_spatial_fit": False,
            }
        return JointEvidenceDesign(None, None, None, None, diagnostics)

    train_design = np.zeros(
        (n_train, max_width, n_components), dtype=np.float64
    )
    prediction_design = np.zeros(
        (n_prediction, max_width, n_components), dtype=np.float64
    )
    base_precision = 1.0 / cfg.evidence.regularization.C
    precision = np.full(
        (max_width, n_components), base_precision, dtype=np.float64
    )
    prior_mean = np.zeros((max_width, n_components), dtype=np.float64)

    for q_idx, name in enumerate(component_names):
        x_train = np.asarray(train_evidence[name], dtype=np.float64)
        x_prediction = np.asarray(prediction_evidence[name], dtype=np.float64)
        width = widths[q_idx]
        if x_train.shape != (n_train, width):
            raise GEOPFAValueError(
                f"component {name!r} training evidence shape is inconsistent"
            )
        if x_prediction.shape != (n_prediction, width):
            raise GEOPFAValueError(
                f"component {name!r} prediction evidence shape is inconsistent"
            )
        observed = observed_train[:, q_idx]
        if width == 0:
            diagnostics[name] = {
                "evidence_stage": "offset_only_no_evidence",
                "evidence_beta": [],
                "evidence_coefficients_frozen_before_spatial_fit": False,
            }
            continue
        (
            train_scaled,
            prediction_scaled,
            center,
            scale,
            finite_counts,
            complete_count,
        ) = _standardize_partial_evidence(
            x_train,
            x_prediction,
            observed,
            component_name=name,
            layer_names=layer_names[name],
            minimum_count=cfg.labels.min_wells_for_fit,
            standardization=cfg.evidence.standardization,
        )
        component_precision, component_mean = _resolved_evidence_prior(
            cfg, layer_names[name], component=name
        )
        train_design[:, :width, q_idx] = train_scaled
        prediction_design[:, :width, q_idx] = prediction_scaled
        precision[:width, q_idx] = component_precision
        prior_mean[:width, q_idx] = component_mean
        diagnostics[name] = {
            "evidence_stage": "joint_fixed_effect_and_spatial_laplace",
            "evidence_beta": None,
            "evidence_center": center.tolist(),
            "evidence_scale": scale.tolist(),
            "evidence_prior_precision": component_precision.tolist(),
            "evidence_prior_mean": component_mean.tolist(),
            "evidence_standardization": cfg.evidence.standardization,
            "evidence_n": int(observed.sum()),
            "evidence_complete_row_n": complete_count,
            "evidence_finite_per_feature": finite_counts.tolist(),
            "evidence_missing_value_policy": (
                "fold_local_feature_mean_zero_standardized_contribution"
            ),
            "evidence_coefficients_frozen_before_spatial_fit": False,
        }
    return JointEvidenceDesign(
        train_design,
        prediction_design,
        precision,
        prior_mean,
        diagnostics,
    )


def _prepare_joint_evidence(
    assembled: AssembledInputs,
    cfg: ProbabilisticConfig,
) -> JointEvidenceDesign:
    """Prepare full-data evidence inputs for a joint spatial fit."""
    return _prepare_joint_evidence_arrays(
        component_names=assembled.component_names,
        train_evidence=assembled.evidence,
        prediction_evidence=assembled.grid_evidence,
        layer_names=assembled.layer_names,
        y_train=assembled.y,
        observed_train=assembled.observed_mask,
        cfg=cfg,
    )


def _fit_evidence_only_offsets(  # noqa: PLR0914
    assembled: AssembledInputs,
    cfg: ProbabilisticConfig,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], dict[str, dict[str, Any]]
]:
    """Fit the explicit evidence-only ablation with no spatial field."""
    well_offsets = assembled.well_offsets.copy()
    grid_offsets = assembled.grid_offsets.copy()
    diagnostics: dict[str, dict[str, Any]] = {}

    for q_idx, name in enumerate(assembled.component_names):
        x_well = np.asarray(assembled.evidence[name], dtype=np.float64)
        x_grid = np.asarray(assembled.grid_evidence[name], dtype=np.float64)
        observed = assembled.observed_mask[:, q_idx]
        if observed.sum() < cfg.labels.min_wells_for_fit:
            raise GEOPFAValueError(
                f"component {name!r} has fewer than "
                f"{cfg.labels.min_wells_for_fit} observed labels"
            )
        if np.unique(assembled.y[observed, q_idx]).size < 2:  # noqa: PLR2004
            raise GEOPFAValueError(
                f"component {name!r} has only one observed outcome class"
            )
        if x_well.shape[1] == 0:
            diagnostics[name] = {
                "evidence_stage": "offset_only_no_evidence",
                "evidence_beta": [],
            }
            continue

        (
            x_well_scaled,
            x_grid_scaled,
            mean,
            scale,
            finite_counts,
            complete_count,
        ) = _standardize_partial_evidence(
            x_well,
            x_grid,
            observed,
            component_name=name,
            layer_names=assembled.layer_names[name],
            minimum_count=cfg.labels.min_wells_for_fit,
            standardization=cfg.evidence.standardization,
        )
        component_y = assembled.y[observed, q_idx]

        layer_names = assembled.layer_names[name]
        weights, prior_means = _resolved_evidence_prior(
            cfg, layer_names, component=name
        )
        fit = _fit_offset_logit(
            x_well_scaled[observed],
            component_y,
            well_offsets[observed, q_idx],
            regularization=1.0 / cfg.evidence.regularization.C,
            per_feature_weights=weights,
            prior_means=prior_means,
        )
        beta = np.asarray(fit.x, dtype=np.float64)
        well_offsets[:, q_idx] += x_well_scaled @ beta
        grid_offsets[:, q_idx] += x_grid_scaled @ beta
        diagnostics[name] = {
            "evidence_stage": "penalized_offset_logit_no_spatial_field",
            "evidence_beta": beta.tolist(),
            "evidence_center": mean.tolist(),
            "evidence_scale": scale.tolist(),
            "evidence_prior_precision": weights.tolist(),
            "evidence_prior_mean": prior_means.tolist(),
            "evidence_standardization": cfg.evidence.standardization,
            "evidence_optimizer_success": bool(fit.success),
            "evidence_n": int(observed.sum()),
            "evidence_complete_row_n": complete_count,
            "evidence_finite_per_feature": finite_counts.tolist(),
            "evidence_missing_value_policy": (
                "fold_local_feature_mean_zero_standardized_contribution"
            ),
        }
    return well_offsets, grid_offsets, diagnostics


def _run_gblk_bayesian(  # noqa: PLR0913
    assembled: AssembledInputs,
    grid_gdf: gpd.GeoDataFrame,
    bayes_cfg: GBLKBayesianConfig,
    *,
    well_offsets: NDArray[np.float64],
    grid_offsets: NDArray[np.float64],
    evidence_diagnostics: dict[str, dict[str, Any]],
    evidence_design: JointEvidenceDesign,
    nc: int,
    nlevel: int,
    a_wght: float,
    coordinate_scaling: str,
) -> tuple[
    dict[str, ComponentProbability],
    gpd.GeoDataFrame,
    dict[str, NDArray[np.float64]],
]:
    """Package predictions from the sole array-level Bayesian GBLK fitter."""
    fit_result = fit_gblk_bayesian_joint(
        assembled.well_coords,
        assembled.y,
        assembled.grid_coords,
        component_names=assembled.component_names,
        bayes_config=bayes_cfg,
        observed_mask=assembled.observed_mask,
        offsets=well_offsets,
        grid_offsets=grid_offsets,
        fixed_effects=evidence_design.train,
        fixed_effects_grid=evidence_design.prediction,
        fixed_precision=evidence_design.precision,
        fixed_prior_mean=evidence_design.prior_mean,
        nc=nc,
        nlevel=nlevel,
        a_wght=a_wght,
        coordinate_scaling=coordinate_scaling,
    )
    diagnostics = dict(fit_result.diagnostics)

    components: dict[str, ComponentProbability] = {}
    component_draws: dict[str, NDArray[np.float64]] = {}
    for q_idx, name in enumerate(assembled.component_names):
        component_diagnostics = dict(evidence_diagnostics[name])
        width = len(assembled.layer_names.get(name, []))
        if fit_result.fixed_coef_draws is not None and width:
            coefficient_draws = fit_result.fixed_coef_draws[:, :width, q_idx]
            tail = (1.0 - bayes_cfg.ci_level) / 2.0
            component_diagnostics["evidence_beta"] = coefficient_draws.mean(
                axis=0
            ).tolist()
            component_diagnostics["evidence_beta_interval"] = np.quantile(
                coefficient_draws, [tail, 1.0 - tail], axis=0
            ).tolist()
        prob_gdf = grid_gdf[["geometry"]].copy()
        prob_gdf = prob_gdf.assign(
            probability=fit_result.p_q_grid[:, q_idx].astype(float),
            probability_lo=fit_result.p_q_interval[0, :, q_idx].astype(float),
            probability_hi=fit_result.p_q_interval[1, :, q_idx].astype(float),
        )
        components[name] = ComponentProbability(
            probability=prob_gdf,
            model=fit_result.fit,
            feature_names=tuple(assembled.layer_names.get(name, [])),
            diagnostics={**diagnostics, **component_diagnostics},
        )
        component_draws[name] = fit_result.p_q_draws[:, :, q_idx]

    combined = grid_gdf[["geometry"]].copy()
    combined = combined.assign(
        probability=fit_result.p_joint_grid.astype(float),
        probability_lo=fit_result.p_joint_interval[0].astype(float),
        probability_hi=fit_result.p_joint_interval[1].astype(float),
    )

    return components, combined, component_draws


def _gaussian_predictive_exceedance_draws(
    fit_result: Any,
    *,
    component_index: int,
    threshold_scaled: float,
) -> NDArray[np.float64]:
    """Return matched posterior predictive exceedance probabilities."""
    mean_draws = np.asarray(
        fit_result.response_draws[:, :, component_index], dtype=np.float64
    )
    precision_draws = np.asarray(
        fit_result.likelihood_precision_draws[:, component_index],
        dtype=np.float64,
    )
    if (
        precision_draws.shape != (mean_draws.shape[0],)
        or not np.all(np.isfinite(precision_draws))
        or np.any(precision_draws <= 0.0)
    ):
        raise RuntimeError(
            "Gaussian prediction requires one positive likelihood-precision "
            "draw per latent-mean draw"
        )
    probabilities = ndtr(
        (mean_draws - threshold_scaled)
        * np.sqrt(precision_draws)[:, np.newaxis]
    )
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "Gaussian predictive exceedance produced nonfinite probabilities"
        )
    return probabilities


def _run_gblk_gaussian_bayesian(  # noqa: PLR0913
    assembled: AssembledInputs,
    grid_gdf: gpd.GeoDataFrame,
    alphas: Mapping[str, AlphaCResult],
    cfg: ProbabilisticConfig,
    bayes_cfg: GBLKBayesianConfig,
    *,
    evidence_design: JointEvidenceDesign,
    nc: int,
    nlevel: int,
    a_wght: float,
    coordinate_scaling: str,
) -> tuple[
    dict[str, ComponentProbability],
    dict[str, NDArray[np.float64]],
]:
    """Fit continuous components and convert draws to event probabilities."""
    fit_result = fit_gblk_gaussian_bayesian_joint(
        assembled.well_coords,
        assembled.y,
        assembled.grid_coords,
        component_names=assembled.component_names,
        bayes_config=bayes_cfg,
        observed_mask=assembled.observed_mask,
        offsets=assembled.well_offsets,
        grid_offsets=assembled.grid_offsets,
        fixed_effects=evidence_design.train,
        fixed_effects_grid=evidence_design.prediction,
        fixed_precision=evidence_design.precision,
        fixed_prior_mean=evidence_design.prior_mean,
        nc=nc,
        nlevel=nlevel,
        a_wght=a_wght,
        coordinate_scaling=coordinate_scaling,
    )
    tail = (1.0 - bayes_cfg.ci_level) / 2.0
    components: dict[str, ComponentProbability] = {}
    component_draws: dict[str, NDArray[np.float64]] = {}
    for q_idx, name in enumerate(assembled.component_names):
        observation = cfg.labels.observation_model_for(name)
        scale = float(observation.response_scale)
        alpha = alphas[name]
        if alpha.event_threshold is None:
            raise RuntimeError(
                f"Gaussian component {name!r} has no event threshold"
            )
        response_draws = fit_result.response_draws[:, :, q_idx] * scale
        event_draws = _gaussian_predictive_exceedance_draws(
            fit_result,
            component_index=q_idx,
            threshold_scaled=float(alpha.event_threshold) / scale,
        )
        probability_interval = np.quantile(
            event_draws, [tail, 1.0 - tail], axis=0
        )
        response_interval = np.quantile(
            response_draws, [tail, 1.0 - tail], axis=0
        )
        probability = (
            grid_gdf[["geometry"]]
            .copy()
            .assign(
                probability=event_draws.mean(axis=0),
                probability_lo=probability_interval[0],
                probability_hi=probability_interval[1],
                response_mean=response_draws.mean(axis=0),
                response_lo=response_interval[0],
                response_hi=response_interval[1],
            )
        )
        diagnostics = {
            **fit_result.diagnostics,
            **evidence_design.diagnostics[name],
            "observation_family": "gaussian",
            "response_scale": scale,
            "event_threshold": float(alpha.event_threshold),
            "event_probability_estimand": (
                "posterior_predictive_response_exceedance"
            ),
            "likelihood_sd_mean": float(
                np.mean(
                    scale
                    / np.sqrt(fit_result.likelihood_precision_draws[:, q_idx])
                )
            ),
        }
        width = len(assembled.layer_names.get(name, []))
        if fit_result.fixed_coef_draws is not None and width:
            coefficient_draws = (
                fit_result.fixed_coef_draws[:, :width, q_idx] * scale
            )
            diagnostics["evidence_beta"] = coefficient_draws.mean(
                axis=0
            ).tolist()
            diagnostics["evidence_beta_interval"] = np.quantile(
                coefficient_draws, [tail, 1.0 - tail], axis=0
            ).tolist()
        components[name] = ComponentProbability(
            probability=probability,
            model=fit_result.fit,
            feature_names=tuple(assembled.layer_names.get(name, [])),
            diagnostics=diagnostics,
        )
        component_draws[name] = event_draws
    return components, component_draws


def _assemble_likelihood_groups(
    adapter: PFAGridAdapter,
    loaded_labels: LoadedLabels,
    fit_alphas: Mapping[str, AlphaCResult],
    cfg: ProbabilisticConfig,
) -> dict[str, AssembledInputs]:
    """Assemble separate same-family fits for one mixed-response workflow."""
    grouped_names: dict[str, list[str]] = {}
    for name in fit_alphas:
        family = cfg.labels.observation_model_for(name).family
        grouped_names.setdefault(family, []).append(name)

    assembled_groups: dict[str, AssembledInputs] = {}
    for family, names in grouped_names.items():
        group_labels = replace(
            loaded_labels.config,
            label_columns={
                name: loaded_labels.config.label_columns[name]
                for name in names
            },
            observation_models={
                name: loaded_labels.config.observation_models[name]
                for name in names
                if name in loaded_labels.config.observation_models
            },
        )
        group_alphas = {name: fit_alphas[name] for name in names}
        if family == "gaussian":
            group_alphas = {}
            for name in names:
                alpha = fit_alphas[name]
                if alpha.latent_mean is None:
                    raise GEOPFAValueError(
                        f"Gaussian component {name!r} requires a continuous "
                        "thermal prior mean"
                    )
                scale = float(
                    cfg.labels.observation_model_for(name).response_scale
                )
                group_alphas[name] = replace(
                    alpha,
                    grid_offset=np.asarray(alpha.latent_mean) / scale,
                )
        assembled = assemble_gblk_inputs(
            adapter,
            LoadedLabels(gdf=loaded_labels.gdf, config=group_labels),
            group_alphas,
            evidence_config=cfg.evidence,
            prior_probability_results={
                name: fit_alphas[name] for name in names
            },
        )
        if family == "gaussian":
            scales = np.asarray(
                [
                    cfg.labels.observation_model_for(name).response_scale
                    for name in assembled.component_names
                ],
                dtype=np.float64,
            )
            assembled = replace(assembled, y=assembled.y / scales)
        assembled_groups[family] = assembled
    return assembled_groups


def _blocked_family_predictions(
    assembled: AssembledInputs,
    family: str,
    cfg: ProbabilisticConfig,
    *,
    nc: int,
    a_wght: float | None,
) -> NDArray[np.float64]:
    """Return leakage-safe out-of-fold event probabilities for one family."""
    predictions = np.full(assembled.y.shape, np.nan, dtype=np.float64)
    folds = spatial_block_cv(
        assembled.well_coords,
        n_folds=cfg.cross_validation.n_folds,
        block_type=cfg.cross_validation.block_type,
        grid_size=cfg.cross_validation.grid_size,
        seed=cfg.inference.gblk_bayesian.seed,
        block_size_km=cfg.cross_validation.block_size_km,
        buffer_distance=cfg.cross_validation.buffer_km * 1000.0,
        dims=(0, 1),
    )
    fold_seeds = _spawn_child_seeds(
        cfg.inference.gblk_bayesian.seed,
        cfg.cross_validation.n_folds,
    )
    for (train_mask, test_mask), fold_seed in zip(
        folds, fold_seeds, strict=True
    ):
        fold_cfg = replace(
            cfg.inference.gblk_bayesian,
            seed=fold_seed,
        )
        evidence_design = _prepare_joint_evidence_arrays(
            component_names=assembled.component_names,
            train_evidence={
                name: assembled.evidence[name][train_mask]
                for name in assembled.component_names
            },
            prediction_evidence={
                name: assembled.evidence[name][test_mask]
                for name in assembled.component_names
            },
            layer_names=assembled.layer_names,
            y_train=assembled.y[train_mask],
            observed_train=assembled.observed_mask[train_mask],
            cfg=cfg,
        )
        common = {
            "component_names": assembled.component_names,
            "bayes_config": fold_cfg,
            "observed_mask": assembled.observed_mask[train_mask],
            "offsets": assembled.well_offsets[train_mask],
            "grid_offsets": assembled.well_offsets[test_mask],
            "fixed_effects": evidence_design.train,
            "fixed_effects_grid": evidence_design.prediction,
            "fixed_precision": evidence_design.precision,
            "fixed_prior_mean": evidence_design.prior_mean,
            "nc": nc,
            "nlevel": cfg.spatial_field.n_levels,
            "a_wght": _resolve_a_wght(assembled.well_coords.shape[1], a_wght),
            "coordinate_scaling": cfg.spatial_field.coordinate_scaling,
        }
        if family == "bernoulli":
            fit = fit_gblk_bayesian_joint(
                assembled.well_coords[train_mask],
                assembled.y[train_mask],
                assembled.well_coords[test_mask],
                **common,
            )
            predictions[test_mask] = fit.p_q_draws.mean(axis=0)
        elif family == "gaussian":
            fit = fit_gblk_gaussian_bayesian_joint(
                assembled.well_coords[train_mask],
                assembled.y[train_mask],
                assembled.well_coords[test_mask],
                **common,
            )
            for q_idx, name in enumerate(assembled.component_names):
                scale = float(
                    cfg.labels.observation_model_for(name).response_scale
                )
                threshold = cfg.alpha[name].threshold
                predictions[test_mask, q_idx] = (
                    _gaussian_predictive_exceedance_draws(
                        fit,
                        component_index=q_idx,
                        threshold_scaled=float(threshold) / scale,
                    ).mean(axis=0)
                )
        else:  # pragma: no cover - guarded by observation config
            raise AssertionError(f"unexpected response family {family!r}")
    return predictions


def _estimate_predictive_stacking(
    assembled_groups: Mapping[str, AssembledInputs],
    cfg: ProbabilisticConfig,
    *,
    nc: int,
    a_wght: float | None,
) -> dict[str, PredictiveStackingResult]:
    """Select one prior/update mixture weight per fitted component."""
    results: dict[str, PredictiveStackingResult] = {}
    for family, assembled in assembled_groups.items():
        if assembled.prior_probability_well is None:
            raise RuntimeError(
                "predictive stacking requires prior probabilities at wells"
            )
        full_probability = _blocked_family_predictions(
            assembled, family, cfg, nc=nc, a_wght=a_wght
        )
        for q_idx, name in enumerate(assembled.component_names):
            observed = assembled.observed_mask[:, q_idx]
            if family == "bernoulli":
                outcomes = assembled.y[observed, q_idx]
            else:
                scale = float(
                    cfg.labels.observation_model_for(name).response_scale
                )
                outcomes = (
                    assembled.y[observed, q_idx] * scale
                    > cfg.alpha[name].threshold
                ).astype(np.float64)
            if not np.all(np.isfinite(full_probability[observed, q_idx])):
                raise RuntimeError(
                    f"component {name!r} has missing out-of-fold predictions"
                )
            results[name] = select_predictive_stacking_weight(
                outcomes,
                assembled.prior_probability_well[observed, q_idx],
                full_probability[observed, q_idx],
            )
    return results


def _apply_componentwise_stacking(
    components: dict[str, ComponentProbability],
    component_draws: dict[str, NDArray[np.float64]],
    assembled_groups: Mapping[str, AssembledInputs],
    stacking: Mapping[str, PredictiveStackingResult],
    *,
    ci_level: float,
) -> None:
    """Apply selected component weights before the existing combination."""
    tail = (1.0 - ci_level) / 2.0
    for assembled in assembled_groups.values():
        if assembled.prior_probability_grid is None:
            raise RuntimeError(
                "predictive stacking requires prior probabilities on the grid"
            )
        for q_idx, name in enumerate(assembled.component_names):
            selection = stacking[name]
            draws = apply_predictive_stacking(
                assembled.prior_probability_grid[:, q_idx],
                component_draws[name],
                weight=selection.weight,
            )
            component_draws[name] = draws
            interval = np.quantile(draws, [tail, 1.0 - tail], axis=0)
            probability = components[name].probability.copy()
            probability["probability"] = draws.mean(axis=0)
            probability["probability_lo"] = interval[0]
            probability["probability_hi"] = interval[1]
            diagnostics = {
                **components[name].diagnostics,
                "predictive_stacking_weight": selection.weight,
                "predictive_stacking_prior_log_score": (
                    selection.prior_log_score
                ),
                "predictive_stacking_full_log_score": selection.full_log_score,
                "predictive_stacking_selected_log_score": (
                    selection.selected_log_score
                ),
                "predictive_stacking_n": selection.n_observations,
                "predictive_stacking_validation": "blocked_out_of_fold",
            }
            components[name] = ComponentProbability(
                probability=probability,
                model=components[name].model,
                feature_names=components[name].feature_names,
                diagnostics=diagnostics,
            )


def _restore_streaming_posterior_states(
    persisted: PersistedPosteriorDrawState,
    assembled: AssembledInputs | None,
) -> tuple[
    Any | None,
    dict[str, PriorPredictiveEvidenceState | None],
    dict[str, dict[str, Any]],
]:
    """Reconstruct projection objects from immutable persisted arrays."""
    from latticekrigx.model.config import LKInfo  # noqa: PLC0415

    from geopfa.prob.gblk_backend import (  # noqa: PLC0415
        GBLKBayesianPosteriorState,
    )

    metadata = persisted.metadata
    fitted_names = tuple(metadata.get("fitted_component_names", ()))
    fitted_state = None
    evidence_diagnostics = dict(metadata.get("evidence_diagnostics", {}))
    if fitted_names:
        if assembled is None or fitted_names != assembled.component_names:
            raise ValueError(
                "persisted fitted-component ordering differs from this run"
            )
        fitted_state = GBLKBayesianPosteriorState(
            component_names=fitted_names,
            coefficient_draws=np.asarray(
                persisted.arrays["field_coefficient_draws"]
            ),
            fixed_coef_draws=(
                None
                if "fixed_coefficient_draws" not in persisted.arrays
                else np.asarray(persisted.arrays["fixed_coefficient_draws"])
            ),
            grid_model=np.asarray(persisted.arrays["fitted_grid_model"]),
            grid_offsets=np.asarray(persisted.arrays["fitted_grid_offsets"]),
            fixed_design_grid=(
                None
                if "fixed_design_grid" not in persisted.arrays
                else np.asarray(persisted.arrays["fixed_design_grid"])
            ),
            lkinfo=LKInfo.model_validate(metadata["lkinfo"]),
            fit=PersistedBayesianFitReference(
                state_fingerprint=persisted.state_fingerprint,
                posterior_draw_index=persisted.index_path,
            ),
            diagnostics=dict(metadata["fitted_diagnostics"]),
        )

    prior_states: dict[str, PriorPredictiveEvidenceState | None] = {}
    roles = metadata["component_roles"]
    prior_metadata = metadata.get("prior_component_state", {})
    for name, role in roles.items():
        if role == "joint_posterior":
            continue
        if role == "fixed_prior_predictive":
            prior_states[name] = None
            continue
        if role != "evidence_coefficient_prior_predictive":
            raise ValueError(f"unknown persisted component role {role!r}")
        component_metadata = prior_metadata[name]
        prior_states[name] = PriorPredictiveEvidenceState(
            standardized_evidence=np.asarray(
                persisted.arrays[component_metadata["evidence_array"]]
            ),
            coefficient_draws=np.asarray(
                persisted.arrays[component_metadata["coefficient_array"]]
            ),
            feature_names=tuple(component_metadata["feature_names"]),
            diagnostics=dict(component_metadata["diagnostics"]),
        )
    return fitted_state, prior_states, evidence_diagnostics


def _run_gblk_bayesian_streaming(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
    assembled: AssembledInputs | None,
    grid_gdf: gpd.GeoDataFrame,
    adapter: PFAGridAdapter,
    alphas: Mapping[str, AlphaCResult],
    prior_only_names: tuple[str, ...],
    cfg: ProbabilisticConfig,
    *,
    evidence_design: JointEvidenceDesign | None,
    nc: int,
    a_wght: float | None,
    scope: str,
) -> ProbabilisticResult:
    """Fit once and persist paired grid draws without a full draw cube."""
    bayes_cfg = cfg.inference.gblk_bayesian
    ordered_names = tuple(sorted(alphas))
    component_index = {name: index for index, name in enumerate(ordered_names)}
    n_draws = bayes_cfg.n_draws
    n_cells = len(grid_gdf)
    q_total = len(ordered_names)
    prior_logit = np.column_stack(
        [
            np.asarray(alphas[name].grid_offset, dtype=np.float64)
            for name in ordered_names
        ]
    )
    if prior_logit.shape != (n_cells, q_total) or not np.all(
        np.isfinite(prior_logit)
    ):
        raise RuntimeError(
            "Bayesian component priors do not share one finite grid"
        )

    prior_names = tuple(sorted(prior_only_names))
    config_hash = hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    persisted = load_posterior_draw_state(
        cfg.output_dir,
        grid_gdf,
        expected_config_hash=config_hash,
        expected_scope=scope,
    )
    if persisted is not None:
        if persisted.component_names != ordered_names:
            raise ValueError(
                "persisted posterior component ordering differs from this run"
            )
        state_arrays = dict(persisted.arrays)
        state_metadata = dict(persisted.metadata)
        if not np.array_equal(state_arrays["prior_logit"], prior_logit):
            raise ValueError("persisted prior logits differ from this run")
        fitted_state, prior_states, evidence_diagnostics = (
            _restore_streaming_posterior_states(persisted, assembled)
        )
    else:
        fitted_state = None
        evidence_diagnostics: dict[str, dict[str, Any]] = {}
        if assembled is not None:
            if evidence_design is None:
                raise RuntimeError(
                    "fitted Bayesian components require an evidence design"
                )
            model_a_wght = _resolve_a_wght(
                assembled.well_coords.shape[1], a_wght
            )
            fitted_state = fit_gblk_bayesian_posterior_state(
                assembled.well_coords,
                assembled.y,
                assembled.grid_coords,
                component_names=assembled.component_names,
                bayes_config=bayes_cfg,
                observed_mask=assembled.observed_mask,
                offsets=assembled.well_offsets,
                grid_offsets=assembled.grid_offsets,
                fixed_effects=evidence_design.train,
                fixed_effects_grid=evidence_design.prediction,
                fixed_precision=evidence_design.precision,
                fixed_prior_mean=evidence_design.prior_mean,
                nc=nc,
                nlevel=cfg.spatial_field.n_levels,
                a_wght=model_a_wght,
                coordinate_scaling=cfg.spatial_field.coordinate_scaling,
            )
            evidence_diagnostics = evidence_design.diagnostics

        prior_states = {}
        prior_seeds = _spawn_child_seeds(bayes_cfg.seed, len(prior_names))
        for name, component_seed in zip(prior_names, prior_seeds, strict=True):
            prior_grid = adapter.pr_norm(name)
            if not prior_grid.geometry.equals(grid_gdf.geometry):
                raise GEOPFAValueError(
                    f"component {name!r} prior grid does not match the Bayesian prediction grid"
                )
            if cfg.alpha[name].use_evidence_prior:
                prior_states[name] = _prior_predictive_evidence_state(
                    adapter, name, alphas[name], cfg, seed=component_seed
                )
            else:
                prior_states[name] = None

        state_arrays = {"prior_logit": prior_logit}
        state_metadata = {
            "scope": scope,
            "config_hash": config_hash,
            "ci_level": bayes_cfg.ci_level,
            "component_roles": {},
            "fitted_component_names": (
                []
                if fitted_state is None
                else list(fitted_state.component_names)
            ),
            "coordinate_transform": cfg.spatial_field.coordinate_scaling,
            "prediction_cell_chunk_size": 10_000,
            "draw_block_size": cfg.outputs.posterior_draw_block_size,
            "cross_scenario_pairing": (
                "within_scope_only"
                if scope == "baseline"
                else "not_identified"
            ),
        }
        if fitted_state is not None:
            state_arrays["field_coefficient_draws"] = (
                fitted_state.coefficient_draws
            )
            state_arrays["fitted_grid_model"] = fitted_state.grid_model
            state_arrays["fitted_grid_offsets"] = fitted_state.grid_offsets
            if fitted_state.fixed_coef_draws is not None:
                state_arrays["fixed_coefficient_draws"] = (
                    fitted_state.fixed_coef_draws
                )
            if fitted_state.fixed_design_grid is not None:
                state_arrays["fixed_design_grid"] = (
                    fitted_state.fixed_design_grid
                )
            state_metadata["lkinfo"] = fitted_state.lkinfo.model_dump(
                mode="json"
            )
            state_metadata["fitted_diagnostics"] = fitted_state.diagnostics
            state_metadata["evidence_diagnostics"] = evidence_diagnostics
            for name in fitted_state.component_names:
                state_metadata["component_roles"][name] = "joint_posterior"
        for prior_index, name in enumerate(prior_names):
            prior_state = prior_states[name]
            if prior_state is None:
                state_metadata["component_roles"][name] = (
                    "fixed_prior_predictive"
                )
                continue
            state_arrays[f"prior_coefficient_draws_{prior_index}"] = (
                prior_state.coefficient_draws
            )
            state_arrays[f"prior_standardized_evidence_{prior_index}"] = (
                prior_state.standardized_evidence
            )
            state_metadata["component_roles"][name] = (
                "evidence_coefficient_prior_predictive"
            )
            state_metadata.setdefault("prior_component_state", {})[name] = {
                "coefficient_array": f"prior_coefficient_draws_{prior_index}",
                "evidence_array": f"prior_standardized_evidence_{prior_index}",
                "feature_names": list(prior_state.feature_names),
                "diagnostics": prior_state.diagnostics,
            }

        roles = set(state_metadata["component_roles"].values())
        if fitted_state is None:
            state_metadata["model"] = "geopfa_probabilistic_prior_predictive"
        elif roles == {"joint_posterior"}:
            state_metadata["model"] = "geopfa_probabilistic_gblk_paige_inla"
        else:
            state_metadata["model"] = (
                "geopfa_probabilistic_gblk_paige_inla_with_"
                "prior_predictive_components"
            )

    roles = set(state_metadata["component_roles"].values())
    expected_model = (
        "geopfa_probabilistic_prior_predictive"
        if "joint_posterior" not in roles
        else (
            "geopfa_probabilistic_gblk_paige_inla"
            if roles == {"joint_posterior"}
            else (
                "geopfa_probabilistic_gblk_paige_inla_with_"
                "prior_predictive_components"
            )
        )
    )
    if state_metadata.get("model") != expected_model:
        raise ValueError(
            "persisted inference model is inconsistent with component roles; "
            "remove the stale generated draw directory and rerun"
        )

    writer = PosteriorDrawBlockWriter(
        grid_gdf,
        cfg.output_dir,
        component_names=ordered_names,
        n_draws=n_draws,
        block_size=cfg.outputs.posterior_draw_block_size,
        seed=bayes_cfg.seed,
        combination_rule=cfg.combination.rule,
        scope=scope,
        state_arrays=state_arrays,
        state_metadata=state_metadata,
    )
    completed_starts = {start for start, _ in writer.completed_draw_ranges}
    for draw_start in range(0, n_draws, cfg.outputs.posterior_draw_block_size):
        if draw_start in completed_starts:
            continue
        draw_stop = min(
            draw_start + cfg.outputs.posterior_draw_block_size, n_draws
        )
        block_shape = (draw_stop - draw_start, n_cells, q_total)
        evidence_logit = np.zeros(block_shape, dtype=np.float64)
        spatial_logit = np.zeros(block_shape, dtype=np.float64)
        if fitted_state is not None:
            fitted_block = project_gblk_bayesian_draw_block(
                fitted_state, draw_start, draw_stop
            )
            for source_index, name in enumerate(fitted_state.component_names):
                target_index = component_index[name]
                if not np.array_equal(
                    fitted_block.prior_logit[:, source_index],
                    prior_logit[:, target_index],
                ):
                    raise RuntimeError(
                        f"component {name!r} fitted and assembled prior logits differ"
                    )
                evidence_logit[:, :, target_index] = (
                    fitted_block.evidence_logit[:, :, source_index]
                )
                spatial_logit[:, :, target_index] = fitted_block.spatial_logit[
                    :, :, source_index
                ]
        for name in prior_names:
            prior_state = prior_states[name]
            if prior_state is None:
                continue
            target_index = component_index[name]
            evidence_logit[:, :, target_index] = (
                prior_state.coefficient_draws[draw_start:draw_stop]
                @ prior_state.standardized_evidence.T
            )
        component_probability = expit(
            prior_logit[np.newaxis, :, :] + evidence_logit + spatial_logit
        )
        writer.write_block(
            draw_start,
            component_probability=component_probability,
            prior_logit=prior_logit,
            evidence_logit=evidence_logit,
            spatial_logit=spatial_logit,
        )
    summary = writer.finalize(ci_level=bayes_cfg.ci_level)

    components: dict[str, ComponentProbability] = {}
    for name in ordered_names:
        q_index = component_index[name]
        if fitted_state is not None and name in fitted_state.component_names:
            fitted_index = fitted_state.component_names.index(name)
            diagnostics = {
                **fitted_state.diagnostics,
                **evidence_diagnostics[name],
                "posterior_draw_storage": "incremental_hashed_blocks",
            }
            width = len(assembled.layer_names.get(name, []))
            if fitted_state.fixed_coef_draws is not None and width:
                coefficient_draws = fitted_state.fixed_coef_draws[
                    :, :width, fitted_index
                ]
                tail = (1.0 - bayes_cfg.ci_level) / 2.0
                diagnostics["evidence_beta"] = coefficient_draws.mean(
                    axis=0
                ).tolist()
                diagnostics["evidence_beta_interval"] = np.quantile(
                    coefficient_draws, [tail, 1.0 - tail], axis=0
                ).tolist()
            model: Any = fitted_state.fit
            feature_names = tuple(assembled.layer_names.get(name, []))
        else:
            prior_state = prior_states[name]
            if prior_state is None:
                diagnostics = {
                    "inference_role": "fixed_prior_predictive",
                    "outcome_update": False,
                    "spatial_field_included": False,
                    "alpha_provenance": alphas[name].provenance,
                    "n_draws": n_draws,
                    "posterior_draw_storage": "incremental_hashed_blocks",
                }
                model = None
                feature_names = ()
            else:
                diagnostics = {
                    **prior_state.diagnostics,
                    "posterior_draw_storage": "incremental_hashed_blocks",
                }
                model = prior_state
                feature_names = prior_state.feature_names
        probability = (
            grid_gdf[["geometry"]]
            .copy()
            .assign(
                probability=summary.component_mean[:, q_index],
                probability_lo=summary.component_interval[0, :, q_index],
                probability_hi=summary.component_interval[1, :, q_index],
            )
        )
        components[name] = ComponentProbability(
            probability=probability,
            model=model,
            feature_names=feature_names,
            diagnostics=diagnostics,
        )
    combined = (
        grid_gdf[["geometry"]]
        .copy()
        .assign(
            probability=summary.combined_mean,
            probability_lo=summary.combined_interval[0],
            probability_hi=summary.combined_interval[1],
        )
    )
    return ProbabilisticResult(
        components=components,
        combined=combined,
        posterior_draw_index=summary.index_path,
        config=cfg,
        skipped=False,
    )


def run_gblk_probabilistic(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    pfa: dict,
    cfg: ProbabilisticConfig,
    *,
    criteria: str = "geologic",
    nc: int = 6,
    a_wght: float | None = None,
    max_outer_iter: int = 100,
    irls_max_iter: int = 50,
    posterior_scope: str = "baseline",
) -> ProbabilisticResult:
    """Run the joint GBLK probabilistic method end-to-end.

    Assembles per-component inputs (labels, alpha offsets, evidence),
    fits all components jointly via the GBLK engine, and packages the
    result into a :class:`~geopfa.prob.runner.ProbabilisticResult` with Q
    per-component probability surfaces and a joint combined surface.

    Parameters
    ----------
    pfa
        The geoPFA dict (output of the preprocessing pipeline).
    cfg
        Validated :class:`~geopfa.prob.config.ProbabilisticConfig` with
        ``inference.backend == "gblk"``.
    criteria
        Criteria key to operate on, default ``"geologic"``.
    nc
        LatticeKrig lattice centers per dimension passed to
        :func:`~geopfa.prob.gblk_backend.fit_gblk_joint`.
    a_wght
        SAR center weight forwarded to the GBLK fitter. When omitted, the
        positive-definite defaults are 4.5 in 2-D and 8.0 in 3-D.
    max_outer_iter
        Maximum outer IRLS iterations.
    irls_max_iter
        Maximum inner IRLS iterations per outer step.

    Returns
    -------
    ProbabilisticResult
        Per-component probability surfaces in ``components``, joint surface
        in ``combined``, and ``config`` set to ``cfg``. Calibration and
        output fields are empty (handled in later phases).
    """
    _validate_gblk_config(cfg)
    if not cfg.enabled:
        return ProbabilisticResult(config=cfg, skipped=True)
    if cfg.labels.pu_mode == "nnpu":
        raise GEOPFAValueError(
            "nnPU is currently supported by the sequential outcome model only; "
            "a Bernoulli GBLK likelihood cannot treat unlabeled rows as negatives"
        )

    adapter = PFAGridAdapter(pfa, criteria=criteria, dimensions=cfg.dimensions)
    validate_declared_components(
        adapter,
        set(cfg.labels.label_columns) | set(cfg.alpha),
    )

    alphas: dict[str, AlphaCResult] = {}
    for name in adapter.components():
        if name not in cfg.alpha:
            continue
        comp_data = adapter.component_data(name)
        alphas[name] = build_alpha_c(
            comp_data, cfg.alpha[name], grid_gdf=adapter.pr_norm(name)
        )

    prior_only_names = tuple(
        name
        for name, alpha_cfg in cfg.alpha.items()
        if name in alphas and alpha_cfg.force_prior_predictive
    )
    fit_alphas = {
        name: alpha
        for name, alpha in alphas.items()
        if name not in prior_only_names
    }
    assembled_groups: dict[str, AssembledInputs] = {}
    if fit_alphas:
        loaded_labels = load_labels(cfg.labels)
        fitted_label_columns = {
            name: column
            for name, column in loaded_labels.config.label_columns.items()
            if name not in prior_only_names
        }
        fitted_labels = LoadedLabels(
            gdf=loaded_labels.gdf,
            config=replace(
                loaded_labels.config,
                label_columns=fitted_label_columns,
                observation_models={
                    name: model
                    for name, model in loaded_labels.config.observation_models.items()
                    if name in fitted_label_columns
                },
            ),
        )
        assembled_groups = _assemble_likelihood_groups(
            adapter, fitted_labels, fit_alphas, cfg
        )
    assembled = assembled_groups.get("bernoulli")
    first_assembled = next(iter(assembled_groups.values()), None)
    if first_assembled is not None and first_assembled.component_names:
        grid_gdf = adapter.pr_norm(first_assembled.component_names[0])
    elif prior_only_names:
        grid_gdf = adapter.pr_norm(prior_only_names[0])
    else:
        raise GEOPFAValueError(
            "GBLK requires at least one labeled or force-prior-predictive component"
        )

    if cfg.inference.gblk_bayesian.enabled:
        if assembled_groups and not cfg.spatial_field.enabled:
            raise GEOPFAValueError(
                "Bayesian GBLK requires spatial_field.enabled=True"
            )
        if cfg.outputs.posterior_draw_blocks:
            if "gaussian" in assembled_groups:
                raise GEOPFAValueError(
                    "streamed posterior blocks do not yet support Gaussian "
                    "components"
                )
            evidence_design = (
                None
                if assembled is None
                else _prepare_joint_evidence(assembled, cfg)
            )
            return _run_gblk_bayesian_streaming(
                assembled,
                grid_gdf,
                adapter,
                alphas,
                prior_only_names,
                cfg,
                evidence_design=evidence_design,
                nc=nc,
                a_wght=a_wght,
                scope=posterior_scope,
            )
        components: dict[str, ComponentProbability] = {}
        component_draws: dict[str, NDArray[np.float64]] = {}
        fit_families = tuple(sorted(assembled_groups))
        family_configs = dict.fromkeys(
            fit_families, cfg.inference.gblk_bayesian
        )
        if len(fit_families) > 1:
            family_seeds = _spawn_child_seeds(
                cfg.inference.gblk_bayesian.seed,
                len(fit_families),
            )
            family_configs = {
                family: replace(
                    cfg.inference.gblk_bayesian,
                    seed=seed,
                )
                for family, seed in zip(
                    fit_families, family_seeds, strict=True
                )
            }
        if assembled is not None:
            evidence_design = _prepare_joint_evidence(assembled, cfg)
            model_a_wght = _resolve_a_wght(
                assembled.well_coords.shape[1], a_wght
            )
            components, _, component_draws = _run_gblk_bayesian(
                assembled,
                grid_gdf,
                family_configs["bernoulli"],
                well_offsets=assembled.well_offsets,
                grid_offsets=assembled.grid_offsets,
                evidence_diagnostics=evidence_design.diagnostics,
                evidence_design=evidence_design,
                nc=nc,
                nlevel=cfg.spatial_field.n_levels,
                a_wght=model_a_wght,
                coordinate_scaling=cfg.spatial_field.coordinate_scaling,
            )
        gaussian_assembled = assembled_groups.get("gaussian")
        if gaussian_assembled is not None:
            gaussian_grid = adapter.pr_norm(
                gaussian_assembled.component_names[0]
            )
            if not gaussian_grid.geometry.equals(grid_gdf.geometry):
                raise GEOPFAValueError(
                    "Gaussian and Bernoulli components must share a prediction grid"
                )
            gaussian_evidence = _prepare_joint_evidence(
                gaussian_assembled, cfg
            )
            model_a_wght = _resolve_a_wght(
                gaussian_assembled.well_coords.shape[1], a_wght
            )
            gaussian_components, gaussian_draws = _run_gblk_gaussian_bayesian(
                gaussian_assembled,
                grid_gdf,
                alphas,
                cfg,
                family_configs["gaussian"],
                evidence_design=gaussian_evidence,
                nc=nc,
                nlevel=cfg.spatial_field.n_levels,
                a_wght=model_a_wght,
                coordinate_scaling=cfg.spatial_field.coordinate_scaling,
            )
            components.update(gaussian_components)
            component_draws.update(gaussian_draws)
        prior_names = sorted(prior_only_names)
        prior_seeds = _spawn_child_seeds(
            cfg.inference.gblk_bayesian.seed,
            len(prior_names),
        )
        tail = (1.0 - cfg.inference.gblk_bayesian.ci_level) / 2.0
        for name, component_seed in zip(prior_names, prior_seeds, strict=True):
            prior_grid = adapter.pr_norm(name)
            if not prior_grid.geometry.equals(grid_gdf.geometry):
                raise GEOPFAValueError(
                    f"component {name!r} prior grid does not match the "
                    "Bayesian prediction grid"
                )
            if cfg.alpha[name].use_evidence_prior:
                prior_draws = _prior_predictive_evidence_draws(
                    adapter,
                    name,
                    alphas[name],
                    cfg,
                    seed=component_seed,
                )
                draws = prior_draws.probability_draws
                feature_names = prior_draws.feature_names
                diagnostics = prior_draws.diagnostics
                model: Any = prior_draws
            else:
                baseline = expit(alphas[name].grid_offset)
                draws = np.broadcast_to(
                    baseline,
                    (
                        cfg.inference.gblk_bayesian.n_draws,
                        baseline.size,
                    ),
                ).copy()
                feature_names = ()
                diagnostics = {
                    "inference_role": "fixed_prior_predictive",
                    "outcome_update": False,
                    "spatial_field_included": False,
                    "alpha_provenance": alphas[name].provenance,
                    "n_draws": cfg.inference.gblk_bayesian.n_draws,
                }
                model = None
            component_draws[name] = draws
            interval = np.quantile(draws, [tail, 1.0 - tail], axis=0)
            probability = (
                prior_grid[["geometry"]]
                .copy()
                .assign(
                    probability=draws.mean(axis=0),
                    probability_lo=interval[0],
                    probability_hi=interval[1],
                )
            )
            components[name] = ComponentProbability(
                probability=probability,
                model=model,
                feature_names=feature_names,
                diagnostics=diagnostics,
            )
        if cfg.inference.predictive_stacking.enabled:
            stacking = _estimate_predictive_stacking(
                assembled_groups,
                cfg,
                nc=nc,
                a_wght=a_wght,
            )
            _apply_componentwise_stacking(
                components,
                component_draws,
                assembled_groups,
                stacking,
                ci_level=cfg.inference.gblk_bayesian.ci_level,
            )
        ordered_names = tuple(sorted(components))
        if ordered_names != tuple(sorted(cfg.alpha)):
            raise RuntimeError(
                "Bayesian component assembly omitted configured components"
            )
        paired_draws = np.stack(
            [component_draws[name] for name in ordered_names], axis=2
        )
        joint_draws = _combine_probability_columns(
            paired_draws.reshape(-1, paired_draws.shape[2]),
            cfg.combination.rule,
        ).reshape(paired_draws.shape[:2])
        joint_interval = np.quantile(joint_draws, [tail, 1.0 - tail], axis=0)
        combined = (
            grid_gdf[["geometry"]]
            .copy()
            .assign(
                probability=joint_draws.mean(axis=0),
                probability_lo=joint_interval[0],
                probability_hi=joint_interval[1],
            )
        )
        return ProbabilisticResult(
            components=components,
            combined=combined,
            component_probability_draws=component_draws,
            combined_probability_draws=joint_draws,
            config=cfg,
            skipped=False,
        )

    components: dict[str, ComponentProbability] = {}
    if assembled is not None and assembled.component_names:
        if not assembled.observed_mask.any():
            raise GEOPFAValueError(
                "no component-label observations are available for the joint GBLK fit"
            )
        evidence_design = _prepare_joint_evidence(assembled, cfg)
        evidence_well_offsets = assembled.well_offsets
        evidence_grid_offsets = assembled.grid_offsets
        evidence_diagnostics = evidence_design.diagnostics
        model_a_wght = _resolve_a_wght(assembled.well_coords.shape[1], a_wght)
        fit_result = None
        if cfg.spatial_field.enabled:
            fit_result = fit_gblk_joint(
                assembled.well_coords,
                assembled.y,
                assembled.grid_coords,
                component_names=assembled.component_names,
                labeled_mask=np.any(assembled.observed_mask, axis=1),
                observed_mask=assembled.observed_mask,
                offsets=evidence_well_offsets,
                grid_offsets=evidence_grid_offsets,
                fixed_effects=evidence_design.train,
                fixed_effects_grid=evidence_design.prediction,
                fixed_precision=evidence_design.precision,
                fixed_prior_mean=evidence_design.prior_mean,
                nc=nc,
                nlevel=cfg.spatial_field.n_levels,
                a_wght=model_a_wght,
                coordinate_scaling=cfg.spatial_field.coordinate_scaling,
                max_outer_iter=max_outer_iter,
                irls_max_iter=irls_max_iter,
            )
            if fit_result.fit.fixed_coef is not None:
                for q_idx, name in enumerate(assembled.component_names):
                    width = len(assembled.layer_names.get(name, []))
                    if width:
                        evidence_diagnostics[name]["evidence_beta"] = (
                            np.asarray(
                                fit_result.fit.fixed_coef[:width, q_idx]
                            )
                            .astype(float)
                            .tolist()
                        )
        else:
            (
                evidence_well_offsets,
                evidence_grid_offsets,
                evidence_diagnostics,
            ) = _fit_evidence_only_offsets(assembled, cfg)
        for q_idx, name in enumerate(assembled.component_names):
            prob_gdf = grid_gdf[["geometry"]].copy()
            prob_gdf = prob_gdf.assign(
                probability=(
                    fit_result.p_q_grid[:, q_idx].astype(float)
                    if fit_result is not None
                    else expit(evidence_grid_offsets[:, q_idx])
                )
            )
            components[name] = ComponentProbability(
                probability=prob_gdf,
                model=fit_result.fit if fit_result is not None else None,
                feature_names=tuple(assembled.layer_names.get(name, [])),
                diagnostics={
                    **(
                        fit_result.diagnostics
                        if fit_result is not None
                        else {}
                    ),
                    "omega": (
                        fit_result.omega.tolist()
                        if fit_result is not None
                        else None
                    ),
                    "inference_role": "data_informed",
                    "spatial_field_included": fit_result is not None,
                    **evidence_diagnostics[name],
                },
            )

    for name in prior_only_names:
        prior_gdf = adapter.pr_norm(name)[["geometry"]].copy()
        prior_gdf["probability"] = expit(alphas[name].grid_offset)
        feature_names: tuple[str, ...] = ()
        diagnostics = {
            "inference_role": "prior_predictive",
            "outcome_update": False,
            "spatial_field_included": False,
        }
        components[name] = ComponentProbability(
            probability=prior_gdf,
            model=None,
            feature_names=feature_names,
            diagnostics=diagnostics,
        )

    combined = grid_gdf[["geometry"]].copy()
    component_probability = np.column_stack(
        [
            components[name].probability["probability"].to_numpy(dtype=float)
            for name in sorted(components)
        ]
    )
    combined["probability"] = _combine_probability_columns(
        component_probability, cfg.combination.rule
    )

    return ProbabilisticResult(
        components=components,
        combined=combined,
        config=cfg,
        skipped=False,
    )


def freeze_gblk_forward_state(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    pfa: dict,
    cfg: ProbabilisticConfig,
    result: ProbabilisticResult,
    *,
    criteria: str = "geologic",
    coordinate_units: str,
    structural_scenario: str,
) -> FrozenGBLKForwardState:
    """Decompose a deterministic GBLK result for downstream evaluation.

    The returned state reconstructs the fitted MAP probabilities from the
    declared prior offset, each named evidence term, and the fitted spatial
    contribution.  Posterior means are intentionally rejected because a mean
    probability generally cannot be decomposed into a single linear predictor.
    """
    _validate_gblk_config(cfg)
    if cfg.combination.rule != "product":
        raise GEOPFAValueError(
            "frozen GBLK forward state currently represents only the product "
            "combination rule"
        )
    if cfg.inference.gblk_bayesian.enabled:
        raise GEOPFAValueError(
            "freeze_gblk_forward_state requires a deterministic GBLK MAP fit; "
            "Bayesian posterior means need a separate draw-level contract"
        )
    if result.skipped or not result.components:
        raise GEOPFAValueError("cannot freeze an empty GBLK result")

    adapter = PFAGridAdapter(pfa, criteria=criteria, dimensions=cfg.dimensions)
    loaded_labels = load_labels(cfg.labels)
    alphas: dict[str, AlphaCResult] = {}
    for name in adapter.components():
        if name in cfg.alpha:
            component_data = adapter.component_data(name)
            alphas[name] = build_alpha_c(
                component_data,
                cfg.alpha[name],
                grid_gdf=adapter.pr_norm(name),
            )
    prior_only_names = tuple(
        name
        for name, alpha_cfg in cfg.alpha.items()
        if name in alphas and alpha_cfg.force_prior_predictive
    )
    fit_alphas = {
        name: alpha
        for name, alpha in alphas.items()
        if name not in prior_only_names
    }
    fitted_labels = LoadedLabels(
        gdf=loaded_labels.gdf,
        config=replace(
            loaded_labels.config,
            label_columns={
                name: column
                for name, column in loaded_labels.config.label_columns.items()
                if name not in prior_only_names
            },
        ),
    )
    if not fit_alphas:
        raise GEOPFAValueError(
            "frozen GBLK state requires at least one data-informed component"
        )
    assembled = assemble_gblk_inputs(
        adapter,
        fitted_labels,
        fit_alphas,
        evidence_config=cfg.evidence,
    )
    evidence_design = _prepare_joint_evidence(assembled, cfg)
    component_names = tuple(sorted(result.components))
    n_cells = assembled.n_grid
    prior_logit = np.zeros((n_cells, len(component_names)), dtype=np.float64)
    spatial_logit = np.zeros_like(prior_logit)
    baseline = np.column_stack(
        [
            result.components[name]
            .probability["probability"]
            .to_numpy(dtype=np.float64)
            for name in component_names
        ]
    )
    evidence_names: list[str] = []
    evidence_components: list[int] = []
    evidence_columns: list[NDArray[np.float64]] = []

    for component_index, name in enumerate(component_names):
        if name in prior_only_names:
            prior_logit[:, component_index] = alphas[name].grid_offset
            continue
        if name not in assembled.component_names:
            raise GEOPFAValueError(
                f"result component {name!r} is absent from assembled fitted inputs"
            )
        fitted_index = assembled.component_names.index(name)
        prior_logit[:, component_index] = assembled.grid_offsets[
            :, fitted_index
        ]
        width = len(assembled.layer_names[name])
        component = result.components[name]
        if width:
            if evidence_design.prediction is None:
                raise RuntimeError(
                    "fitted evidence is missing its prediction design"
                )
            if component.model is not None:
                fixed_coef = component.model.fixed_coef
                if fixed_coef is None:
                    raise RuntimeError(
                        "joint GBLK model omitted fitted evidence coefficients"
                    )
                beta = np.asarray(fixed_coef, dtype=np.float64)[
                    :width, fitted_index
                ]
            else:
                beta = np.asarray(
                    component.diagnostics["evidence_beta"], dtype=np.float64
                )
            for feature_index, feature_name in enumerate(
                assembled.layer_names[name]
            ):
                evidence_names.append(f"{name}:{feature_name}")
                evidence_components.append(component_index)
                evidence_columns.append(
                    evidence_design.prediction[:, feature_index, fitted_index]
                    * beta[feature_index]
                )
        evidence_sum = np.zeros(n_cells, dtype=np.float64)
        for term, term_component in zip(
            evidence_columns, evidence_components, strict=True
        ):
            if term_component == component_index:
                evidence_sum += term
        clipped = np.clip(baseline[:, component_index], 1e-15, 1.0 - 1e-15)
        fitted_logit = np.log(clipped) - np.log1p(-clipped)
        spatial_logit[:, component_index] = (
            fitted_logit - prior_logit[:, component_index] - evidence_sum
        )

    combined = result.combined["probability"].to_numpy(dtype=np.float64)
    product = np.prod(baseline, axis=1)
    if not np.allclose(combined, product, rtol=0.0, atol=1e-12):
        raise GEOPFAValueError(
            "result combined surface is not the declared component product"
        )
    coordinate_names = ("x", "y", "z")[: assembled.grid_coords.shape[1]]
    return FrozenGBLKForwardState(
        coordinates=assembled.grid_coords,
        coordinate_names=coordinate_names,
        coordinate_units=coordinate_units,
        component_names=component_names,
        prior_only_components=tuple(
            name for name in component_names if name in prior_only_names
        ),
        prior_logit=prior_logit,
        evidence_logit_contribution=(
            np.column_stack(evidence_columns)
            if evidence_columns
            else np.empty((n_cells, 0), dtype=np.float64)
        ),
        evidence_term_names=tuple(evidence_names),
        evidence_term_component=np.asarray(
            evidence_components, dtype=np.int64
        ),
        spatial_logit=spatial_logit,
        baseline_component_probability=baseline,
        structural_scenario=structural_scenario,
    )


def _assemble_from_config(
    pfa: dict, cfg: ProbabilisticConfig, criteria: str
) -> AssembledInputs:
    """Build :class:`AssembledInputs` from a ``pfa`` dict and config."""
    adapter = PFAGridAdapter(pfa, criteria=criteria, dimensions=cfg.dimensions)
    loaded_labels = load_labels(cfg.labels)

    alphas: dict[str, AlphaCResult] = {}
    for name in adapter.components():
        if name not in cfg.alpha:
            continue
        comp_data = adapter.component_data(name)
        alphas[name] = build_alpha_c(
            comp_data, cfg.alpha[name], grid_gdf=adapter.pr_norm(name)
        )
    prior_only_names = {
        name
        for name, alpha_cfg in cfg.alpha.items()
        if alpha_cfg.force_prior_predictive
    }
    fit_alphas = {
        name: alpha
        for name, alpha in alphas.items()
        if name not in prior_only_names
    }
    fitted_labels = LoadedLabels(
        gdf=loaded_labels.gdf,
        config=replace(
            loaded_labels.config,
            label_columns={
                name: column
                for name, column in loaded_labels.config.label_columns.items()
                if name not in prior_only_names
            },
        ),
    )
    return assemble_gblk_inputs(
        adapter,
        fitted_labels,
        fit_alphas,
        evidence_config=cfg.evidence,
    )


def _cv_evidence_only_offsets(  # noqa: PLR0913
    assembled: AssembledInputs,
    row_idx: NDArray[np.intp],
    component_indices: tuple[int, ...],
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
    *,
    cfg: ProbabilisticConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit the evidence-only CV ablation on the joint model's labeled rows."""
    train_offsets = assembled.well_offsets[row_idx[train_mask]][
        :, component_indices
    ].copy()
    test_offsets = assembled.well_offsets[row_idx[test_mask]][
        :, component_indices
    ].copy()
    for local_idx, q_idx in enumerate(component_indices):
        name = assembled.component_names[q_idx]
        x_all = np.asarray(assembled.evidence[name][row_idx], dtype=np.float64)
        if x_all.shape[1] == 0:
            continue
        x_train = x_all[train_mask]
        x_test = x_all[test_mask]
        train_scaled, test_scaled, *_ = _standardize_partial_evidence(
            x_train,
            x_test,
            np.ones(x_train.shape[0], dtype=bool),
            component_name=name,
            layer_names=assembled.layer_names[name],
            minimum_count=cfg.labels.min_wells_for_fit,
        )
        weights, prior_means = _resolved_evidence_prior(
            cfg, assembled.layer_names[name], component=name
        )
        fit = _fit_offset_logit(
            train_scaled,
            assembled.y[row_idx[train_mask], q_idx],
            train_offsets[:, local_idx],
            regularization=1.0 / cfg.evidence.regularization.C,
            per_feature_weights=weights,
            prior_means=prior_means,
        )
        beta = np.asarray(fit.x, dtype=np.float64)
        train_offsets[:, local_idx] += train_scaled @ beta
        test_offsets[:, local_idx] += test_scaled @ beta
    return train_offsets, test_offsets


def _cv_joint_evidence(  # noqa: PLR0913
    assembled: AssembledInputs,
    row_idx: NDArray[np.intp],
    component_indices: tuple[int, ...],
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
    *,
    cfg: ProbabilisticConfig,
) -> JointEvidenceDesign:
    """Prepare fold-local evidence designs without fitting or leakage."""
    train_rows = row_idx[train_mask]
    test_rows = row_idx[test_mask]
    names = tuple(
        assembled.component_names[index] for index in component_indices
    )
    return _prepare_joint_evidence_arrays(
        component_names=names,
        train_evidence={
            name: np.asarray(
                assembled.evidence[name][train_rows], dtype=np.float64
            )
            for name in names
        },
        prediction_evidence={
            name: np.asarray(
                assembled.evidence[name][test_rows], dtype=np.float64
            )
            for name in names
        },
        layer_names={name: assembled.layer_names[name] for name in names},
        y_train=assembled.y[train_rows][:, component_indices],
        observed_train=assembled.observed_mask[train_rows][
            :, component_indices
        ],
        cfg=cfg,
    )


def _make_fit_fn(  # noqa: PLR0913
    assembled: AssembledInputs,
    row_idx: NDArray[np.intp],
    component_indices: tuple[int, ...],
    *,
    target: str | int,
    cfg: ProbabilisticConfig,
    nc: int,
    nlevel: int,
    a_wght: float | None,
    max_outer_iter: int,
    irls_max_iter: int,
):
    """Build a ``calibration_cv``-compatible ``fit_fn``.

    ``target`` is either ``"joint"`` (return ``p_joint`` at test wells) or
    an integer component index (return ``p_q`` at test wells for that
    component).
    """
    coords_lab = assembled.well_coords[row_idx]
    y_lab = assembled.y[row_idx][:, component_indices]
    component_names = tuple(
        assembled.component_names[q] for q in component_indices
    )

    def fit_fn(
        train_mask: NDArray[np.bool_], test_mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        train_mask = np.asarray(train_mask, dtype=bool)
        test_mask = np.asarray(test_mask, dtype=bool)
        if not cfg.spatial_field.enabled:
            _train_offsets, test_offsets = _cv_evidence_only_offsets(
                assembled,
                row_idx,
                component_indices,
                train_mask,
                test_mask,
                cfg=cfg,
            )
            probabilities = expit(test_offsets)
            if target == "joint":
                return _combine_probability_columns(
                    probabilities, cfg.combination.rule
                )
            return probabilities[:, int(target)]
        evidence_design = _cv_joint_evidence(
            assembled,
            row_idx,
            component_indices,
            train_mask,
            test_mask,
            cfg=cfg,
        )
        train_offsets = assembled.well_offsets[row_idx[train_mask]][
            :, component_indices
        ]
        test_offsets = assembled.well_offsets[row_idx[test_mask]][
            :, component_indices
        ]
        model_a_wght = _resolve_a_wght(coords_lab.shape[1], a_wght)
        fit = fit_gblk_joint(
            coords_lab[train_mask],
            y_lab[train_mask],
            coords_lab[test_mask],
            component_names=component_names,
            labeled_mask=None,
            offsets=train_offsets,
            grid_offsets=test_offsets,
            fixed_effects=evidence_design.train,
            fixed_effects_grid=evidence_design.prediction,
            fixed_precision=evidence_design.precision,
            fixed_prior_mean=evidence_design.prior_mean,
            nc=nc,
            nlevel=nlevel,
            a_wght=model_a_wght,
            coordinate_scaling=cfg.spatial_field.coordinate_scaling,
            max_outer_iter=max_outer_iter,
            irls_max_iter=irls_max_iter,
        )
        if target == "joint":
            return _combine_probability_columns(
                fit.p_q_grid, cfg.combination.rule
            )
        return fit.p_q_grid[:, int(target)]

    return fit_fn


def _calibration_cv_from_splits(  # noqa: PLR0914
    fit_fn: Any,
    labels: NDArray[np.float64],
    *,
    splits: Sequence[tuple[NDArray[np.bool_], NDArray[np.bool_]]],
    fold_ids: NDArray[np.intp],
    n_bins: int,
) -> CalibrationCVResult:
    """Evaluate geoPFA's audited buffered folds with LatticeKrigX metrics."""
    from latticekrigx.glk.calibration import (  # noqa: PLC0415
        CalibrationCVResult,
        FoldMetrics,
        brier_score,
        reliability_diagram,
    )

    y = np.asarray(labels, dtype=np.float64)
    fold_array = np.asarray(fold_ids, dtype=np.intp)
    if y.ndim != 1 or fold_array.shape != y.shape:
        raise ValueError(
            "labels and fold_ids must be one-dimensional and aligned"
        )
    folds: list[FoldMetrics] = []
    for fold, (train_mask, test_mask) in enumerate(splits):
        train = np.asarray(train_mask, dtype=bool)
        test = np.asarray(test_mask, dtype=bool)
        if train.shape != y.shape or test.shape != y.shape:
            raise ValueError("calibration split masks must match labels")
        if not train.any() or not test.any() or np.any(train & test):
            raise ValueError(
                f"calibration fold {fold} must have disjoint nonempty train/test sets"
            )
        if not np.array_equal(test, fold_array == fold):
            raise ValueError(
                f"calibration fold {fold} test mask does not match fold_ids"
            )
        probabilities = np.asarray(
            fit_fn(train, test), dtype=np.float64
        ).ravel()
        if probabilities.shape != (int(test.sum()),):
            raise ValueError(
                f"fit_fn returned {probabilities.shape} for "
                f"{int(test.sum())} test rows in fold {fold}"
            )
        labels_test = y[test]
        score = brier_score(probabilities, labels_test)
        train_prevalence = float(y[train].mean())
        reference_score = float(np.mean((train_prevalence - labels_test) ** 2))
        extra = {
            "n_train": int(train.sum()),
            "n_buffered": int((~train & ~test).sum()),
        }
        if reference_score == 0.0:
            skill = float("nan")
            extra["brier_skill_score_status"] = (
                "undefined_zero_reference_score"
            )
        else:
            skill = float(1.0 - score / reference_score)
        reliability = reliability_diagram(
            probabilities, labels_test, n_bins=n_bins
        )
        occupied = reliability.bin_counts > 0
        calibration_error = float(
            np.sum(
                reliability.bin_counts[occupied]
                * np.abs(
                    reliability.bin_mean_pred[occupied]
                    - reliability.bin_fracs[occupied]
                )
            )
            / reliability.bin_counts.sum()
        )
        calibration_parameters = calibration_intercept_slope(
            labels_test, probabilities
        )
        folds.append(
            FoldMetrics(
                fold_id=fold,
                n_train=int(train.sum()),
                n_test=int(test.sum()),
                n_buffered=int((~train & ~test).sum()),
                prevalence=float(labels_test.mean()),
                brier_score=score,
                brier_skill_score=skill,
                log_score=log_loss(labels_test, probabilities),
                expected_calibration_error=calibration_error,
                calibration_intercept=calibration_parameters["intercept"],
                calibration_slope=calibration_parameters["slope"],
                reliability=reliability,
                extra=extra,
            )
        )
    defined_skill = [
        fold.brier_skill_score
        for fold in folds
        if np.isfinite(fold.brier_skill_score)
    ]
    return CalibrationCVResult(
        folds=folds,
        mean_brier_score=float(np.mean([fold.brier_score for fold in folds])),
        mean_bss=(
            float(np.mean(defined_skill)) if defined_skill else float("nan")
        ),
        fold_ids=fold_array.copy(),
    )


def run_gblk_calibration_cv(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    pfa: dict,
    cfg: ProbabilisticConfig,
    *,
    criteria: str = "geologic",
    n_folds: int | None = None,
    n_bins: int = 10,
    random_state: int = 0,
    nc: int = 6,
    a_wght: float | None = None,
    max_outer_iter: int = 100,
    irls_max_iter: int = 50,
    dims: tuple[int, ...] | None = None,
    components: Sequence[str] | None = None,
) -> dict[str, CalibrationCVResult]:
    """Spatially-blocked cross-validation of the joint GBLK method.

    Wraps :func:`latticekrigx.glk.calibration.calibration_cv` for each
    component and for the conditional plug-in co-occurrence target. Folds are
    generated by :func:`~latticekrigx.glk.calibration.spatial_blocks`
    over the labeled wells' 2-D coordinates; the same
    ``random_state`` seeds all runs, so all returned results share the
    same fold assignments. Per-fold metrics include the mean Brier
    score, the Brier Skill Score referenced against the per-fold
    prevalence, and a reliability diagram.

    Parameters
    ----------
    pfa
        The geoPFA dict (output of the preprocessing pipeline).
    cfg
        Validated :class:`~geopfa.prob.config.ProbabilisticConfig` with
        ``inference.backend == "gblk"``.
    criteria
        Criteria key to operate on, default ``"geologic"``.
    n_folds : int, default=5
        Number of spatial CV folds.
    n_bins : int, default=10
        Reliability-diagram bin count.
    random_state : int, default=0
        Seed for the block permutation; also used to seed each per-target
        call so all results share fold assignments.
    nc : int, default=6
        LatticeKrig lattice centers per dimension.
    a_wght : float, optional
        SAR center weight. Defaults to 4.5 in 2-D and 8.0 in 3-D.
    max_outer_iter : int, default=100
        Outer IRLS iteration cap.
    irls_max_iter : int, default=50
        Inner IRLS iteration cap.
    dims : tuple of int, optional
        Coordinate columns used for spatial blocking (default first two).
    components : sequence of str, optional
        Restrict calibration-CV to this subset of names (from the
        model's component names, plus the literal ``"joint"``). Each
        component metric uses that component's observed labels; ``"joint"``
        uses complete cases for all fitted components. Use this when one or more
        components have too few (or class-degenerate) real labels to
        support an identifiable calibration-CV metric — e.g. a
        producibility or insulation layer with a single labeled well —
        while another component (e.g. heat) has enough labels for an
        honest reliability check. Defaults to ``None``, which computes
        every component plus ``"joint"`` (previous behavior).

    Returns
    -------
    dict of str to CalibrationCVResult
        One entry per requested name (default: every component name plus
        ``"joint"``, keyed on the conditional plug-in co-occurrence target). Each
        value is a :class:`latticekrigx.glk.calibration.CalibrationCVResult`
        whose ``mean_brier_score`` and ``mean_bss`` summarize the run.

    Raises
    ------
    geopfa.exceptions.GEOPFAValueError
        If ``cfg.enabled`` is ``False``, no labeled wells are available,
        or ``components`` contains a name that is not a model component
        name and is not ``"joint"``.

    Notes
    -----
    Per-component diagnostics use that component's observed-label mask. Joint
    diagnostics use complete-case rows and element-wise products across the
    component labels. Evidence effects are estimated anew inside every training
    fold so held-out outcomes cannot leak into predictions.
    """
    _validate_gblk_config(cfg)
    if not cfg.enabled:
        raise GEOPFAValueError(
            "run_gblk_calibration_cv requires cfg.enabled=True"
        )

    if cfg.labels.pu_mode == "nnpu":
        raise GEOPFAValueError(
            "GBLK calibration cannot score nnPU labels with a Bernoulli-negative likelihood"
        )
    if cfg.inference.gblk_bayesian.enabled:
        raise GEOPFAValueError(
            "Bayesian GBLK cross-validation is not implemented by this "
            "deterministic calibration runner; fit Bayesian replicates through "
            "the canonical fit_gblk_bayesian_joint path"
        )
    if n_folds is None:
        effective_n_folds = cfg.cross_validation.n_folds
    elif isinstance(n_folds, bool | np.bool_) or not isinstance(
        n_folds, int | np.integer
    ):
        raise GEOPFAValueError("n_folds must be an integer")
    else:
        effective_n_folds = int(n_folds)
    if (
        cfg.cross_validation.buffer_km > 0.0
        or cfg.cross_validation.block_size_km is not None
    ):
        adapter = PFAGridAdapter(
            pfa, criteria=criteria, dimensions=cfg.dimensions
        )
        first_component = adapter.components()[0]
        crs = adapter.pr_norm(first_component).crs
        if (
            crs is None
            or not crs.is_projected
            or any(
                axis.unit_name.lower() not in {"metre", "meter"}
                for axis in crs.axis_info[:2]
            )
        ):
            raise GEOPFAValueError(
                "cross_validation block_size_km/buffer_km controls require a "
                "projected metre-based CRS"
            )
    assembled = _assemble_from_config(pfa, cfg, criteria)
    allowed_names = set(assembled.component_names) | {"joint"}
    if components is None:
        requested_names = [*assembled.component_names, "joint"]
    else:
        requested_names = list(components)
        unknown = [n for n in requested_names if n not in allowed_names]
        if unknown:
            raise GEOPFAValueError(
                f"components contains unknown name(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(allowed_names)}"
            )

    for q_idx, name in enumerate(assembled.component_names):
        if name not in requested_names:
            continue
        n_observed = int(assembled.observed_mask[:, q_idx].sum())
        if n_observed < effective_n_folds:
            raise GEOPFAValueError(
                f"component {name!r} has fewer than {effective_n_folds} observed labels "
                f"({n_observed}); calibration cross-validation is not identifiable"
            )
        values = assembled.y[assembled.observed_mask[:, q_idx], q_idx]
        if np.unique(values).size < 2:  # noqa: PLR2004
            raise GEOPFAValueError(
                f"component {name!r} has only one observed label class; "
                "calibration cross-validation is not identifiable"
            )

    complete_idx = np.flatnonzero(assembled.labeled_mask).astype(np.intp)
    if "joint" in requested_names:
        if complete_idx.size < effective_n_folds:
            raise GEOPFAValueError(
                f"joint target has fewer than {effective_n_folds} complete-case labels "
                f"({complete_idx.size}); calibration cross-validation is not identifiable"
            )
        joint_values = np.prod(assembled.y[complete_idx], axis=1)
        if np.unique(joint_values).size < 2:  # noqa: PLR2004
            raise GEOPFAValueError(
                "joint target has only one observed label class; calibration "
                "cross-validation is not identifiable"
            )

    results: dict[str, CalibrationCVResult] = {}
    for q_idx, name in enumerate(assembled.component_names):
        if name not in requested_names:
            continue
        component_idx = np.flatnonzero(
            assembled.observed_mask[:, q_idx]
        ).astype(np.intp)
        coords_component = assembled.well_coords[component_idx]
        fit_fn = _make_fit_fn(
            assembled,
            component_idx,
            (q_idx,),
            target=0,
            cfg=cfg,
            nc=nc,
            nlevel=cfg.spatial_field.n_levels,
            a_wght=a_wght,
            max_outer_iter=max_outer_iter,
            irls_max_iter=irls_max_iter,
        )
        component_splits = list(
            spatial_block_cv(
                coords_component,
                n_folds=effective_n_folds,
                block_type=cfg.cross_validation.block_type,
                grid_size=cfg.cross_validation.grid_size,
                seed=random_state,
                block_size_km=cfg.cross_validation.block_size_km,
                buffer_distance=cfg.cross_validation.buffer_km * 1000.0,
                dims=(0, 1) if dims is None else dims,
            )
        )
        component_fold_ids = np.empty(coords_component.shape[0], dtype=np.intp)
        for fold, (_train_mask, test_mask) in enumerate(component_splits):
            component_fold_ids[test_mask] = fold
        results[name] = _calibration_cv_from_splits(
            fit_fn,
            assembled.y[component_idx, q_idx].astype(np.float64),
            splits=component_splits,
            fold_ids=component_fold_ids,
            n_bins=n_bins,
        )

    if "joint" in requested_names:
        y_joint = np.prod(assembled.y[complete_idx], axis=1)
        coords_joint = assembled.well_coords[complete_idx]
        fit_fn_joint = _make_fit_fn(
            assembled,
            complete_idx,
            tuple(range(len(assembled.component_names))),
            target="joint",
            cfg=cfg,
            nc=nc,
            nlevel=cfg.spatial_field.n_levels,
            a_wght=a_wght,
            max_outer_iter=max_outer_iter,
            irls_max_iter=irls_max_iter,
        )
        joint_splits = list(
            spatial_block_cv(
                coords_joint,
                n_folds=effective_n_folds,
                block_type=cfg.cross_validation.block_type,
                grid_size=cfg.cross_validation.grid_size,
                seed=random_state,
                block_size_km=cfg.cross_validation.block_size_km,
                buffer_distance=cfg.cross_validation.buffer_km * 1000.0,
                dims=(0, 1) if dims is None else dims,
            )
        )
        joint_fold_ids = np.empty(coords_joint.shape[0], dtype=np.intp)
        for fold, (_train_mask, test_mask) in enumerate(joint_splits):
            joint_fold_ids[test_mask] = fold
        results["joint"] = _calibration_cv_from_splits(
            fit_fn_joint,
            y_joint.astype(np.float64),
            splits=joint_splits,
            fold_ids=joint_fold_ids,
            n_bins=n_bins,
        )

    return results


def run_gblk_hierarchical_regional(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    pfa: dict,
    cfg: ProbabilisticConfig,
    region_per_well: NDArray,
    region_per_grid: NDArray,
    play_type_per_region: dict[str, str],
    *,
    criteria: str = "geologic",
    nc: int = 6,
    a_wght: float | None = None,
    max_outer_iter: int = 10,
    irls_max_iter: int = 15,
) -> ProbabilisticResult:
    """Run multi-region GBLK with play-type partial pooling.

    Fits a single joint GBLK model on all labeled wells, then applies
    empirical-Bayes regional pooling grouped by play type.  Regions with
    zero labeled wells receive predictions from the pooled play-type mean
    (shrinkage = 1); data-rich regions depart from the pool (shrinkage
    near 0).

    Parameters
    ----------
    pfa
        The geoPFA dict (output of the preprocessing pipeline).
    cfg
        Validated :class:`~geopfa.prob.config.ProbabilisticConfig` with
        ``inference.backend == "gblk"``.
    region_per_well
        Region label for each well, shape ``(n_wells,)``.  Must be the
        same order as wells in the assembled inputs.
    region_per_grid
        Region label for each grid cell, shape ``(G,)``.  Values must be
        keys in ``play_type_per_region``.
    play_type_per_region
        Mapping from region name to play type.  All region labels
        appearing in ``region_per_well`` and ``region_per_grid`` must
        be present.
    criteria
        Criteria key to operate on, default ``"geologic"``.
    nc
        LatticeKrig lattice centers per dimension.
    a_wght
        SAR center weight. Defaults to 4.5 in 2-D and 8.0 in 3-D.
    max_outer_iter
        Maximum outer IRLS iterations.
    irls_max_iter
        Maximum inner IRLS iterations per outer step.

    Returns
    -------
    ProbabilisticResult
        Per-component and joint probability surfaces with hierarchical
        regional corrections applied.  The ``diagnostics`` field of each
        component includes ``pool_shrinkage`` and ``d_post``.

    Raises
    ------
    geopfa.exceptions.GEOPFAValueError
        If ``cfg.enabled`` is ``False``, if any region label is missing
        from ``play_type_per_region``, or if no labeled wells are
        available after filtering.
    """
    _validate_gblk_config(cfg)
    if not cfg.enabled:
        return ProbabilisticResult(config=cfg, skipped=True)

    adapter = PFAGridAdapter(pfa, criteria=criteria, dimensions=cfg.dimensions)
    loaded_labels = load_labels(cfg.labels)

    alphas: dict[str, AlphaCResult] = {}
    for name in adapter.components():
        if name not in cfg.alpha:
            continue
        comp_data = adapter.component_data(name)
        alphas[name] = build_alpha_c(
            comp_data, cfg.alpha[name], grid_gdf=adapter.pr_norm(name)
        )

    assembled = assemble_gblk_inputs(
        adapter,
        loaded_labels,
        alphas,
        evidence_config=cfg.evidence,
    )
    grid_gdf = adapter.pr_norm(assembled.component_names[0])

    regions_well = np.asarray(region_per_well)
    regions_grid = np.asarray(region_per_grid)
    if regions_well.shape != (assembled.n,):
        raise GEOPFAValueError(
            f"region_per_well must have shape ({assembled.n},)"
        )
    if regions_grid.shape != (assembled.n_grid,):
        raise GEOPFAValueError(
            f"region_per_grid must have shape ({assembled.n_grid},)"
        )
    region_names = sorted(play_type_per_region.keys())

    missing = set(np.unique(regions_grid).tolist()) - set(region_names)
    if missing:
        raise GEOPFAValueError(
            f"region_per_grid contains labels not in play_type_per_region: "
            f"{missing}"
        )
    missing_well = set(np.unique(regions_well).tolist()) - set(region_names)
    if missing_well:
        raise GEOPFAValueError(
            f"region_per_well contains labels not in play_type_per_region: "
            f"{missing_well}"
        )

    evidence_design = _prepare_joint_evidence(assembled, cfg)
    prediction_coords = np.vstack(
        [assembled.well_coords, assembled.grid_coords]
    )
    prediction_offsets = np.vstack(
        [assembled.well_offsets, assembled.grid_offsets]
    )
    prediction_fixed = (
        None
        if evidence_design.train is None
        else np.concatenate(
            [evidence_design.train, evidence_design.prediction], axis=0
        )
    )
    model_a_wght = _resolve_a_wght(assembled.well_coords.shape[1], a_wght)
    fit_result = fit_gblk_joint(
        assembled.well_coords,
        assembled.y,
        prediction_coords,
        component_names=assembled.component_names,
        labeled_mask=np.any(assembled.observed_mask, axis=1),
        observed_mask=assembled.observed_mask,
        offsets=assembled.well_offsets,
        grid_offsets=prediction_offsets,
        fixed_effects=evidence_design.train,
        fixed_effects_grid=prediction_fixed,
        fixed_precision=evidence_design.precision,
        fixed_prior_mean=evidence_design.prior_mean,
        nc=nc,
        nlevel=cfg.spatial_field.n_levels,
        a_wght=model_a_wght,
        coordinate_scaling=cfg.spatial_field.coordinate_scaling,
        max_outer_iter=max_outer_iter,
        irls_max_iter=irls_max_iter,
    )

    n_well = assembled.n
    eta_pred_train = np.asarray(
        fit_result.fit.eta_pred[:n_well], dtype=np.float64
    )
    fixed_contrib = np.zeros_like(eta_pred_train)
    if evidence_design.train is not None:
        fixed_coef = fit_result.fit.fixed_coef
        if fixed_coef is None:  # pragma: no cover - backend contract guard
            raise RuntimeError(
                "joint GBLK fit omitted coefficients for supplied fixed effects"
            )
        fixed_contrib = np.einsum(
            "npq,pq->nq", evidence_design.train, fixed_coef
        )
    spatial_contrib = eta_pred_train - assembled.well_offsets - fixed_contrib

    R = len(region_names)
    Q = len(assembled.component_names)
    large_var = 1e10

    d_hat = np.zeros((R, Q), dtype=np.float64)
    d_var = np.full((R, Q), large_var, dtype=np.float64)

    for r_idx, rname in enumerate(region_names):
        for q_idx in range(Q):
            well_mask = (regions_well == rname) & assembled.observed_mask[
                :, q_idx
            ]
            n_r = int(well_mask.sum())
            if n_r > 0:
                d_hat[r_idx, q_idx] = spatial_contrib[well_mask, q_idx].mean()
                d_var[r_idx, q_idx] = 1.0 / n_r

    play_types = [play_type_per_region[r] for r in region_names]
    pool_result = pool_regional_coefficients(d_hat, d_var, play_types)
    d_post = pool_result.d_pooled  # (R, Q)

    correction = d_post - d_hat  # (R, Q)

    region_to_idx = {rname: idx for idx, rname in enumerate(region_names)}
    grid_region_idx = np.array(
        [region_to_idx[r] for r in regions_grid.tolist()], dtype=np.intp
    )
    grid_correction = correction[grid_region_idx]  # (G, Q)

    base_prob = np.clip(fit_result.p_q_grid[n_well:], 1e-12, 1.0 - 1e-12)
    eta_grid = np.log(base_prob) - np.log1p(-base_prob) + grid_correction

    p_q_grid = expit(eta_grid)
    p_joint_grid = _combine_probability_columns(p_q_grid, cfg.combination.rule)

    diag_base = {
        **fit_result.diagnostics,
        "omega": fit_result.omega.tolist(),
        "hierarchical": True,
        "pool_shrinkage": pool_result.shrinkage.tolist(),
        "d_post": d_post.tolist(),
        "region_names": region_names,
        "hierarchical_effect_source": "spatial_field_after_joint_evidence_fit",
        "prediction_grid_size": int(assembled.n_grid),
    }

    components: dict[str, ComponentProbability] = {}
    for q_idx, name in enumerate(assembled.component_names):
        prob_gdf = grid_gdf[["geometry"]].copy()
        prob_gdf = prob_gdf.assign(
            probability=p_q_grid[:, q_idx].astype(float)
        )
        components[name] = ComponentProbability(
            probability=prob_gdf,
            model=fit_result.fit,
            feature_names=tuple(assembled.layer_names.get(name, [])),
            diagnostics={**diag_base, **evidence_design.diagnostics[name]},
        )

    combined = grid_gdf[["geometry"]].copy()
    combined = combined.assign(probability=p_joint_grid.astype(float))

    return ProbabilisticResult(
        components=components,
        combined=combined,
        config=cfg,
        skipped=False,
    )


__all__ = [
    "freeze_gblk_forward_state",
    "run_gblk_calibration_cv",
    "run_gblk_hierarchical_regional",
    "run_gblk_probabilistic",
]
