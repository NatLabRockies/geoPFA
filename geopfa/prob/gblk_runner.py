"""Joint Generalized Bayesian LatticeKrig runner for geoPFA.

The module owns the model-level orchestration shared by deterministic MAP and
Bayesian Paige/INLA execution: input assembly, same-family fitting, optional
predictive stacking, prior-only components, and probability-field packaging.
Large Bayesian outputs use the internal streamed posterior writer so draw
projection can resume without refitting or holding a full draw cube in memory.

The public entry point is :func:`run_gblk_probabilistic`; the remaining helpers
are implementation details rather than a second workflow or artifact API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp, ndtr, ndtri
from scipy.spatial import cKDTree

from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.alpha import AlphaCResult, build_alpha_c
from geopfa.prob.calibration import calibration_intercept_slope, log_loss
from geopfa.prob.config import (
    GBLKBayesianConfig,
    LabelsConfig,
    ProbabilisticConfig,
)
from geopfa.prob.cv import spatial_block_cv
from geopfa.prob.fitting import ComponentProbability
from geopfa.prob.fitting import _fit_offset_logit
from geopfa.prob.gblk_assemble import (
    AssembledInputs,
    _component_values_on_reference,
    assemble_gblk_inputs,
    build_component_grid_evidence,
)
from geopfa.prob.gblk_backend import (
    fit_gblk_bayesian_joint,
    fit_gblk_bayesian_posterior_state,
    fit_gblk_gaussian_bayesian_joint,
    fit_gblk_joint,
    project_gblk_bayesian_draw_block,
    project_gblk_gaussian_bayesian_draw_block,
)
from geopfa.prob.io import (
    PersistedPosteriorDrawState,
    PosteriorDrawBlockWriter,
    _probabilistic_implementation_hash,
    load_posterior_draw_state,
)
from geopfa.prob.labels import LoadedLabels, load_labels
from geopfa.prob.pfa_grid import PFAGridAdapter, validate_declared_components
from geopfa.prob.predictive_stacking import (
    PredictiveStackingEvidence,
    PredictiveStackingResult,
    apply_predictive_stacking,
    select_predictive_density_stacking_weight,
    select_predictive_stacking_weight,
)
from geopfa.prob.runner import ProbabilisticResult
from geopfa.prob.spatial_alignment import require_same_grid
from geopfa.prob.spatial_alignment import extract_coordinates

if TYPE_CHECKING:  # pragma: no cover - typing only
    from latticekrigx.glk.calibration import CalibrationCVResult

_TWO_DIMENSIONS = 2
_THREE_DIMENSIONS = 3
_OPEN_PROBABILITY_EPSILON = 1e-12
_PREDICTIVE_SUMMARY_CHUNK_SIZE = 10_000


def _posterior_draw_cell_indices(
    cfg: ProbabilisticConfig, *, n_cells: int
) -> NDArray[np.int64] | None:
    """Load and hash-verify one strict prediction-cell subset."""
    source = cfg.outputs.posterior_draw_cell_indices_source
    expected_sha256 = cfg.outputs.posterior_draw_cell_indices_sha256
    if source is None:
        return None
    if expected_sha256 is None:  # pragma: no cover - config invariant
        raise RuntimeError("posterior draw cell-index checksum is missing")
    path = Path(source)
    if not path.is_file():
        raise ValueError(
            f"posterior draw cell-index source does not exist: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("posterior draw cell-index source checksum mismatch")
    try:
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "posterior draw cell-index source must be a NumPy .npy array"
        ) from exc
    try:
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError(
                "posterior draw cell indices must be a one-dimensional "
                "integer array"
            )
        indices = np.array(raw, dtype=np.int64, copy=True)
    finally:
        mapping = getattr(raw, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()
    if (
        not 0 < indices.size < n_cells
        or np.any(indices < 0)
        or np.any(indices >= n_cells)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "posterior draw cell indices must be a strictly increasing, "
            "unique, in-range strict subset of the prediction grid"
        )
    return indices


def _spawn_child_seeds(seed: int, count: int) -> tuple[int, ...]:
    """Derive deterministic seeds accepted by pyINLA and legacy NumPy APIs."""
    children = np.random.SeedSequence(seed).spawn(count)
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    )


def _stacking_training_minimum(cfg: ProbabilisticConfig) -> int:
    """Return the declared unique-well support required inside each fold."""
    configured = cfg.inference.predictive_stacking.minimum_training_wells
    return cfg.labels.min_wells_for_fit if configured is None else configured


def _unique_well_count(well_ids: np.ndarray, mask: np.ndarray) -> int:
    """Count declared well identities among selected observation rows."""
    identifiers = np.asarray(well_ids)
    selected = np.asarray(mask, dtype=bool)
    if identifiers.ndim != 1 or selected.shape != identifiers.shape:
        raise ValueError("well_ids and mask must be aligned row vectors")
    return int(np.unique(identifiers[selected]).size)


def _fold_has_required_training_support(
    assembled: AssembledInputs,
    train_mask: NDArray[np.bool_],
    *,
    minimum_wells: int,
) -> bool:
    """Return whether every family component can be fitted in one CV fold."""
    selected = np.asarray(train_mask, dtype=bool)
    if selected.shape != (len(assembled.well_ids),):
        raise ValueError("training mask must align with assembled well rows")
    for q_idx, name in enumerate(assembled.component_names):
        observed = selected & assembled.observed_mask[:, q_idx]
        if _unique_well_count(assembled.well_ids, observed) < minimum_wells:
            return False
        evidence = np.asarray(assembled.evidence[name], dtype=np.float64)
        if evidence.shape[0] != selected.size:
            raise ValueError(
                f"component {name!r} evidence must align with assembled well rows"
            )
        for feature_index in range(evidence.shape[1]):
            finite = observed & np.isfinite(evidence[:, feature_index])
            if _unique_well_count(assembled.well_ids, finite) < minimum_wells:
                return False
    return True


def _grouped_spatial_folds(  # noqa: PLR0913
    coordinates: np.ndarray,
    well_ids: np.ndarray,
    *,
    n_folds: int,
    block_type: str,
    grid_size: int,
    seed: int,
    block_size_km: float | None,
    buffer_distance: float,
    dims: tuple[int, ...] = (0, 1),
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Build spatial folds on wells and expand them to all profile rows."""
    coords = np.asarray(coordinates, dtype=np.float64)
    identifiers = np.asarray(well_ids)
    if identifiers.ndim != 1 or identifiers.shape[0] != coords.shape[0]:
        raise ValueError("well_ids must be aligned with coordinates")
    if not np.isfinite(buffer_distance) or buffer_distance < 0.0:
        raise ValueError("buffer_distance must be non-negative and finite")
    unique_ids, inverse = np.unique(identifiers, return_inverse=True)
    representative = np.empty((unique_ids.size, coords.shape[1]), dtype=float)
    for index in range(unique_ids.size):
        representative[index] = coords[inverse == index].mean(axis=0)
    group_folds = spatial_block_cv(
        representative,
        n_folds=n_folds,
        block_type=block_type,
        grid_size=grid_size,
        seed=seed,
        block_size_km=block_size_km,
        buffer_distance=0.0,
        dims=dims,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_groups, test_groups in group_folds:
        train_rows = train_groups[inverse].copy()
        test_rows = test_groups[inverse]
        if buffer_distance > 0.0:
            tree = cKDTree(coords[test_rows][:, list(dims)])
            candidate_rows = np.flatnonzero(train_rows)
            distances, _ = tree.query(
                coords[candidate_rows][:, list(dims)],
                k=1,
            )
            near_ids = np.unique(
                identifiers[candidate_rows[distances < buffer_distance]]
            )
            if near_ids.size:
                train_rows[np.isin(identifiers, near_ids)] = False
        if not train_rows.any():
            raise ValueError(
                "spatial fold contains zero training wells after grouped "
                f"buffer_distance={buffer_distance:g}"
            )
        folds.append((train_rows, test_rows))
    return tuple(folds)


def _spatial_domain_bounds(
    assembled: AssembledInputs,
    spatial_domain: tuple[tuple[float, float], ...] | None = None,
) -> NDArray[np.float64]:
    """Freeze physical-coordinate bounds across final and validation fits."""
    if spatial_domain is not None:
        return np.asarray(spatial_domain, dtype=np.float64)
    coordinates = np.vstack(
        [assembled.well_coords, assembled.grid_coords]
    ).astype(np.float64, copy=False)
    return np.column_stack(
        [np.min(coordinates, axis=0), np.max(coordinates, axis=0)]
    )


def _select_component_stacking(  # noqa: PLR0913
    *,
    outcomes: np.ndarray,
    prior_probability: np.ndarray,
    full_probability: np.ndarray,
    validation_coordinates: np.ndarray,
    validation_well_ids: np.ndarray,
    minimum_wells: int,
) -> PredictiveStackingResult:
    """Select an update weight, retaining the prior under weak validation."""
    outcome_values = np.asarray(outcomes)
    prior_values = np.asarray(prior_probability)
    full_values = np.asarray(full_probability)
    coordinates = np.asarray(validation_coordinates, dtype=np.float64)
    well_ids = np.asarray(validation_well_ids)
    invalid_coordinates = (
        coordinates.ndim != _TWO_DIMENSIONS
        or outcome_values.ndim != 1
        or coordinates.shape[0] != outcome_values.size
        or coordinates.shape[1] == 0
        or not np.all(np.isfinite(coordinates))
    )
    invalid_ids = (
        well_ids.ndim != 1 or well_ids.shape[0] != outcome_values.size
    )
    if invalid_coordinates or invalid_ids:
        raise ValueError(
            "validation_coordinates must be a finite matrix aligned to outcomes"
        )
    if minimum_wells < 1:
        raise ValueError("minimum_wells must be positive")
    n_wells = int(np.unique(well_ids).size)
    evidence = PredictiveStackingEvidence(
        family="bernoulli",
        well_ids=well_ids.copy(),
        validation_coordinates=coordinates.copy(),
        outcomes=outcome_values.copy(),
        prior_probability=prior_values.copy(),
        full_probability=full_values.copy(),
    )
    if outcome_values.size == 0:
        if prior_values.shape != (0,) or full_values.shape != (0,):
            raise ValueError(
                "probability arrays must be empty when outcomes are empty"
            )
        return PredictiveStackingResult(
            weight=0.0,
            prior_log_score=None,
            full_log_score=None,
            selected_log_score=None,
            n_observations=0,
            n_wells=0,
            status="prior_retained_no_validation_wells",
            evidence=evidence,
        )
    selection = replace(
        select_predictive_stacking_weight(
            outcome_values,
            prior_values,
            full_values,
        ),
        n_wells=n_wells,
        evidence=evidence,
    )
    if n_wells >= minimum_wells:
        return selection
    return replace(
        selection,
        weight=0.0,
        selected_log_score=selection.prior_log_score,
        status="prior_retained_insufficient_validation_wells",
    )


def _select_component_density_stacking(  # noqa: PLR0913
    *,
    outcomes: np.ndarray,
    prior_probability: np.ndarray,
    full_probability: np.ndarray,
    prior_log_density: np.ndarray,
    full_log_density: np.ndarray,
    validation_coordinates: np.ndarray,
    validation_well_ids: np.ndarray,
    minimum_wells: int,
) -> PredictiveStackingResult:
    """Select a continuous predictive-density mixture with support guards."""
    prior = np.asarray(prior_log_density, dtype=np.float64)
    full = np.asarray(full_log_density, dtype=np.float64)
    prior_event = np.asarray(prior_probability, dtype=np.float64)
    full_event = np.asarray(full_probability, dtype=np.float64)
    outcome_values = np.asarray(outcomes, dtype=np.float64)
    coordinates = np.asarray(validation_coordinates, dtype=np.float64)
    well_ids = np.asarray(validation_well_ids)
    invalid_densities = (
        prior.ndim != 1
        or full.shape != prior.shape
        or prior_event.shape != prior.shape
        or full_event.shape != prior.shape
        or not np.all(np.isfinite(prior_event))
        or np.any((prior_event < 0.0) | (prior_event > 1.0))
        or not np.all(np.isfinite(full_event))
        or np.any((full_event < 0.0) | (full_event > 1.0))
    )
    invalid_coordinates = (
        coordinates.ndim != _TWO_DIMENSIONS
        or coordinates.shape[0] != prior.size
        or coordinates.shape[1] == 0
        or not np.all(np.isfinite(coordinates))
        or outcome_values.shape != prior.shape
        or not np.all(np.isfinite(outcome_values))
        or well_ids.ndim != 1
        or well_ids.shape[0] != prior.size
    )
    if invalid_densities or invalid_coordinates:
        raise ValueError(
            "predictive densities and validation coordinates must be aligned"
        )
    if minimum_wells < 1:
        raise ValueError("minimum_wells must be positive")
    n_wells = int(np.unique(well_ids).size)
    evidence = PredictiveStackingEvidence(
        family="gaussian",
        well_ids=well_ids.copy(),
        validation_coordinates=coordinates.copy(),
        outcomes=outcome_values.copy(),
        prior_probability=prior_event.copy(),
        full_probability=full_event.copy(),
        prior_log_density=prior.copy(),
        full_log_density=full.copy(),
    )
    if prior.size == 0:
        return PredictiveStackingResult(
            weight=0.0,
            prior_log_score=None,
            full_log_score=None,
            selected_log_score=None,
            n_observations=0,
            n_wells=0,
            status="prior_retained_no_validation_wells",
            evidence=evidence,
        )
    selection = replace(
        select_predictive_density_stacking_weight(prior, full),
        n_wells=n_wells,
        evidence=evidence,
    )
    if n_wells >= minimum_wells:
        return selection
    return replace(
        selection,
        weight=0.0,
        selected_log_score=selection.prior_log_score,
        status="prior_retained_insufficient_validation_wells",
    )


def _stacking_validation_mask(
    *,
    observed_mask: np.ndarray,
    well_depths_m: np.ndarray | None,
    component_name: str,
    validation_depths_m: Mapping[str, float],
) -> np.ndarray:
    """Restrict one component's validation rows to its declared depth."""
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.ndim != 1:
        raise ValueError("observed_mask must be a row vector")
    validation_depth = validation_depths_m.get(component_name)
    if validation_depth is None:
        return observed.copy()
    if well_depths_m is None:
        raise GEOPFAValueError(
            "target-depth stacking requires labels.depth_col"
        )
    depths = np.asarray(well_depths_m, dtype=np.float64)
    if depths.shape != observed.shape or not np.all(np.isfinite(depths)):
        raise ValueError(
            "well depths must be finite and aligned to observations"
        )
    selected = observed & np.isclose(
        depths,
        validation_depth,
        rtol=0.0,
        atol=1e-6,
    )
    return selected


def _validate_gblk_config(cfg: ProbabilisticConfig) -> None:
    """Require a valid config that explicitly selects the GBLK backend."""
    cfg.validate_raise()
    if cfg.inference.backend != "gblk":
        raise GEOPFAValueError(
            "direct GBLK runners require inference.backend='gblk'"
        )


def _is_gaussian_component(cfg: ProbabilisticConfig, name: str) -> bool:
    """Return whether a component explicitly declares a Gaussian response."""
    observation = cfg.labels.observation_models.get(name)
    return observation is not None and observation.family == "gaussian"


def _resolve_config_integer(
    requested: int | None,
    *,
    configured: int,
    argument: str,
    config_path: str,
) -> int:
    """Resolve a public runner argument without permitting config divergence."""
    if requested is None:
        return configured
    if isinstance(requested, bool | np.bool_) or not isinstance(
        requested, int | np.integer
    ):
        raise GEOPFAValueError(f"{argument} must be an integer")
    value = int(requested)
    if value != configured:
        raise GEOPFAValueError(
            f"{argument}={value} conflicts with {config_path}={configured}; "
            "change the config instead of overriding a config-driven run"
        )
    return configured


def _canonical_prediction_grid(
    adapter: PFAGridAdapter,
    cfg: ProbabilisticConfig,
) -> gpd.GeoDataFrame:
    """Return the one configured PFA grid used by every GBLK code path."""
    validate_declared_components(
        adapter,
        set(cfg.labels.label_columns) | set(cfg.alpha),
    )
    configured_names = tuple(
        name for name in adapter.components() if name in cfg.alpha
    )
    if not configured_names:
        raise GEOPFAValueError(
            "GBLK requires at least one configured analysis component"
        )
    return adapter.pr_norm(configured_names[0])


def _require_projected_metre_crs(
    grid_gdf: gpd.GeoDataFrame,
    *,
    context: str,
) -> None:
    """Require horizontal coordinates that make metre distances meaningful."""
    crs = grid_gdf.crs
    axes = () if crs is None else crs.axis_info[:2]
    if (
        crs is None
        or not crs.is_projected
        or len(axes) != _TWO_DIMENSIONS
        or any(
            axis.unit_name.lower() not in {"metre", "meter"}
            or not np.isclose(axis.unit_conversion_factor, 1.0)
            for axis in axes
        )
    ):
        raise GEOPFAValueError(
            f"{context} requires a projected metre-based CRS"
        )


def _component_probability_product(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multiply the columns of an ``(n, q)`` probability matrix."""
    return np.prod(probabilities, axis=1)


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
class _BayesianFamilyFit:
    """Prepared inputs shared by one Bayesian likelihood-family fit."""

    assembled: AssembledInputs
    evidence: JointEvidenceDesign
    bayes_config: GBLKBayesianConfig
    spatial_domain: tuple[tuple[float, float], ...] | None
    nc: int
    nlevel: int
    a_wght: float
    coordinate_scaling: str

    def backend_kwargs(self) -> dict[str, Any]:
        """Return the canonical backend keyword arguments."""
        assembled = self.assembled
        return {
            "component_names": assembled.component_names,
            "bayes_config": self.bayes_config,
            "observed_mask": assembled.observed_mask,
            "observation_weights": assembled.observation_weights,
            "observation_weight_semantics": (
                assembled.observation_weight_semantics
            ),
            "offsets": assembled.well_offsets,
            "grid_offsets": assembled.grid_offsets,
            "fixed_effects": self.evidence.train,
            "fixed_effects_grid": self.evidence.prediction,
            "fixed_precision": self.evidence.precision,
            "fixed_prior_mean": self.evidence.prior_mean,
            "spatial_domain": _spatial_domain_bounds(
                assembled,
                self.spatial_domain,
            ),
            "nc": self.nc,
            "nlevel": self.nlevel,
            "a_wght": self.a_wght,
            "coordinate_scaling": self.coordinate_scaling,
        }


@dataclass(frozen=True)
class _GBLKRun:
    """Inputs shared by every GBLK execution mode."""

    adapter: PFAGridAdapter
    grid: gpd.GeoDataFrame
    alphas: Mapping[str, AlphaCResult]
    prior_only_names: tuple[str, ...]
    assembled_groups: Mapping[str, AssembledInputs]
    config: ProbabilisticConfig
    nc: int
    a_wght: float | None


def _prepare_bayesian_family_fit(
    assembled: AssembledInputs,
    cfg: ProbabilisticConfig,
    bayes_config: GBLKBayesianConfig,
    *,
    lattice_controls: tuple[int, float | None],
    evidence: JointEvidenceDesign | None = None,
) -> _BayesianFamilyFit:
    """Prepare one likelihood-family fit from the canonical run config."""
    nc, a_wght = lattice_controls
    return _BayesianFamilyFit(
        assembled=assembled,
        evidence=(
            _prepare_joint_evidence(assembled, cfg)
            if evidence is None
            else evidence
        ),
        bayes_config=bayes_config,
        spatial_domain=cfg.spatial_field.spatial_domain,
        nc=nc,
        nlevel=cfg.spatial_field.n_levels,
        a_wght=_resolve_a_wght(assembled.well_coords.shape[1], a_wght),
        coordinate_scaling=cfg.spatial_field.coordinate_scaling,
    )


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
class _GaussianPredictiveResponseState:
    """Physical-unit inputs for one Gaussian predictive distribution."""

    latent_mean_draws: NDArray[np.float64]
    likelihood_sd_draws: NDArray[np.float64]
    prior_mean: NDArray[np.float64] | None
    prior_sd: NDArray[np.float64] | None
    seed: int


@dataclass(frozen=True)
class _GaussianPriorResponseState:
    """One aligned fixed Gaussian prior predictive distribution."""

    mean: NDArray[np.float64]
    sd: NDArray[np.float64]
    event_probability: NDArray[np.float64]
    event_threshold: float
    p_min: float
    p_max: float


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


def _prior_predictive_evidence_state(  # noqa: PLR0913
    adapter: PFAGridAdapter,
    component: str,
    alpha: AlphaCResult,
    cfg: ProbabilisticConfig,
    *,
    seed: int,
    reference_grid: gpd.GeoDataFrame | None = None,
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
        reference_grid=reference_grid,
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


def _prior_predictive_evidence_draws(  # noqa: PLR0913
    adapter: PFAGridAdapter,
    component: str,
    alpha: AlphaCResult,
    cfg: ProbabilisticConfig,
    *,
    seed: int,
    reference_grid: gpd.GeoDataFrame | None = None,
) -> PriorPredictiveEvidenceDraws:
    """Evaluate all draws through the existing in-memory API."""
    state = _prior_predictive_evidence_state(
        adapter,
        component,
        alpha,
        cfg,
        seed=seed,
        reference_grid=reference_grid,
    )
    component_grid = adapter.pr_norm(component)
    target_grid = component_grid if reference_grid is None else reference_grid
    grid_offset = _component_values_on_reference(
        target_grid,
        component_grid,
        alpha.grid_offset,
        context=f"component {component!r} prior grid",
    )
    eta_draws = (
        grid_offset[np.newaxis, :]
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
    """Resolve explicit proper coefficient priors."""
    base_precision = 1.0 / cfg.evidence.regularization.C
    default_precision: dict[str, float] = {}
    default_precision.update(cfg.evidence.regularization.per_feature_weights)
    default_precision.update(cfg.evidence.regularization.prior_precisions)
    default_means = dict(cfg.evidence.regularization.prior_means)
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
    well_ids: NDArray,
    *,
    component_name: str,
    layer_names: list[str],
    minimum_wells: int,
    standardization: str = "observed_labels",
    prediction_support: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    int,
    int,
]:
    """Apply fold-local per-feature scaling with neutral mean imputation."""
    finite_observed = observed[:, np.newaxis] & np.isfinite(x_train)
    finite_row_counts = finite_observed.sum(axis=0).astype(np.int64)
    finite_well_counts = np.asarray(
        [
            _unique_well_count(well_ids, finite_observed[:, index])
            for index in range(finite_observed.shape[1])
        ],
        dtype=np.int64,
    )
    unsupported = np.flatnonzero(finite_well_counts < minimum_wells)
    if unsupported.size:
        details = ", ".join(
            f"{layer_names[index]}={int(finite_well_counts[index])} wells"
            for index in unsupported
        )
        raise GEOPFAValueError(
            f"component {component_name!r} has fewer than {minimum_wells} "
            f"finite observed wells for evidence layer(s): {details}"
        )
    if standardization == "observed_labels":
        masked = np.where(finite_observed, x_train, np.nan)
    elif standardization == "prediction_support":
        support = (
            x_prediction
            if prediction_support is None
            else np.asarray(prediction_support, dtype=np.float64)
        )
        if (
            support.ndim != x_train.ndim
            or support.shape[1] != x_train.shape[1]
        ):
            raise GEOPFAValueError(
                f"component {component_name!r} prediction support shape is "
                "inconsistent"
            )
        masked = np.where(np.isfinite(support), support, np.nan)
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
    complete_well_count = _unique_well_count(
        well_ids,
        observed & np.all(np.isfinite(x_train), axis=1),
    )
    return (
        train_scaled,
        prediction_scaled,
        center,
        scale,
        finite_row_counts,
        finite_well_counts,
        complete_count,
        complete_well_count,
    )


def _prepare_joint_evidence_arrays(  # noqa: PLR0913, PLR0914
    *,
    component_names: tuple[str, ...],
    train_evidence: dict[str, NDArray[np.float64]],
    prediction_evidence: dict[str, NDArray[np.float64]],
    layer_names: dict[str, list[str]],
    y_train: NDArray[np.float64],
    observed_train: NDArray[np.bool_],
    well_ids: NDArray,
    cfg: ProbabilisticConfig,
    standardization_evidence: dict[str, NDArray[np.float64]] | None = None,
) -> JointEvidenceDesign:
    """Prepare leakage-safe component designs for one joint model fit."""
    n_train, n_components = y_train.shape
    if component_names != tuple(train_evidence) or n_components != len(
        component_names
    ):
        raise GEOPFAValueError(
            "joint evidence component ordering is inconsistent"
        )
    if standardization_evidence is not None and set(
        standardization_evidence
    ) != set(component_names):
        raise GEOPFAValueError(
            "joint evidence standardization support components are inconsistent"
        )
    n_prediction = len(prediction_evidence[component_names[0]])
    widths = [len(layer_names[name]) for name in component_names]
    max_width = max(widths, default=0)
    diagnostics: dict[str, dict[str, Any]] = {}
    for q_idx, name in enumerate(component_names):
        observed = observed_train[:, q_idx]
        observed_wells = _unique_well_count(well_ids, observed)
        if observed_wells < cfg.labels.min_wells_for_fit:
            raise GEOPFAValueError(
                f"component {name!r} has fewer than "
                f"{cfg.labels.min_wells_for_fit} observed wells"
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
            finite_row_counts,
            finite_well_counts,
            complete_count,
            complete_well_count,
        ) = _standardize_partial_evidence(
            x_train,
            x_prediction,
            observed,
            well_ids,
            component_name=name,
            layer_names=layer_names[name],
            minimum_wells=cfg.labels.min_wells_for_fit,
            standardization=cfg.evidence.standardization,
            prediction_support=(
                None
                if standardization_evidence is None
                else standardization_evidence[name]
            ),
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
            "evidence_well_n": observed_wells,
            "evidence_complete_row_n": complete_count,
            "evidence_complete_well_n": complete_well_count,
            "evidence_finite_per_feature": finite_row_counts.tolist(),
            "evidence_finite_wells_per_feature": finite_well_counts.tolist(),
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
        well_ids=assembled.well_ids,
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
        observed_wells = _unique_well_count(assembled.well_ids, observed)
        if observed_wells < cfg.labels.min_wells_for_fit:
            raise GEOPFAValueError(
                f"component {name!r} has fewer than "
                f"{cfg.labels.min_wells_for_fit} observed wells"
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
            finite_row_counts,
            finite_well_counts,
            complete_count,
            complete_well_count,
        ) = _standardize_partial_evidence(
            x_well,
            x_grid,
            observed,
            assembled.well_ids,
            component_name=name,
            layer_names=assembled.layer_names[name],
            minimum_wells=cfg.labels.min_wells_for_fit,
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
            "evidence_well_n": observed_wells,
            "evidence_complete_row_n": complete_count,
            "evidence_complete_well_n": complete_well_count,
            "evidence_finite_per_feature": finite_row_counts.tolist(),
            "evidence_finite_wells_per_feature": finite_well_counts.tolist(),
            "evidence_missing_value_policy": (
                "fold_local_feature_mean_zero_standardized_contribution"
            ),
        }
    return well_offsets, grid_offsets, diagnostics


def _run_gblk_bayesian(
    fit: _BayesianFamilyFit,
    grid_gdf: gpd.GeoDataFrame,
) -> tuple[
    dict[str, ComponentProbability],
    gpd.GeoDataFrame,
    dict[str, NDArray[np.float64]],
]:
    """Package predictions from the sole array-level Bayesian GBLK fitter."""
    assembled = fit.assembled
    bayes_config = fit.bayes_config
    fit_result = fit_gblk_bayesian_joint(
        assembled.well_coords,
        assembled.y,
        assembled.grid_coords,
        **fit.backend_kwargs(),
    )
    diagnostics = dict(fit_result.diagnostics)

    components: dict[str, ComponentProbability] = {}
    component_draws: dict[str, NDArray[np.float64]] = {}
    for q_idx, name in enumerate(assembled.component_names):
        component_diagnostics = dict(fit.evidence.diagnostics[name])
        width = len(assembled.layer_names.get(name, []))
        if fit_result.fixed_coef_draws is not None and width:
            coefficient_draws = fit_result.fixed_coef_draws[:, :width, q_idx]
            tail = (1.0 - bayes_config.ci_level) / 2.0
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
    return np.clip(
        probabilities,
        _OPEN_PROBABILITY_EPSILON,
        1.0 - _OPEN_PROBABILITY_EPSILON,
    )


def _gaussian_prior_exceedance_probability(
    *,
    mean: np.ndarray,
    sd: np.ndarray,
    threshold: float,
    p_min: float = _OPEN_PROBABILITY_EPSILON,
    p_max: float = 1.0 - _OPEN_PROBABILITY_EPSILON,
) -> NDArray[np.float64]:
    """Evaluate one Gaussian prior's threshold-exceedance probability."""
    mean_values = np.asarray(mean, dtype=np.float64)
    sd_values = np.asarray(sd, dtype=np.float64)
    invalid_arrays = (
        mean_values.shape != sd_values.shape
        or not np.all(np.isfinite(mean_values))
        or not np.all(np.isfinite(sd_values))
        or np.any(sd_values <= 0.0)
    )
    invalid_bounds = (
        not np.isfinite(p_min)
        or not np.isfinite(p_max)
        or not 0.0 < p_min < p_max < 1.0
    )
    if invalid_arrays or not np.isfinite(threshold) or invalid_bounds:
        raise RuntimeError(
            "Gaussian prior probability requires finite aligned means, "
            "positive standard deviations, a finite threshold, and valid "
            "open clipping bounds"
        )
    probability = ndtr((mean_values - float(threshold)) / sd_values)
    return np.clip(probability, p_min, p_max)


def _gaussian_prior_response_on_reference(
    reference_grid: gpd.GeoDataFrame,
    component_grid: gpd.GeoDataFrame,
    alpha: AlphaCResult,
    *,
    component_name: str,
) -> _GaussianPriorResponseState:
    """Align fixed Gaussian moments once and derive their event probability."""
    if alpha.latent_mean is None or alpha.latent_sd is None:
        raise RuntimeError(
            f"Gaussian prior-only component {component_name!r} requires "
            "configured response means and standard deviations"
        )
    if alpha.event_threshold is None or not np.isfinite(alpha.event_threshold):
        raise RuntimeError(
            f"Gaussian prior-only component {component_name!r} requires a "
            "finite event threshold"
        )
    try:
        p_min = float(alpha.provenance["p_min"])
        p_max = float(alpha.provenance["p_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Gaussian prior-only component {component_name!r} requires "
            "configured probability clipping bounds"
        ) from exc
    response_mean = _component_values_on_reference(
        reference_grid,
        component_grid,
        alpha.latent_mean,
        context=f"component {component_name!r} Gaussian prior mean",
    )
    response_sd = _component_values_on_reference(
        reference_grid,
        component_grid,
        alpha.latent_sd,
        context=f"component {component_name!r} Gaussian prior SD",
    )
    probability = _gaussian_prior_exceedance_probability(
        mean=response_mean,
        sd=response_sd,
        threshold=float(alpha.event_threshold),
        p_min=p_min,
        p_max=p_max,
    )
    return _GaussianPriorResponseState(
        mean=response_mean,
        sd=response_sd,
        event_probability=probability,
        event_threshold=float(alpha.event_threshold),
        p_min=p_min,
        p_max=p_max,
    )


def _package_gaussian_prior_response(
    probability: gpd.GeoDataFrame,
    *,
    response: _GaussianPriorResponseState,
    ci_level: float,
    component_name: str,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Attach an analytic physical-unit summary for one fixed Gaussian prior."""
    if not np.isfinite(ci_level) or not 0.0 < ci_level < 1.0:
        raise ValueError("Gaussian prior response ci_level must be in (0, 1)")
    if (
        response.mean.shape != (len(probability),)
        or response.sd.shape != response.mean.shape
        or response.event_probability.shape != response.mean.shape
    ):
        raise RuntimeError(
            f"Gaussian prior-only component {component_name!r} has an "
            "inconsistent aligned response shape"
        )
    z_score = float(ndtri((1.0 + ci_level) / 2.0))
    columns: dict[str, NDArray[np.float64]] = {
        "probability": response.event_probability,
        "response_predictive_mean": response.mean,
        "response_predictive_lo": response.mean - z_score * response.sd,
        "response_predictive_hi": response.mean + z_score * response.sd,
    }
    for interval_column in ("probability_lo", "probability_hi"):
        if interval_column in probability:
            columns[interval_column] = response.event_probability
    packaged = probability.assign(**columns)
    return packaged, {
        "observation_family": "gaussian",
        "event_threshold": response.event_threshold,
        "event_probability_estimand": (
            "clipped_configured_prior_predictive_response_exceedance"
        ),
        "event_probability_clipping_bounds": [
            response.p_min,
            response.p_max,
        ],
        "response_summary_estimand": "prior_predictive_response",
        "response_interval_method": "analytic_normal_quantile",
        "response_interval_level": float(ci_level),
        "response_interval_uncertainty_source": "configured_thermal_model_sd",
        "response_interval_includes_prior_uncertainty": True,
        "response_interval_includes_likelihood_variance": False,
    }


def _gaussian_predictive_log_density(
    fit_result: Any,
    *,
    outcomes_scaled: np.ndarray,
    component_index: int,
    response_scale: float,
) -> NDArray[np.float64]:
    """Evaluate the paired posterior predictive density in response units."""
    mean_draws = np.asarray(
        fit_result.response_draws[:, :, component_index], dtype=np.float64
    )
    precision_draws = np.asarray(
        fit_result.likelihood_precision_draws[:, component_index],
        dtype=np.float64,
    )
    outcomes = np.asarray(outcomes_scaled, dtype=np.float64)
    invalid_means = mean_draws.shape[0] == 0 or not np.all(
        np.isfinite(mean_draws)
    )
    invalid_outcomes = outcomes.shape != (mean_draws.shape[1],) or not np.all(
        np.isfinite(outcomes)
    )
    invalid_precision = (
        precision_draws.shape != (mean_draws.shape[0],)
        or not np.all(np.isfinite(precision_draws))
        or np.any(precision_draws <= 0.0)
    )
    invalid_scale = not np.isfinite(response_scale) or response_scale <= 0.0
    if invalid_means or invalid_outcomes or invalid_precision or invalid_scale:
        raise RuntimeError(
            "Gaussian predictive density requires finite aligned outcomes, "
            "means, positive likelihood precisions, and a positive response scale"
        )
    log_density_draws = (
        0.5 * (np.log(precision_draws)[:, np.newaxis] - np.log(2.0 * np.pi))
        - 0.5
        * precision_draws[:, np.newaxis]
        * (outcomes[np.newaxis, :] - mean_draws) ** 2
    )
    return (
        logsumexp(log_density_draws, axis=0)
        - np.log(mean_draws.shape[0])
        - np.log(response_scale)
    )


def _gaussian_predictive_response_summary(
    state: _GaussianPredictiveResponseState,
    *,
    weight: float,
    ci_level: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Summarize one prior/full posterior-predictive response mixture."""
    latent = np.asarray(state.latent_mean_draws, dtype=np.float64)
    likelihood_sd = np.asarray(state.likelihood_sd_draws, dtype=np.float64)
    invalid_latent = (
        latent.ndim != _TWO_DIMENSIONS
        or latent.shape[0] == 0
        or not np.all(np.isfinite(latent))
    )
    if invalid_latent or (
        likelihood_sd.shape != (latent.shape[0],)
        or not np.all(np.isfinite(likelihood_sd))
        or np.any(likelihood_sd <= 0.0)
    ):
        raise RuntimeError(
            "Gaussian predictive response summaries require finite latent-mean "
            "draws and one positive likelihood SD per draw"
        )
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(
            "Gaussian predictive mixture weight must be in [0, 1]"
        )
    if not np.isfinite(ci_level) or not 0.0 < ci_level < 1.0:
        raise ValueError("Gaussian predictive ci_level must be in (0, 1)")

    prior_mean = None
    prior_sd = None
    if weight < 1.0:
        if state.prior_mean is None or state.prior_sd is None:
            raise RuntimeError(
                "Gaussian predictive stacking requires finite prior response "
                "means and positive prior response SDs"
            )
        prior_mean = np.asarray(state.prior_mean, dtype=np.float64)
        prior_sd = np.asarray(state.prior_sd, dtype=np.float64)
        if (
            prior_mean.shape != (latent.shape[1],)
            or prior_sd.shape != prior_mean.shape
            or not np.all(np.isfinite(prior_mean))
            or not np.all(np.isfinite(prior_sd))
            or np.any(prior_sd <= 0.0)
        ):
            raise RuntimeError(
                "Gaussian predictive stacking requires finite prior response "
                "means and positive prior response SDs"
            )

    full_mean = latent.mean(axis=0)
    mean = (
        full_mean
        if weight == 1.0
        else ((1.0 - weight) * prior_mean + weight * full_mean)
    )
    tail = (1.0 - ci_level) / 2.0
    interval = np.empty((2, latent.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(state.seed)
    for start in range(0, latent.shape[1], _PREDICTIVE_SUMMARY_CHUNK_SIZE):
        stop = min(start + _PREDICTIVE_SUMMARY_CHUNK_SIZE, latent.shape[1])
        chunk_shape = (latent.shape[0], stop - start)
        if weight > 0.0:
            predictive = (
                latent[:, start:stop]
                + rng.standard_normal(chunk_shape)
                * likelihood_sd[:, np.newaxis]
            )
        if weight < 1.0:
            prior = (
                prior_mean[np.newaxis, start:stop]
                + rng.standard_normal(chunk_shape)
                * prior_sd[np.newaxis, start:stop]
            )
            if weight == 0.0:
                predictive = prior
            else:
                select_full = rng.random(chunk_shape) < weight
                predictive = np.where(select_full, predictive, prior)
        interval[:, start:stop] = np.quantile(
            predictive, [tail, 1.0 - tail], axis=0
        )
    return np.asarray(mean, dtype=np.float64), interval


def _run_gblk_gaussian_bayesian(  # noqa: PLR0914
    fit: _BayesianFamilyFit,
    grid_gdf: gpd.GeoDataFrame,
    alphas: Mapping[str, AlphaCResult],
    cfg: ProbabilisticConfig,
) -> tuple[
    dict[str, ComponentProbability],
    dict[str, NDArray[np.float64]],
    dict[str, _GaussianPredictiveResponseState],
]:
    """Fit continuous components and convert draws to event probabilities."""
    assembled = fit.assembled
    bayes_config = fit.bayes_config
    fit_result = fit_gblk_gaussian_bayesian_joint(
        assembled.well_coords,
        assembled.y,
        assembled.grid_coords,
        **fit.backend_kwargs(),
    )
    tail = (1.0 - bayes_config.ci_level) / 2.0
    components: dict[str, ComponentProbability] = {}
    component_draws: dict[str, NDArray[np.float64]] = {}
    response_states: dict[str, _GaussianPredictiveResponseState] = {}
    response_seeds = _spawn_child_seeds(
        bayes_config.seed, len(assembled.component_names)
    )
    for q_idx, (name, response_seed) in enumerate(
        zip(assembled.component_names, response_seeds, strict=True)
    ):
        observation = cfg.labels.observation_model_for(name)
        scale = float(observation.response_scale)
        alpha = alphas[name]
        if alpha.event_threshold is None:
            raise RuntimeError(
                f"Gaussian component {name!r} has no event threshold"
            )
        response_state = _GaussianPredictiveResponseState(
            latent_mean_draws=(
                np.asarray(fit_result.response_draws[:, :, q_idx]) * scale
            ),
            likelihood_sd_draws=(
                scale
                / np.sqrt(fit_result.likelihood_precision_draws[:, q_idx])
            ),
            prior_mean=(
                None
                if assembled.prior_response_mean_grid is None
                else assembled.prior_response_mean_grid[:, q_idx]
            ),
            prior_sd=(
                None
                if assembled.prior_response_sd_grid is None
                else assembled.prior_response_sd_grid[:, q_idx]
            ),
            seed=response_seed,
        )
        response_mean, response_interval = (
            _gaussian_predictive_response_summary(
                response_state,
                weight=1.0,
                ci_level=bayes_config.ci_level,
            )
        )
        event_draws = _gaussian_predictive_exceedance_draws(
            fit_result,
            component_index=q_idx,
            threshold_scaled=float(alpha.event_threshold) / scale,
        )
        probability_interval = np.quantile(
            event_draws, [tail, 1.0 - tail], axis=0
        )
        probability = (
            grid_gdf[["geometry"]]
            .copy()
            .assign(
                probability=event_draws.mean(axis=0),
                probability_lo=probability_interval[0],
                probability_hi=probability_interval[1],
                response_predictive_mean=response_mean,
                response_predictive_lo=response_interval[0],
                response_predictive_hi=response_interval[1],
            )
        )
        diagnostics = {
            **fit_result.diagnostics,
            **fit.evidence.diagnostics[name],
            "observation_family": "gaussian",
            "response_scale": scale,
            "event_threshold": float(alpha.event_threshold),
            "event_probability_estimand": (
                "posterior_predictive_response_exceedance"
            ),
            "response_summary_estimand": "posterior_predictive_response",
            "response_interval_includes_likelihood_variance": True,
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
        response_states[name] = response_state
    return components, component_draws, response_states


def _fitted_label_config(
    labels: LabelsConfig,
    fitted_components: Collection[str],
) -> LabelsConfig:
    """Restrict every component-keyed label field to fitted components."""
    fitted = frozenset(fitted_components)
    observation_weight_columns = {
        name: column
        for name, column in labels.observation_weight_columns.items()
        if name in fitted
    }
    return replace(
        labels,
        label_columns={
            name: column
            for name, column in labels.label_columns.items()
            if name in fitted
        },
        observation_models={
            name: model
            for name, model in labels.observation_models.items()
            if name in fitted
        },
        prior_response_mean_columns={
            name: column
            for name, column in labels.prior_response_mean_columns.items()
            if name in fitted
        },
        prior_response_sd_columns={
            name: column
            for name, column in labels.prior_response_sd_columns.items()
            if name in fitted
        },
        observation_weight_columns=observation_weight_columns,
        observation_weight_semantics=(
            labels.observation_weight_semantics
            if observation_weight_columns
            else None
        ),
    )


def _assemble_likelihood_groups(
    adapter: PFAGridAdapter,
    loaded_labels: LoadedLabels,
    fit_alphas: Mapping[str, AlphaCResult],
    cfg: ProbabilisticConfig,
    *,
    reference_grid: gpd.GeoDataFrame,
) -> dict[str, AssembledInputs]:
    """Assemble separate same-family fits for one mixed-response workflow."""
    grouped_names: dict[str, list[str]] = {}
    for name in fit_alphas:
        family = cfg.labels.observation_model_for(name).family
        grouped_names.setdefault(family, []).append(name)

    assembled_groups: dict[str, AssembledInputs] = {}
    for family, names in grouped_names.items():
        group_labels = _fitted_label_config(loaded_labels.config, names)
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
            reference_grid=reference_grid,
        )
        if family == "gaussian":
            if cfg.inference.predictive_stacking.enabled:
                if (
                    assembled.prior_response_mean_grid is None
                    or assembled.prior_response_sd_grid is None
                    or assembled.prior_response_mean_well is None
                    or assembled.prior_response_sd_well is None
                ):
                    raise RuntimeError(
                        "Gaussian predictive stacking requires prior "
                        "response moments"
                    )
                thresholds = tuple(
                    float(cfg.alpha[name].threshold)
                    for name in assembled.component_names
                )
                try:
                    probability_bounds = tuple(
                        (
                            float(fit_alphas[name].provenance["p_min"]),
                            float(fit_alphas[name].provenance["p_max"]),
                        )
                        for name in assembled.component_names
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Gaussian predictive stacking requires configured "
                        "probability clipping bounds"
                    ) from exc
                prior_probability_grid = np.column_stack(
                    [
                        _gaussian_prior_exceedance_probability(
                            mean=assembled.prior_response_mean_grid[:, q_idx],
                            sd=assembled.prior_response_sd_grid[:, q_idx],
                            threshold=threshold,
                            p_min=probability_bounds[q_idx][0],
                            p_max=probability_bounds[q_idx][1],
                        )
                        for q_idx, threshold in enumerate(thresholds)
                    ]
                )
                prior_probability_well = np.column_stack(
                    [
                        _gaussian_prior_exceedance_probability(
                            mean=assembled.prior_response_mean_well[:, q_idx],
                            sd=assembled.prior_response_sd_well[:, q_idx],
                            threshold=threshold,
                            p_min=probability_bounds[q_idx][0],
                            p_max=probability_bounds[q_idx][1],
                        )
                        for q_idx, threshold in enumerate(thresholds)
                    ]
                )
                assembled = replace(
                    assembled,
                    prior_probability_grid=prior_probability_grid,
                    prior_probability_well=prior_probability_well,
                )
            scales = np.asarray(
                [
                    cfg.labels.observation_model_for(name).response_scale
                    for name in assembled.component_names
                ],
                dtype=np.float64,
            )
            if assembled.prior_response_mean_well is None:
                raise RuntimeError(
                    "Gaussian components require prior response means at "
                    "observations"
                )
            assembled = replace(
                assembled,
                y=assembled.y / scales,
                well_offsets=assembled.prior_response_mean_well / scales,
            )
        assembled_groups[family] = assembled
    return assembled_groups


def _blocked_family_predictions(
    assembled: AssembledInputs,
    family: str,
    cfg: ProbabilisticConfig,
    *,
    nc: int,
    a_wght: float | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.bool_],
]:
    """Return leakage-safe out-of-fold predictions for one family."""
    predictions = np.full(assembled.y.shape, np.nan, dtype=np.float64)
    unsupported_rows = np.zeros(assembled.y.shape[0], dtype=bool)
    log_density = (
        np.full(assembled.y.shape, np.nan, dtype=np.float64)
        if family == "gaussian"
        else None
    )
    folds = _grouped_spatial_folds(
        assembled.well_coords,
        assembled.well_ids,
        n_folds=cfg.cross_validation.n_folds,
        block_type=cfg.cross_validation.block_type,
        grid_size=cfg.cross_validation.grid_size,
        seed=cfg.inference.gblk_bayesian.seed,
        block_size_km=cfg.cross_validation.block_size_km,
        buffer_distance=cfg.cross_validation.buffer_km * 1000.0,
    )
    fold_seeds = _spawn_child_seeds(
        cfg.inference.gblk_bayesian.seed,
        cfg.cross_validation.n_folds,
    )
    fold_stage_cfg = replace(
        cfg,
        labels=replace(
            cfg.labels,
            min_wells_for_fit=_stacking_training_minimum(cfg),
        ),
    )
    spatial_domain = _spatial_domain_bounds(
        assembled,
        cfg.spatial_field.spatial_domain,
    )
    for (train_mask, test_mask), fold_seed in zip(
        folds, fold_seeds, strict=True
    ):
        if not _fold_has_required_training_support(
            assembled,
            train_mask,
            minimum_wells=_stacking_training_minimum(cfg),
        ):
            unsupported_rows[test_mask] = True
            continue
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
            well_ids=assembled.well_ids[train_mask],
            cfg=fold_stage_cfg,
            standardization_evidence=(
                assembled.grid_evidence
                if cfg.evidence.standardization == "prediction_support"
                else None
            ),
        )
        common = {
            "component_names": assembled.component_names,
            "bayes_config": fold_cfg,
            "observed_mask": assembled.observed_mask[train_mask],
            "observation_weights": assembled.observation_weights[train_mask],
            "observation_weight_semantics": (
                assembled.observation_weight_semantics
            ),
            "offsets": assembled.well_offsets[train_mask],
            "grid_offsets": assembled.well_offsets[test_mask],
            "fixed_effects": evidence_design.train,
            "fixed_effects_grid": evidence_design.prediction,
            "fixed_precision": evidence_design.precision,
            "fixed_prior_mean": evidence_design.prior_mean,
            "spatial_domain": spatial_domain,
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
                assert log_density is not None
                log_density[test_mask, q_idx] = (
                    _gaussian_predictive_log_density(
                        fit,
                        outcomes_scaled=assembled.y[test_mask, q_idx],
                        component_index=q_idx,
                        response_scale=scale,
                    )
                )
        else:  # pragma: no cover - guarded by observation config
            raise AssertionError(f"unexpected response family {family!r}")
    return predictions, log_density, unsupported_rows


def _incomplete_spatial_cv_selection(  # noqa: PLR0913
    assembled: AssembledInputs,
    *,
    family: str,
    component_index: int,
    observed: NDArray[np.bool_],
    full_probability: NDArray[np.float64],
    full_log_density: NDArray[np.float64] | None,
    cfg: ProbabilisticConfig,
) -> PredictiveStackingResult:
    """Retain the prior when a buffered fold cannot support the family fit."""
    if assembled.prior_probability_well is None:
        raise RuntimeError("predictive stacking requires prior probabilities")
    name = assembled.component_names[component_index]
    outcomes = assembled.y[observed, component_index]
    evidence_kwargs: dict[str, Any]
    if family == "bernoulli":
        evidence_outcomes = outcomes.copy()
        evidence_kwargs = {
            "prior_probability": assembled.prior_probability_well[
                observed, component_index
            ].copy(),
            "full_probability": full_probability[
                observed, component_index
            ].copy(),
        }
    elif family == "gaussian":
        if (
            full_log_density is None
            or assembled.prior_response_mean_well is None
            or assembled.prior_response_sd_well is None
        ):
            raise RuntimeError(
                f"Gaussian component {name!r} lacks predictive-density "
                "inputs for stacking"
            )
        scale = float(cfg.labels.observation_model_for(name).response_scale)
        prior_mean = (
            assembled.prior_response_mean_well[observed, component_index]
            / scale
        )
        prior_sd = (
            assembled.prior_response_sd_well[observed, component_index] / scale
        )
        if (
            not np.all(np.isfinite(prior_mean))
            or not np.all(np.isfinite(prior_sd))
            or np.any(prior_sd <= 0.0)
        ):
            raise RuntimeError(
                f"Gaussian component {name!r} has invalid prior predictive moments"
            )
        prior_log_density = (
            -np.log(prior_sd)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * ((outcomes - prior_mean) / prior_sd) ** 2
            - np.log(scale)
        )
        evidence_outcomes = outcomes.copy() * scale
        evidence_kwargs = {
            "prior_probability": assembled.prior_probability_well[
                observed, component_index
            ].copy(),
            "full_probability": full_probability[
                observed, component_index
            ].copy(),
            "prior_log_density": prior_log_density,
            "full_log_density": full_log_density[
                observed, component_index
            ].copy(),
        }
    else:  # pragma: no cover - guarded by observation config
        raise AssertionError(f"unexpected response family {family!r}")
    return PredictiveStackingResult(
        weight=0.0,
        prior_log_score=None,
        full_log_score=None,
        selected_log_score=None,
        n_observations=int(outcomes.size),
        n_wells=_unique_well_count(assembled.well_ids, observed),
        status="prior_retained_incomplete_spatial_cv",
        evidence=PredictiveStackingEvidence(
            family=family,
            well_ids=assembled.well_ids[observed].copy(),
            validation_coordinates=assembled.well_coords[observed, :2].copy(),
            outcomes=evidence_outcomes,
            **evidence_kwargs,
        ),
    )


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
        (
            full_probability,
            full_log_density,
            unsupported_rows,
        ) = _blocked_family_predictions(
            assembled, family, cfg, nc=nc, a_wght=a_wght
        )
        for q_idx, name in enumerate(assembled.component_names):
            validation_depth = (
                cfg.inference.predictive_stacking.validation_depths_m.get(name)
            )
            observed = _stacking_validation_mask(
                observed_mask=assembled.observed_mask[:, q_idx],
                well_depths_m=assembled.well_depths_m,
                component_name=name,
                validation_depths_m=(
                    cfg.inference.predictive_stacking.validation_depths_m
                ),
            )
            incomplete_cv = bool(np.any(unsupported_rows & observed))
            if incomplete_cv:
                selection = _incomplete_spatial_cv_selection(
                    assembled,
                    family=family,
                    component_index=q_idx,
                    observed=observed,
                    full_probability=full_probability,
                    full_log_density=full_log_density,
                    cfg=cfg,
                )
                results[name] = replace(
                    selection,
                    validation_depth_m=validation_depth,
                )
                continue
            if not np.all(np.isfinite(full_probability[observed, q_idx])):
                raise RuntimeError(
                    f"component {name!r} has nonfinite out-of-fold predictions"
                )
            if (
                family == "gaussian"
                and full_log_density is not None
                and not np.all(np.isfinite(full_log_density[observed, q_idx]))
            ):
                raise RuntimeError(
                    f"component {name!r} has nonfinite out-of-fold log densities"
                )
            if family == "bernoulli":
                outcomes = assembled.y[observed, q_idx]
                selection = _select_component_stacking(
                    outcomes=outcomes,
                    prior_probability=assembled.prior_probability_well[
                        observed, q_idx
                    ],
                    full_probability=full_probability[observed, q_idx],
                    validation_coordinates=assembled.well_coords[observed, :2],
                    validation_well_ids=assembled.well_ids[observed],
                    minimum_wells=cfg.labels.min_wells_for_fit,
                )
            else:
                scale = float(
                    cfg.labels.observation_model_for(name).response_scale
                )
                if (
                    full_log_density is None
                    or assembled.prior_response_mean_well is None
                    or assembled.prior_response_sd_well is None
                ):
                    raise RuntimeError(
                        f"Gaussian component {name!r} lacks predictive-density "
                        "inputs for stacking"
                    )
                prior_mean = (
                    assembled.prior_response_mean_well[observed, q_idx] / scale
                )
                prior_sd = (
                    assembled.prior_response_sd_well[observed, q_idx] / scale
                )
                outcomes_scaled = assembled.y[observed, q_idx]
                if (
                    not np.all(np.isfinite(prior_mean))
                    or not np.all(np.isfinite(prior_sd))
                    or np.any(prior_sd <= 0.0)
                ):
                    raise RuntimeError(
                        f"Gaussian component {name!r} has invalid prior "
                        "predictive moments"
                    )
                prior_log_density = (
                    -np.log(prior_sd)
                    - 0.5 * np.log(2.0 * np.pi)
                    - 0.5 * ((outcomes_scaled - prior_mean) / prior_sd) ** 2
                    - np.log(scale)
                )
                selection = _select_component_density_stacking(
                    outcomes=outcomes_scaled * scale,
                    prior_probability=assembled.prior_probability_well[
                        observed, q_idx
                    ],
                    full_probability=full_probability[observed, q_idx],
                    prior_log_density=prior_log_density,
                    full_log_density=full_log_density[observed, q_idx],
                    validation_coordinates=assembled.well_coords[observed, :2],
                    validation_well_ids=assembled.well_ids[observed],
                    minimum_wells=cfg.labels.min_wells_for_fit,
                )
            results[name] = replace(
                selection,
                validation_depth_m=validation_depth,
            )
    return results


def _apply_componentwise_stacking(  # noqa: PLR0913
    components: dict[str, ComponentProbability],
    component_draws: dict[str, NDArray[np.float64]],
    assembled_groups: Mapping[str, AssembledInputs],
    stacking: Mapping[str, PredictiveStackingResult],
    gaussian_response_states: Mapping[str, _GaussianPredictiveResponseState],
    *,
    ci_level: float,
) -> None:
    """Apply selected component weights before the existing combination."""
    tail = (1.0 - ci_level) / 2.0
    for family, assembled in assembled_groups.items():
        if assembled.prior_probability_grid is None:
            raise RuntimeError(
                "predictive stacking requires prior probabilities on the grid"
            )
        for q_idx, name in enumerate(assembled.component_names):
            selection = stacking[name]
            full_update_draws = component_draws[name]
            draws = apply_predictive_stacking(
                assembled.prior_probability_grid[:, q_idx],
                full_update_draws,
                weight=selection.weight,
            )
            component_draws[name] = draws
            interval = np.quantile(draws, [tail, 1.0 - tail], axis=0)
            probability = components[name].probability.copy()
            probability["probability_prior"] = (
                assembled.prior_probability_grid[:, q_idx]
            )
            probability["probability_full_update"] = full_update_draws.mean(
                axis=0
            )
            probability["probability"] = draws.mean(axis=0)
            probability["probability_lo"] = interval[0]
            probability["probability_hi"] = interval[1]
            if family == "gaussian":
                if name not in gaussian_response_states:
                    raise RuntimeError(
                        f"Gaussian component {name!r} lacks predictive response state"
                    )
                response_mean, response_interval = (
                    _gaussian_predictive_response_summary(
                        gaussian_response_states[name],
                        weight=selection.weight,
                        ci_level=ci_level,
                    )
                )
                probability["response_predictive_mean"] = response_mean
                probability["response_predictive_lo"] = response_interval[0]
                probability["response_predictive_hi"] = response_interval[1]
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
                "predictive_stacking_n_wells": selection.n_wells,
                "predictive_stacking_status": selection.status,
                "predictive_stacking_score": (
                    "gaussian_log_predictive_density"
                    if family == "gaussian"
                    else "bernoulli_log_score"
                ),
                "predictive_stacking_validation": "blocked_out_of_fold",
                "predictive_stacking_validation_depth_m": (
                    selection.validation_depth_m
                ),
            }
            if family == "gaussian":
                diagnostics.update(
                    {
                        "response_summary_estimand": (
                            "stacked_posterior_predictive_response"
                        ),
                        "response_interval_includes_likelihood_variance": True,
                    }
                )
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
    fitted_response_family = metadata.get(
        "fitted_response_family", "bernoulli"
    )
    if fitted_response_family not in {"bernoulli", "gaussian"}:
        raise ValueError(
            "persisted fitted response family must be 'bernoulli' or "
            "'gaussian'"
        )
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
            response_family=fitted_response_family,
            likelihood_precision_draws=(
                None
                if fitted_response_family == "bernoulli"
                else np.asarray(persisted.arrays["likelihood_precision_draws"])
            ),
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


def _update_analysis_hash(
    digest: Any,
    name: str,
    payload: bytes,
) -> None:
    """Add one length-delimited field to an analysis-input digest."""
    for value in (name.encode("utf-8"), payload):
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)


def _update_analysis_json(digest: Any, name: str, value: Any) -> None:
    """Hash one strict canonical JSON field."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _update_analysis_hash(digest, name, payload)


def _update_analysis_array(
    digest: Any,
    name: str,
    values: np.ndarray | None,
    *,
    dtype: np.dtype[Any],
) -> None:
    """Hash one optional numeric array in a platform-independent encoding."""
    if values is None:
        _update_analysis_json(digest, name, None)
        return
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    if (
        np.issubdtype(array.dtype, np.floating)
        and not np.isfinite(array).all()
    ):
        raise ValueError(f"analysis input {name!r} must contain finite values")
    _update_analysis_json(
        digest,
        f"{name}.array",
        {"dtype": array.dtype.str, "shape": list(array.shape)},
    )
    _update_analysis_hash(digest, f"{name}.bytes", array.tobytes(order="C"))


def _canonical_well_identifiers(values: np.ndarray) -> list[list[Any]]:
    """Encode supported well identifiers without object-memory or repr hashing."""
    identifiers: list[list[Any]] = []
    for raw_value in np.asarray(values).tolist():
        value = (
            raw_value.item()
            if isinstance(raw_value, np.generic)
            else raw_value
        )
        if isinstance(value, bool):
            identifiers.append(["bool", value])
        elif isinstance(value, int):
            identifiers.append(["integer", str(value)])
        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("well identifiers cannot be non-finite")
            identifiers.append(["float", value.hex()])
        elif isinstance(value, str):
            identifiers.append(["string", value])
        else:
            raise TypeError(
                "well identifiers must be strings, integers, booleans, or "
                "finite floating-point values"
            )
    return identifiers


def _streaming_analysis_input_sha256(  # noqa: PLR0913
    *,
    grid_gdf: gpd.GeoDataFrame,
    ordered_names: tuple[str, ...],
    prior_names: tuple[str, ...],
    prior_predictor: np.ndarray,
    assembled: AssembledInputs | None,
    evidence_design: JointEvidenceDesign | None,
    prior_states: Mapping[str, PriorPredictiveEvidenceState | None],
    nc: int,
    a_wght: float | None,
) -> str:
    """Fingerprint every value that defines a streamed Bayesian analysis."""
    fitted_names = () if assembled is None else assembled.component_names
    if set(fitted_names).intersection(prior_names) or set(
        ordered_names
    ) != set(fitted_names).union(prior_names):
        raise ValueError(
            "streamed Bayesian fitted and prior-only component roles are inconsistent"
        )
    if set(prior_states) != set(prior_names):
        raise ValueError(
            "streamed Bayesian prior state does not match prior-only components"
        )
    roles = {
        name: (
            "joint_posterior"
            if name in fitted_names
            else (
                "fixed_prior_predictive"
                if prior_states[name] is None
                else "evidence_coefficient_prior_predictive"
            )
        )
        for name in ordered_names
    }
    digest = hashlib.sha256()
    _update_analysis_json(
        digest,
        "analysis",
        {
            "schema": "geopfa_streamed_bayesian_analysis_inputs_v1",
            "component_names": list(ordered_names),
            "component_roles": roles,
            "nc": int(nc),
            "a_wght": a_wght,
            "grid_crs": (
                None if grid_gdf.crs is None else grid_gdf.crs.to_string()
            ),
        },
    )
    _update_analysis_array(
        digest,
        "prediction_coordinates",
        extract_coordinates(grid_gdf),
        dtype=np.dtype("<f8"),
    )
    _update_analysis_array(
        digest,
        "prior_linear_predictor",
        prior_predictor,
        dtype=np.dtype("<f8"),
    )

    if assembled is None:
        _update_analysis_json(digest, "fitted_inputs", None)
    else:
        observed = np.asarray(assembled.observed_mask, dtype=bool)
        outcomes = np.where(observed, assembled.y, 0.0)
        _update_analysis_json(
            digest,
            "fitted_component_names",
            list(assembled.component_names),
        )
        _update_analysis_array(
            digest, "outcomes", outcomes, dtype=np.dtype("<f8")
        )
        _update_analysis_array(
            digest,
            "observed_mask",
            observed,
            dtype=np.dtype("u1"),
        )
        _update_analysis_array(
            digest,
            "observation_weights",
            assembled.observation_weights,
            dtype=np.dtype("<f8"),
        )
        _update_analysis_json(
            digest,
            "well_ids",
            _canonical_well_identifiers(assembled.well_ids),
        )
        for name, values in (
            ("well_coordinates", assembled.well_coords),
            ("model_grid_coordinates", assembled.grid_coords),
            ("well_offsets", assembled.well_offsets),
            ("grid_offsets", assembled.grid_offsets),
            ("well_depths_m", assembled.well_depths_m),
        ):
            _update_analysis_array(digest, name, values, dtype=np.dtype("<f8"))
        _update_analysis_json(
            digest,
            "evidence_layer_names",
            {
                name: list(assembled.layer_names[name])
                for name in assembled.component_names
            },
        )
        if evidence_design is None:
            raise RuntimeError(
                "fitted Bayesian components require an evidence design"
            )
        for name, values in (
            ("evidence_train", evidence_design.train),
            ("evidence_prediction", evidence_design.prediction),
            ("evidence_precision", evidence_design.precision),
            ("evidence_prior_mean", evidence_design.prior_mean),
        ):
            _update_analysis_array(digest, name, values, dtype=np.dtype("<f8"))

    for name in prior_names:
        state = prior_states[name]
        if state is None:
            _update_analysis_json(digest, f"prior_state.{name}", None)
            continue
        _update_analysis_json(
            digest,
            f"prior_state.{name}.feature_names",
            list(state.feature_names),
        )
        _update_analysis_array(
            digest,
            f"prior_state.{name}.standardized_evidence",
            state.standardized_evidence,
            dtype=np.dtype("<f8"),
        )
        _update_analysis_array(
            digest,
            f"prior_state.{name}.coefficient_draws",
            state.coefficient_draws,
            dtype=np.dtype("<f8"),
        )
    return digest.hexdigest()


def _run_gblk_bayesian_streaming(  # noqa: PLR0912, PLR0914, PLR0915
    run: _GBLKRun,
    *,
    evidence_design: JointEvidenceDesign | None,
    fitted_family: str,
    scope: str,
) -> ProbabilisticResult:
    """Fit once and persist paired grid draws without a full draw cube."""
    if fitted_family not in {"bernoulli", "gaussian"}:
        raise ValueError(
            "streamed Bayesian fitted family must be 'bernoulli' or 'gaussian'"
        )
    assembled = run.assembled_groups.get(fitted_family)
    grid_gdf = run.grid
    adapter = run.adapter
    alphas = run.alphas
    prior_only_names = run.prior_only_names
    cfg = run.config
    nc = run.nc
    a_wght = run.a_wght
    bayes_cfg = cfg.inference.gblk_bayesian
    ordered_names = tuple(sorted(alphas))
    has_gaussian_components = any(
        _is_gaussian_component(cfg, name) for name in ordered_names
    )
    component_index = {name: index for index, name in enumerate(ordered_names)}
    n_draws = bayes_cfg.n_draws
    n_cells = len(grid_gdf)
    q_total = len(ordered_names)
    prior_names = tuple(sorted(prior_only_names))
    gaussian_prior_responses: dict[str, _GaussianPriorResponseState] = {}
    prior_predictor_columns: list[NDArray[np.float64]] = []
    for name in ordered_names:
        if (
            fitted_family == "gaussian"
            and assembled is not None
            and name in assembled.component_names
        ):
            source_index = assembled.component_names.index(name)
            prior_predictor_columns.append(
                assembled.grid_offsets[:, source_index]
            )
        elif name in prior_names and _is_gaussian_component(cfg, name):
            response = _gaussian_prior_response_on_reference(
                grid_gdf,
                adapter.pr_norm(name),
                alphas[name],
                component_name=name,
            )
            gaussian_prior_responses[name] = response
            prior_predictor_columns.append(response.mean)
        else:
            prior_predictor_columns.append(
                _component_values_on_reference(
                    grid_gdf,
                    adapter.pr_norm(name),
                    alphas[name].grid_offset,
                    context=f"component {name!r} prior grid",
                )
            )
    prior_predictor = np.column_stack(prior_predictor_columns)
    if prior_predictor.shape != (n_cells, q_total) or not np.all(
        np.isfinite(prior_predictor)
    ):
        raise RuntimeError(
            "Bayesian component priors do not share one finite grid"
        )

    model_a_wght = (
        None
        if assembled is None
        else _resolve_a_wght(assembled.well_coords.shape[1], a_wght)
    )
    current_prior_states: dict[str, PriorPredictiveEvidenceState | None] = {}
    fitted_family_count = int(assembled is not None)
    child_seeds = _spawn_child_seeds(
        bayes_cfg.seed, fitted_family_count + len(prior_names)
    )
    fitted_seed = child_seeds[0] if fitted_family_count else None
    fitted_bayes_cfg = (
        bayes_cfg
        if fitted_seed is None
        else replace(bayes_cfg, seed=fitted_seed)
    )
    prior_seeds = child_seeds[fitted_family_count:]
    for name, component_seed in zip(prior_names, prior_seeds, strict=True):
        if cfg.alpha[name].use_evidence_prior:
            current_prior_states[name] = _prior_predictive_evidence_state(
                adapter,
                name,
                alphas[name],
                cfg,
                seed=component_seed,
                reference_grid=grid_gdf,
            )
        else:
            current_prior_states[name] = None
    analysis_input_sha256 = _streaming_analysis_input_sha256(
        grid_gdf=grid_gdf,
        ordered_names=ordered_names,
        prior_names=prior_names,
        prior_predictor=prior_predictor,
        assembled=assembled,
        evidence_design=evidence_design,
        prior_states=current_prior_states,
        nc=nc,
        a_wght=model_a_wght,
    )
    config_hash = hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    implementation_sha256 = _probabilistic_implementation_hash()
    persisted = load_posterior_draw_state(
        cfg.output_dir,
        grid_gdf,
        expected_config_hash=config_hash,
        expected_analysis_input_sha256=analysis_input_sha256,
        expected_implementation_sha256=implementation_sha256,
        expected_scope=scope,
    )
    if persisted is not None:
        if persisted.component_names != ordered_names:
            raise ValueError(
                "persisted posterior component ordering differs from this run"
            )
        state_arrays = dict(persisted.arrays)
        state_metadata = dict(persisted.metadata)
        prior_state_name = (
            "prior_linear_predictor"
            if has_gaussian_components
            else "prior_logit"
        )
        if not np.array_equal(state_arrays[prior_state_name], prior_predictor):
            raise ValueError(
                "persisted prior linear predictors differ from this run"
            )
        # The writer retains the prior predictor after fingerprint validation.
        # Keep that retained array independent of the disk mappings that must
        # be closed before an incomplete namespace can be renamed on Windows.
        state_arrays[prior_state_name] = prior_predictor
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
            family_fit = _prepare_bayesian_family_fit(
                assembled,
                cfg,
                fitted_bayes_cfg,
                lattice_controls=(nc, model_a_wght),
                evidence=evidence_design,
            )
            fitted_state = fit_gblk_bayesian_posterior_state(
                assembled.well_coords,
                assembled.y,
                assembled.grid_coords,
                **family_fit.backend_kwargs(),
                response_family=fitted_family,
            )
            evidence_diagnostics = family_fit.evidence.diagnostics

        prior_states = current_prior_states

        prior_state_name = (
            "prior_linear_predictor"
            if has_gaussian_components
            else "prior_logit"
        )
        state_arrays = {prior_state_name: prior_predictor}
        state_metadata = {
            "scope": scope,
            "config_hash": config_hash,
            "analysis_input_sha256": analysis_input_sha256,
            "implementation_sha256": implementation_sha256,
            "ci_level": bayes_cfg.ci_level,
            "component_roles": {},
            "fitted_component_names": (
                []
                if fitted_state is None
                else list(fitted_state.component_names)
            ),
            "fitted_response_family": fitted_family,
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
            if fitted_state.likelihood_precision_draws is not None:
                state_arrays["likelihood_precision_draws"] = (
                    fitted_state.likelihood_precision_draws
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
        else:
            model = "geopfa_probabilistic_gblk_paige_inla"
            state_metadata["model"] = (
                model
                if roles == {"joint_posterior"}
                else f"{model}_with_prior_predictive_components"
            )

    roles = set(state_metadata["component_roles"].values())
    fitted_model = "geopfa_probabilistic_gblk_paige_inla"
    expected_model = (
        "geopfa_probabilistic_prior_predictive"
        if "joint_posterior" not in roles
        else (
            fitted_model
            if roles == {"joint_posterior"}
            else f"{fitted_model}_with_prior_predictive_components"
        )
    )
    if state_metadata.get("model") != expected_model:
        raise ValueError(
            "persisted inference model is inconsistent with component roles; "
            "remove the stale generated draw directory and rerun"
        )

    component_models_list: list[dict[str, Any]] = []
    for name in ordered_names:
        if not _is_gaussian_component(cfg, name):
            component_models_list.append({"family": "bernoulli"})
            continue
        threshold = alphas[name].event_threshold
        is_fitted = (
            fitted_state is not None and name in fitted_state.component_names
        )
        response_scale = (
            cfg.labels.observation_model_for(name).response_scale
            if is_fitted
            else 1.0
        )
        if threshold is None or response_scale is None:
            raise RuntimeError(
                f"Gaussian component {name!r} requires an event threshold "
                "and response scale"
            )
        component_models_list.append(
            {
                "family": "gaussian",
                "event_threshold_scaled": float(threshold)
                / float(response_scale),
                **(
                    {
                        "probability_min": gaussian_prior_responses[
                            name
                        ].p_min,
                        "probability_max": gaussian_prior_responses[
                            name
                        ].p_max,
                    }
                    if name in gaussian_prior_responses
                    else {}
                ),
            }
        )
    component_models = tuple(component_models_list)
    draw_cell_indices = _posterior_draw_cell_indices(cfg, n_cells=n_cells)
    if draw_cell_indices is not None:
        state_metadata["posterior_draw_cell_selection"] = {
            "source_sha256": (cfg.outputs.posterior_draw_cell_indices_sha256),
            "n_cells": int(draw_cell_indices.size),
        }
    writer = PosteriorDrawBlockWriter(
        grid_gdf,
        cfg.output_dir,
        component_names=ordered_names,
        component_models=component_models,
        n_draws=n_draws,
        block_size=cfg.outputs.posterior_draw_block_size,
        seed=bayes_cfg.seed,
        combination_rule=cfg.combination.rule,
        scope=scope,
        state_arrays=state_arrays,
        state_metadata=state_metadata,
        draw_cell_indices=draw_cell_indices,
    )
    completed_starts = {start for start, _ in writer.completed_draw_ranges}
    for draw_start in range(0, n_draws, cfg.outputs.posterior_draw_block_size):
        if draw_start in completed_starts:
            continue
        draw_stop = min(
            draw_start + cfg.outputs.posterior_draw_block_size, n_draws
        )
        block_shape = (draw_stop - draw_start, n_cells, q_total)
        evidence_predictor = np.zeros(block_shape, dtype=np.float64)
        spatial_predictor = np.zeros(block_shape, dtype=np.float64)
        precision_shape = (
            block_shape
            if gaussian_prior_responses
            else (draw_stop - draw_start, q_total)
        )
        likelihood_precision = np.ones(precision_shape, dtype=np.float64)
        if fitted_state is not None:
            if fitted_family == "gaussian":
                fitted_block = project_gblk_gaussian_bayesian_draw_block(
                    fitted_state, draw_start, draw_stop
                )
                fitted_prior = fitted_block.prior_mean
                fitted_evidence = fitted_block.evidence_mean
                fitted_spatial = fitted_block.spatial_mean
                fitted_precision = fitted_block.likelihood_precision
            else:
                fitted_block = project_gblk_bayesian_draw_block(
                    fitted_state, draw_start, draw_stop
                )
                fitted_prior = fitted_block.prior_logit
                fitted_evidence = fitted_block.evidence_logit
                fitted_spatial = fitted_block.spatial_logit
                fitted_precision = None
            for source_index, name in enumerate(fitted_state.component_names):
                target_index = component_index[name]
                if not np.array_equal(
                    fitted_prior[:, source_index],
                    prior_predictor[:, target_index],
                ):
                    raise RuntimeError(
                        f"component {name!r} fitted and assembled prior "
                        "linear predictors differ"
                    )
                evidence_predictor[:, :, target_index] = fitted_evidence[
                    :, :, source_index
                ]
                spatial_predictor[:, :, target_index] = fitted_spatial[
                    :, :, source_index
                ]
                if fitted_precision is not None:
                    if likelihood_precision.ndim == _TWO_DIMENSIONS:
                        likelihood_precision[:, target_index] = (
                            fitted_precision[:, source_index]
                        )
                    else:
                        likelihood_precision[:, :, target_index] = (
                            fitted_precision[:, source_index, np.newaxis]
                        )
        for name in prior_names:
            prior_state = prior_states[name]
            if prior_state is None:
                continue
            target_index = component_index[name]
            evidence_predictor[:, :, target_index] = (
                prior_state.coefficient_draws[draw_start:draw_stop]
                @ prior_state.standardized_evidence.T
            )
        for name, response in gaussian_prior_responses.items():
            target_index = component_index[name]
            if likelihood_precision.ndim != _THREE_DIMENSIONS:
                raise RuntimeError(
                    "Gaussian prior response requires cell-specific precision"
                )
            likelihood_precision[:, :, target_index] = np.broadcast_to(
                1.0 / response.sd**2,
                (draw_stop - draw_start, n_cells),
            )
        eta = (
            prior_predictor[np.newaxis, :, :]
            + evidence_predictor
            + spatial_predictor
        )
        component_probability = np.empty(block_shape, dtype=np.float64)
        for target_index, model in enumerate(component_models):
            if model["family"] == "bernoulli":
                component_probability[:, :, target_index] = expit(
                    eta[:, :, target_index]
                )
                continue
            component_probability[:, :, target_index] = np.clip(
                ndtr(
                    (eta[:, :, target_index] - model["event_threshold_scaled"])
                    * np.sqrt(
                        likelihood_precision[:, target_index, np.newaxis]
                        if likelihood_precision.ndim == _TWO_DIMENSIONS
                        else likelihood_precision[:, :, target_index]
                    )
                ),
                float(model.get("probability_min", _OPEN_PROBABILITY_EPSILON)),
                float(
                    model.get(
                        "probability_max",
                        1.0 - _OPEN_PROBABILITY_EPSILON,
                    )
                ),
            )
        if not has_gaussian_components:
            writer.write_block(
                draw_start,
                component_probability=component_probability,
                prior_logit=prior_predictor,
                evidence_logit=evidence_predictor,
                spatial_logit=spatial_predictor,
            )
        else:
            writer.write_predictive_block(
                draw_start,
                component_probability=component_probability,
                prior_linear_predictor=prior_predictor,
                evidence_linear_predictor=evidence_predictor,
                spatial_linear_predictor=spatial_predictor,
                likelihood_precision=likelihood_precision,
            )
    resumed_incomplete_state = persisted is not None and not persisted.complete
    if resumed_incomplete_state:
        persisted.close()
    summary = writer.finalize(ci_level=bayes_cfg.ci_level)
    if resumed_incomplete_state:
        persisted = load_posterior_draw_state(
            cfg.output_dir,
            grid_gdf,
            expected_config_hash=config_hash,
            expected_analysis_input_sha256=analysis_input_sha256,
            expected_implementation_sha256=implementation_sha256,
            expected_scope=scope,
        )
        if persisted is None or not persisted.complete:
            raise RuntimeError(
                "finalized posterior state could not be reopened"
            )
        fitted_state, prior_states, evidence_diagnostics = (
            _restore_streaming_posterior_states(persisted, assembled)
        )

    components: dict[str, ComponentProbability] = {}
    for name in ordered_names:
        q_index = component_index[name]
        fixed_prior = False
        if fitted_state is not None and name in fitted_state.component_names:
            fitted_index = fitted_state.component_names.index(name)
            diagnostics = {
                **fitted_state.diagnostics,
                **evidence_diagnostics[name],
                "posterior_draw_storage": "incremental_hashed_blocks",
            }
            response_scale = 1.0
            if fitted_family == "gaussian":
                observation = cfg.labels.observation_model_for(name)
                if observation.response_scale is None:
                    raise RuntimeError(
                        f"Gaussian component {name!r} lacks a response scale"
                    )
                response_scale = float(observation.response_scale)
                threshold = alphas[name].event_threshold
                if threshold is None:
                    raise RuntimeError(
                        f"Gaussian component {name!r} lacks an event threshold"
                    )
                precision_draws = fitted_state.likelihood_precision_draws
                if precision_draws is None:
                    raise RuntimeError(
                        f"Gaussian component {name!r} lacks likelihood precision"
                    )
                diagnostics.update(
                    {
                        "observation_family": "gaussian",
                        "response_scale": response_scale,
                        "event_threshold": float(threshold),
                        "event_probability_estimand": (
                            "posterior_predictive_response_exceedance"
                        ),
                        "likelihood_sd_mean": float(
                            np.mean(
                                response_scale
                                / np.sqrt(precision_draws[:, fitted_index])
                            )
                        ),
                    }
                )
            width = len(assembled.layer_names.get(name, []))
            if fitted_state.fixed_coef_draws is not None and width:
                coefficient_draws = fitted_state.fixed_coef_draws[
                    :, :width, fitted_index
                ]
                if fitted_family == "gaussian":
                    coefficient_draws = np.multiply(
                        coefficient_draws, response_scale
                    )
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
                fixed_prior = True
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
        if fixed_prior:
            fixed_probability = (
                gaussian_prior_responses[name].event_probability
                if name in gaussian_prior_responses
                else expit(prior_predictor[:, q_index])
            )
            probability_values = fixed_probability
            probability_lower = fixed_probability
            probability_upper = fixed_probability
        else:
            probability_values = summary.component_mean[:, q_index]
            probability_lower = summary.component_interval[0, :, q_index]
            probability_upper = summary.component_interval[1, :, q_index]
        probability = (
            grid_gdf[["geometry"]]
            .copy()
            .assign(
                probability=probability_values,
                probability_lo=probability_lower,
                probability_hi=probability_upper,
            )
        )
        if name in prior_names and _is_gaussian_component(cfg, name):
            probability, gaussian_diagnostics = (
                _package_gaussian_prior_response(
                    probability,
                    response=gaussian_prior_responses[name],
                    ci_level=bayes_cfg.ci_level,
                    component_name=name,
                )
            )
            diagnostics = {**diagnostics, **gaussian_diagnostics}
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


def _evidence_prior_component(
    run: _GBLKRun,
    name: str,
    *,
    seed: int,
) -> tuple[ComponentProbability, NDArray[np.float64]]:
    """Package one Bayesian evidence-coefficient prior component."""
    cfg = run.config
    prior_draws = _prior_predictive_evidence_draws(
        run.adapter,
        name,
        run.alphas[name],
        cfg,
        seed=seed,
        reference_grid=run.grid,
    )
    draws = prior_draws.probability_draws
    tail = (1.0 - cfg.inference.gblk_bayesian.ci_level) / 2.0
    interval = np.quantile(draws, [tail, 1.0 - tail], axis=0)
    probability = (
        run.grid[["geometry"]]
        .copy()
        .assign(
            probability=draws.mean(axis=0),
            probability_lo=interval[0],
            probability_hi=interval[1],
        )
    )
    if _is_gaussian_component(cfg, name):
        raise RuntimeError(
            f"Gaussian prior-only component {name!r} lacks an aligned "
            "response state"
        )
    return (
        ComponentProbability(
            probability=probability,
            model=prior_draws,
            feature_names=prior_draws.feature_names,
            diagnostics=prior_draws.diagnostics,
        ),
        draws,
    )


def _prior_component(
    run: _GBLKRun,
    name: str,
    *,
    seed: int | None = None,
) -> tuple[ComponentProbability, NDArray[np.float64] | None]:
    """Package one fixed prior component for deterministic or Bayesian runs."""
    cfg = run.config
    alpha = run.alphas[name]
    bayesian = seed is not None
    if bayesian and cfg.alpha[name].use_evidence_prior:
        return _evidence_prior_component(run, name, seed=seed)

    response: _GaussianPriorResponseState | None = None
    if _is_gaussian_component(cfg, name):
        response = _gaussian_prior_response_on_reference(
            run.grid,
            run.adapter.pr_norm(name),
            alpha,
            component_name=name,
        )
        baseline = response.event_probability
    else:
        baseline = expit(
            _component_values_on_reference(
                run.grid,
                run.adapter.pr_norm(name),
                alpha.grid_offset,
                context=f"component {name!r} prior grid",
            )
        )
    diagnostics = {
        "inference_role": (
            "fixed_prior_predictive" if bayesian else "prior_predictive"
        ),
        "outcome_update": False,
        "spatial_field_included": False,
        "alpha_provenance": alpha.provenance,
    }
    draws = None
    interval = None
    if bayesian:
        n_draws = cfg.inference.gblk_bayesian.n_draws
        diagnostics["n_draws"] = n_draws
        draws = np.broadcast_to(baseline, (n_draws, baseline.size)).copy()
        interval = np.broadcast_to(baseline, (2, baseline.size))

    probability = run.grid[["geometry"]].copy()
    probability["probability"] = baseline
    if interval is not None:
        probability["probability_lo"] = interval[0]
        probability["probability_hi"] = interval[1]
    if response is not None:
        probability, gaussian_diagnostics = _package_gaussian_prior_response(
            probability,
            response=response,
            ci_level=cfg.inference.gblk_bayesian.ci_level,
            component_name=name,
        )
        diagnostics = {**diagnostics, **gaussian_diagnostics}
    return (
        ComponentProbability(
            probability=probability,
            model=None,
            feature_names=(),
            diagnostics=diagnostics,
        ),
        draws,
    )


def _fit_bayesian_groups(
    run: _GBLKRun,
    family_configs: Mapping[str, GBLKBayesianConfig],
) -> tuple[
    dict[str, ComponentProbability],
    dict[str, NDArray[np.float64]],
    dict[str, _GaussianPredictiveResponseState],
]:
    """Fit each observed likelihood family through its typed backend."""
    components: dict[str, ComponentProbability] = {}
    draws: dict[str, NDArray[np.float64]] = {}
    gaussian_states: dict[str, _GaussianPredictiveResponseState] = {}
    for family, assembled in sorted(run.assembled_groups.items()):
        fit = _prepare_bayesian_family_fit(
            assembled,
            run.config,
            family_configs[family],
            lattice_controls=(run.nc, run.a_wght),
        )
        if family == "gaussian":
            family_components, family_draws, gaussian_states = (
                _run_gblk_gaussian_bayesian(
                    fit,
                    run.grid,
                    run.alphas,
                    run.config,
                )
            )
        else:
            family_components, _, family_draws = _run_gblk_bayesian(
                fit, run.grid
            )
        components.update(family_components)
        draws.update(family_draws)
    return components, draws, gaussian_states


def _bayesian_seed_plan(
    run: _GBLKRun,
) -> tuple[
    dict[str, GBLKBayesianConfig],
    tuple[tuple[str, int], ...],
]:
    """Allocate deterministic child seeds to fitted and prior components."""
    families = tuple(sorted(run.assembled_groups))
    prior_names = tuple(sorted(run.prior_only_names))
    seeds = _spawn_child_seeds(
        run.config.inference.gblk_bayesian.seed,
        len(families) + len(prior_names),
    )
    family_configs = {
        family: replace(run.config.inference.gblk_bayesian, seed=seed)
        for family, seed in zip(families, seeds[: len(families)], strict=True)
    }
    return family_configs, tuple(
        zip(prior_names, seeds[len(families) :], strict=True)
    )


def _run_gblk_bayesian_memory(run: _GBLKRun) -> ProbabilisticResult:
    """Run the in-memory Bayesian path after common input preparation."""
    cfg = run.config
    family_configs, prior_seeds = _bayesian_seed_plan(run)
    components, component_draws, gaussian_states = _fit_bayesian_groups(
        run, family_configs
    )
    for name, seed in prior_seeds:
        component, draws = _prior_component(run, name, seed=seed)
        if draws is None:  # pragma: no cover - Bayesian helper contract
            raise RuntimeError("Bayesian prior component lacks draws")
        components[name] = component
        component_draws[name] = draws

    stacking: dict[str, PredictiveStackingResult] = {}
    if cfg.inference.predictive_stacking.enabled:
        stacking = _estimate_predictive_stacking(
            run.assembled_groups,
            cfg,
            nc=run.nc,
            a_wght=run.a_wght,
        )
        _apply_componentwise_stacking(
            components,
            component_draws,
            run.assembled_groups,
            stacking,
            gaussian_states,
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
    joint_draws = _component_probability_product(
        paired_draws.reshape(-1, paired_draws.shape[2]),
    ).reshape(paired_draws.shape[:2])
    tail = (1.0 - cfg.inference.gblk_bayesian.ci_level) / 2.0
    joint_interval = np.quantile(joint_draws, [tail, 1.0 - tail], axis=0)
    combined = (
        run.grid[["geometry"]]
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
        predictive_stacking=stacking,
        config=cfg,
        skipped=False,
    )


def _run_gblk_deterministic(
    run: _GBLKRun,
    *,
    max_outer_iter: int,
    irls_max_iter: int,
) -> ProbabilisticResult:
    """Run the deterministic MAP path after common input preparation."""
    cfg = run.config
    assembled = run.assembled_groups.get("bernoulli")
    components: dict[str, ComponentProbability] = {}
    if assembled is not None and assembled.component_names:
        if not assembled.observed_mask.any():
            raise GEOPFAValueError(
                "no component-label observations are available for the joint GBLK fit"
            )
        evidence = _prepare_joint_evidence(assembled, cfg)
        well_offsets = assembled.well_offsets
        grid_offsets = assembled.grid_offsets
        evidence_diagnostics = evidence.diagnostics
        fit_result = None
        if cfg.spatial_field.enabled:
            fit_result = fit_gblk_joint(
                assembled.well_coords,
                assembled.y,
                assembled.grid_coords,
                component_names=assembled.component_names,
                labeled_mask=np.any(assembled.observed_mask, axis=1),
                observed_mask=assembled.observed_mask,
                offsets=well_offsets,
                grid_offsets=grid_offsets,
                fixed_effects=evidence.train,
                fixed_effects_grid=evidence.prediction,
                fixed_precision=evidence.precision,
                fixed_prior_mean=evidence.prior_mean,
                spatial_domain=_spatial_domain_bounds(
                    assembled,
                    cfg.spatial_field.spatial_domain,
                ),
                nc=run.nc,
                nlevel=cfg.spatial_field.n_levels,
                a_wght=_resolve_a_wght(
                    assembled.well_coords.shape[1], run.a_wght
                ),
                coordinate_scaling=cfg.spatial_field.coordinate_scaling,
                max_outer_iter=max_outer_iter,
                irls_max_iter=irls_max_iter,
            )
            if fit_result.fit.fixed_coef is not None:
                for index, name in enumerate(assembled.component_names):
                    width = len(assembled.layer_names.get(name, []))
                    if width:
                        evidence_diagnostics[name]["evidence_beta"] = (
                            np.asarray(
                                fit_result.fit.fixed_coef[:width, index]
                            )
                            .astype(float)
                            .tolist()
                        )
        else:
            well_offsets, grid_offsets, evidence_diagnostics = (
                _fit_evidence_only_offsets(assembled, cfg)
            )
        for index, name in enumerate(assembled.component_names):
            probability = run.grid[["geometry"]].copy()
            probability["probability"] = (
                fit_result.p_q_grid[:, index].astype(float)
                if fit_result is not None
                else expit(grid_offsets[:, index])
            )
            components[name] = ComponentProbability(
                probability=probability,
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

    for name in run.prior_only_names:
        components[name], _ = _prior_component(run, name)
    combined = run.grid[["geometry"]].copy()
    component_probability = np.column_stack(
        [
            components[name].probability["probability"].to_numpy(dtype=float)
            for name in sorted(components)
        ]
    )
    combined["probability"] = _component_probability_product(
        component_probability
    )
    return ProbabilisticResult(
        components=components,
        combined=combined,
        config=cfg,
        skipped=False,
    )


def run_gblk_probabilistic(  # noqa: PLR0913
    pfa: dict,
    cfg: ProbabilisticConfig,
    *,
    criteria: str = "geologic",
    nc: int | None = None,
    a_wght: float | None = None,
    max_outer_iter: int = 100,
    irls_max_iter: int = 50,
    posterior_scope: str = "baseline",
) -> ProbabilisticResult:
    """Run the joint GBLK probabilistic method end-to-end.

    Assembles per-component inputs (labels, alpha offsets, evidence),
    fits each same-family component group jointly via the GBLK engine, and packages the
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
        Optional assertion of the configured LatticeKrig centers per dimension.
        When omitted, ``spatial_field.lattice_centers_per_dimension`` is used;
        a conflicting value is rejected.
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
    nc = _resolve_config_integer(
        nc,
        configured=cfg.spatial_field.lattice_centers_per_dimension,
        argument="nc",
        config_path="spatial_field.lattice_centers_per_dimension",
    )
    if cfg.labels.pu_mode == "nnpu":
        raise GEOPFAValueError(
            "nnPU is currently supported by the sequential outcome model only; "
            "a Bernoulli GBLK likelihood cannot treat unlabeled rows as negatives"
        )

    adapter = PFAGridAdapter(pfa, criteria=criteria, dimensions=cfg.dimensions)
    grid_gdf = _canonical_prediction_grid(adapter, cfg)
    uses_metric_stacking_controls = (
        cfg.inference.predictive_stacking.enabled
        and (
            cfg.cross_validation.buffer_km > 0.0
            or cfg.cross_validation.block_size_km is not None
        )
    )
    if (
        cfg.spatial_field.enabled
        and cfg.spatial_field.coordinate_scaling == "physical_isotropic"
    ) or uses_metric_stacking_controls:
        _require_projected_metre_crs(
            grid_gdf,
            context=(
                "physical-isotropic spatial scaling and kilometre-based "
                "spatial validation controls"
            ),
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
        fitted_labels = LoadedLabels(
            gdf=loaded_labels.gdf,
            config=_fitted_label_config(
                loaded_labels.config,
                fit_alphas,
            ),
        )
        assembled_groups = _assemble_likelihood_groups(
            adapter,
            fitted_labels,
            fit_alphas,
            cfg,
            reference_grid=grid_gdf,
        )
    if not assembled_groups and not prior_only_names:
        raise GEOPFAValueError(
            "GBLK requires at least one labeled or force-prior-predictive component"
        )
    run = _GBLKRun(
        adapter=adapter,
        grid=grid_gdf,
        alphas=alphas,
        prior_only_names=prior_only_names,
        assembled_groups=assembled_groups,
        config=cfg,
        nc=nc,
        a_wght=a_wght,
    )

    if cfg.inference.gblk_bayesian.enabled:
        if assembled_groups and not cfg.spatial_field.enabled:
            raise GEOPFAValueError(
                "Bayesian GBLK requires spatial_field.enabled=True"
            )
        if cfg.outputs.posterior_draw_blocks:
            if len(assembled_groups) > 1:
                raise GEOPFAValueError(
                    "streamed posterior blocks require all fitted components "
                    "to use one likelihood family"
                )
            fitted_family = (
                next(iter(assembled_groups))
                if assembled_groups
                else "bernoulli"
            )
            streamed_assembled = assembled_groups.get(fitted_family)
            evidence_design = (
                None
                if streamed_assembled is None
                else _prepare_joint_evidence(streamed_assembled, cfg)
            )
            return _run_gblk_bayesian_streaming(
                run,
                evidence_design=evidence_design,
                fitted_family=fitted_family,
                scope=posterior_scope,
            )
        return _run_gblk_bayesian_memory(run)

    return _run_gblk_deterministic(
        run,
        max_outer_iter=max_outer_iter,
        irls_max_iter=irls_max_iter,
    )


def _assemble_from_config(
    pfa: dict, cfg: ProbabilisticConfig, criteria: str
) -> AssembledInputs:
    """Build :class:`AssembledInputs` from a ``pfa`` dict and config."""
    adapter = PFAGridAdapter(pfa, criteria=criteria, dimensions=cfg.dimensions)
    grid_gdf = _canonical_prediction_grid(adapter, cfg)
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
        config=_fitted_label_config(loaded_labels.config, fit_alphas),
    )
    return assemble_gblk_inputs(
        adapter,
        fitted_labels,
        fit_alphas,
        evidence_config=cfg.evidence,
        reference_grid=grid_gdf,
    )


def _cv_evidence_only_offsets(  # noqa: PLR0913, PLR0914
    assembled: AssembledInputs,
    row_idx: NDArray[np.intp],
    component_indices: tuple[int, ...],
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
    *,
    cfg: ProbabilisticConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit the evidence-only CV ablation with componentwise missingness."""
    train_rows = row_idx[train_mask]
    test_rows = row_idx[test_mask]
    train_offsets = assembled.well_offsets[train_rows][
        :, component_indices
    ].copy()
    test_offsets = assembled.well_offsets[test_rows][
        :, component_indices
    ].copy()
    for local_idx, q_idx in enumerate(component_indices):
        name = assembled.component_names[q_idx]
        x_all = np.asarray(assembled.evidence[name][row_idx], dtype=np.float64)
        if x_all.shape[1] == 0:
            continue
        x_train = x_all[train_mask]
        x_test = x_all[test_mask]
        observed_train = assembled.observed_mask[train_rows, q_idx]
        train_scaled, test_scaled, *_ = _standardize_partial_evidence(
            x_train,
            x_test,
            observed_train,
            assembled.well_ids[train_rows],
            component_name=name,
            layer_names=assembled.layer_names[name],
            minimum_wells=cfg.labels.min_wells_for_fit,
            standardization=cfg.evidence.standardization,
            prediction_support=(
                assembled.grid_evidence[name]
                if cfg.evidence.standardization == "prediction_support"
                else None
            ),
        )
        weights, prior_means = _resolved_evidence_prior(
            cfg, assembled.layer_names[name], component=name
        )
        fit = _fit_offset_logit(
            train_scaled[observed_train],
            assembled.y[train_rows, q_idx][observed_train],
            train_offsets[observed_train, local_idx],
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
        well_ids=assembled.well_ids[train_rows],
        cfg=cfg,
        standardization_evidence=(
            {
                name: np.asarray(
                    assembled.grid_evidence[name], dtype=np.float64
                )
                for name in names
            }
            if cfg.evidence.standardization == "prediction_support"
            else None
        ),
    )


def _make_fit_fn(  # noqa: PLR0913
    assembled: AssembledInputs,
    row_idx: NDArray[np.intp],
    component_indices: tuple[int, ...],
    *,
    cfg: ProbabilisticConfig,
    nc: int,
    nlevel: int,
    a_wght: float | None,
    max_outer_iter: int,
    irls_max_iter: int,
):
    """Build one full-component fold fit returning all test probabilities."""
    coords_lab = assembled.well_coords[row_idx]
    y_lab = assembled.y[row_idx][:, component_indices]
    component_names = tuple(
        assembled.component_names[q] for q in component_indices
    )
    spatial_domain = _spatial_domain_bounds(
        assembled,
        cfg.spatial_field.spatial_domain,
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
            return expit(test_offsets)
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
        observed_train = assembled.observed_mask[row_idx[train_mask]][
            :, component_indices
        ]
        model_a_wght = _resolve_a_wght(coords_lab.shape[1], a_wght)
        fit = fit_gblk_joint(
            coords_lab[train_mask],
            y_lab[train_mask],
            coords_lab[test_mask],
            component_names=component_names,
            labeled_mask=np.any(observed_train, axis=1),
            observed_mask=observed_train,
            offsets=train_offsets,
            grid_offsets=test_offsets,
            fixed_effects=evidence_design.train,
            fixed_effects_grid=evidence_design.prediction,
            fixed_precision=evidence_design.precision,
            fixed_prior_mean=evidence_design.prior_mean,
            spatial_domain=spatial_domain,
            nc=nc,
            nlevel=nlevel,
            a_wght=model_a_wght,
            coordinate_scaling=cfg.spatial_field.coordinate_scaling,
            max_outer_iter=max_outer_iter,
            irls_max_iter=irls_max_iter,
        )
        return np.asarray(fit.p_q_grid, dtype=np.float64)

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
    n_bins: int | None = None,
    random_state: int = 0,
    nc: int | None = None,
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
    n_bins : int, optional
        Optional assertion of ``calibration.n_bins``. The configured value is
        used when omitted; a conflicting value is rejected.
    random_state : int, default=0
        Seed for the block permutation; also used to seed each per-target
        call so all results share fold assignments.
    nc : int, optional
        Optional assertion of
        ``spatial_field.lattice_centers_per_dimension``. The configured value
        is used when omitted; a conflicting value is rejected.
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
        uses complete cases for all fitted components. This controls reported
        metrics only; every fold still refits the complete same-family model.
        Defaults to ``None``, which computes every component plus ``"joint"``.

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
    One grouped spatial fold plan is built over the union of component-label
    observations. Each fold refits the complete same-family model with its
    componentwise observation mask. Per-component diagnostics then slice the
    shared out-of-fold predictions to that component's observed rows; joint
    diagnostics slice complete-case rows and use element-wise products across
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
    n_bins = _resolve_config_integer(
        n_bins,
        configured=cfg.calibration.n_bins,
        argument="n_bins",
        config_path="calibration.n_bins",
    )
    nc = _resolve_config_integer(
        nc,
        configured=cfg.spatial_field.lattice_centers_per_dimension,
        argument="nc",
        config_path="spatial_field.lattice_centers_per_dimension",
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
        or (
            cfg.spatial_field.enabled
            and cfg.spatial_field.coordinate_scaling == "physical_isotropic"
        )
    ):
        adapter = PFAGridAdapter(
            pfa, criteria=criteria, dimensions=cfg.dimensions
        )
        _require_projected_metre_crs(
            _canonical_prediction_grid(adapter, cfg),
            context=(
                "cross-validation distance controls and physical-isotropic "
                "spatial scaling"
            ),
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

    if not requested_names:
        return {}

    for q_idx, name in enumerate(assembled.component_names):
        n_observed_wells = _unique_well_count(
            assembled.well_ids,
            assembled.observed_mask[:, q_idx],
        )
        if n_observed_wells < cfg.labels.min_wells_for_fit:
            raise GEOPFAValueError(
                f"full calibration model component {name!r} has fewer than "
                f"{cfg.labels.min_wells_for_fit} observed wells "
                f"({n_observed_wells})"
            )
        values = assembled.y[assembled.observed_mask[:, q_idx], q_idx]
        if np.unique(values).size < 2:  # noqa: PLR2004
            raise GEOPFAValueError(
                f"full calibration model component {name!r} has only one "
                "observed label class"
            )
        if name in requested_names and n_observed_wells < effective_n_folds:
            raise GEOPFAValueError(
                f"component {name!r} has fewer than {effective_n_folds} "
                f"observed wells ({n_observed_wells}); calibration "
                "cross-validation is not identifiable"
            )

    complete_mask = np.all(assembled.observed_mask, axis=1)
    complete_idx = np.flatnonzero(complete_mask).astype(np.intp)
    if "joint" in requested_names:
        complete_wells = _unique_well_count(
            assembled.well_ids,
            complete_mask,
        )
        if complete_wells < effective_n_folds:
            raise GEOPFAValueError(
                f"joint target has fewer than {effective_n_folds} "
                f"complete-case wells ({complete_wells}); calibration "
                "cross-validation is not identifiable"
            )
        joint_values = np.prod(assembled.y[complete_idx], axis=1)
        if np.unique(joint_values).size < 2:  # noqa: PLR2004
            raise GEOPFAValueError(
                "joint target has only one observed label class; calibration "
                "cross-validation is not identifiable"
            )

    union_mask = np.any(assembled.observed_mask, axis=1)
    union_idx = np.flatnonzero(union_mask).astype(np.intp)
    union_splits = tuple(
        _grouped_spatial_folds(
            assembled.well_coords[union_idx],
            assembled.well_ids[union_idx],
            n_folds=effective_n_folds,
            block_type=cfg.cross_validation.block_type,
            grid_size=cfg.cross_validation.grid_size,
            seed=random_state,
            block_size_km=cfg.cross_validation.block_size_km,
            buffer_distance=cfg.cross_validation.buffer_km * 1000.0,
            dims=(0, 1) if dims is None else dims,
        )
    )
    union_fold_ids = np.full(union_idx.size, -1, dtype=np.intp)
    for fold, (_train_mask, test_mask) in enumerate(union_splits):
        if np.any(union_fold_ids[test_mask] >= 0):
            raise RuntimeError("union calibration folds overlap")
        union_fold_ids[test_mask] = fold
    if np.any(union_fold_ids < 0):
        raise RuntimeError("union calibration folds do not cover every well")

    component_indices = tuple(range(len(assembled.component_names)))
    fit_fn = _make_fit_fn(
        assembled,
        union_idx,
        component_indices,
        cfg=cfg,
        nc=nc,
        nlevel=cfg.spatial_field.n_levels,
        a_wght=a_wght,
        max_outer_iter=max_outer_iter,
        irls_max_iter=irls_max_iter,
    )
    oof_probability = np.full(
        (union_idx.size, len(component_indices)), np.nan, dtype=np.float64
    )
    for fold, (train_mask, test_mask) in enumerate(union_splits):
        prediction = np.asarray(
            fit_fn(train_mask, test_mask), dtype=np.float64
        )
        expected_shape = (int(np.sum(test_mask)), len(component_indices))
        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"full calibration fit returned {prediction.shape} for fold "
                f"{fold}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(prediction)) or np.any(
            (prediction < 0.0) | (prediction > 1.0)
        ):
            raise RuntimeError(
                f"full calibration fit returned invalid probabilities in fold {fold}"
            )
        oof_probability[test_mask] = prediction
    if not np.all(np.isfinite(oof_probability)):
        raise RuntimeError("full calibration fit left missing OOF predictions")

    def score_target(
        labels: NDArray[np.float64],
        probability: NDArray[np.float64],
        target_positions: NDArray[np.intp],
        *,
        target_name: str,
    ) -> CalibrationCVResult:
        target_splits: list[tuple[NDArray[np.bool_], NDArray[np.bool_]]] = []
        for fold, (train_mask, test_mask) in enumerate(union_splits):
            target_train = train_mask[target_positions]
            target_test = test_mask[target_positions]
            if not target_train.any() or not target_test.any():
                raise GEOPFAValueError(
                    f"target {target_name!r} has no observed "
                    f"{'training' if not target_train.any() else 'test'} "
                    f"wells in union calibration fold {fold}"
                )
            target_splits.append((target_train, target_test))

        def cached_fit(
            _train_mask: NDArray[np.bool_], test_mask: NDArray[np.bool_]
        ) -> NDArray[np.float64]:
            return probability[test_mask]

        return _calibration_cv_from_splits(
            cached_fit,
            labels,
            splits=target_splits,
            fold_ids=union_fold_ids[target_positions],
            n_bins=n_bins,
        )

    union_observed = assembled.observed_mask[union_idx]
    results: dict[str, CalibrationCVResult] = {}
    for q_idx, name in enumerate(assembled.component_names):
        if name not in requested_names:
            continue
        target_positions = np.flatnonzero(union_observed[:, q_idx]).astype(
            np.intp
        )
        results[name] = score_target(
            assembled.y[union_idx[target_positions], q_idx].astype(np.float64),
            oof_probability[target_positions, q_idx],
            target_positions,
            target_name=name,
        )

    if "joint" in requested_names:
        target_positions = np.flatnonzero(
            np.all(union_observed, axis=1)
        ).astype(np.intp)
        joint_probability = _component_probability_product(oof_probability)
        results["joint"] = score_target(
            np.prod(assembled.y[union_idx[target_positions]], axis=1).astype(
                np.float64
            ),
            joint_probability[target_positions],
            target_positions,
            target_name="joint",
        )

    return results


__all__ = [
    "run_gblk_calibration_cv",
    "run_gblk_probabilistic",
]
