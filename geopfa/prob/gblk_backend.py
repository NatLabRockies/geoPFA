"""Generalized Bayesian LatticeKrig (GBLK) inference backend for ``geopfa.prob``.

This module is the geoPFA *application adapter* onto the generic,
application-agnostic :mod:`latticekrigx.glk` engine.  It maps geoPFA-native
inputs -- per-component well labels, physics-informed prior offsets
(``alpha_c``), and a prediction grid -- onto the generic joint multivariate GLM
:func:`latticekrigx.glk.joint.fit_joint`, which fits all components **jointly**
(coupled through a scale-specific cross-component correlation matrix ``Omega``
and shared hyperparameters) rather than independently.

Compared with the sequential per-component logistic + sparse-GPy field path, the
GBLK backend provides a joint multivariate fit with an *estimated* cross-component
dependence ``Omega`` (not assumed independent), a multiresolution LatticeKrig
latent field with sparse SAR precision, and a conditional plug-in co-occurrence
surface ``p_joint = prod_q p_q`` computed from the jointly estimated component
marginals. The current joint training path materializes its assembled design and
is therefore not advertised as sample-count-independent streaming. This exported
MAP quantity is not a posterior expectation of the joint event.

Everything geothermal/PFA-specific stays in geoPFA; this adapter only calls
public ``latticekrigx.glk`` APIs.  It requires the dedicated ``gblk`` pixi
environment (``pixi run -e gblk``), which provides ``latticekrigx`` and
``scikit-sparse``.

Notes
-----
Offsets are on the linear-predictor (logit) scale:
``eta_q = o_q + X_q beta_q + Phi c_q``. Pass geoPFA ``alpha_c`` logit offsets
as ``o_q`` and evidence arrays through the fixed-effect arguments so ``beta_q``
and ``c_q`` share one likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from latticekrigx.glk.joint import JointResult

    from geopfa.prob.config import GBLKBayesianConfig


__all__ = [
    "GBLKBayesianDrawBlock",
    "GBLKBayesianFitResult",
    "GBLKBayesianPosteriorState",
    "GBLKFitResult",
    "build_lkinfo",
    "build_lkinfo_2d",
    "build_paige_prior",
    "fit_gblk_bayesian_joint",
    "fit_gblk_bayesian_posterior_state",
    "fit_gblk_joint",
    "project_gblk_bayesian_draw_block",
    "scale_spatial_coordinates",
]

_ARRAY_NDIM = 2
_THREE_D = 3
_MAX_PAIGE_COMPONENTS = 2
_SPATIAL_DIMS = {2, 3}
_BAYESIAN_PREDICTION_CHUNK_SIZE = 10_000


@dataclass
class GBLKFitResult:
    """Result of a joint GBLK fit over ``Q`` components on a prediction grid.

    Attributes
    ----------
    component_names
        Ordered component names matching the columns of ``p_q_grid``.
    p_q_grid
        Per-component predicted probability at grid locations, shape ``(G, Q)``.
    p_joint_grid
        Joint co-occurrence probability at grid locations, shape ``(G,)``.
    omega
        Estimated cross-component correlation matrix, shape ``(Q, Q)``.
    coef
        Latent field coefficients from the joint fit, shape ``(M, Q)``.
    fit
        The raw :class:`latticekrigx.glk.joint.JointResult`.
    diagnostics
        Auxiliary metadata (sizes, hyperparameters, basis dimension).
    """

    component_names: tuple[str, ...]
    p_q_grid: NDArray[np.float64]
    p_joint_grid: NDArray[np.float64]
    omega: NDArray[np.float64]
    coef: NDArray[np.float64]
    fit: JointResult
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GBLKBayesianFitResult:
    """Posterior predictions from the canonical Bayesian GBLK fitter.

    The probability draws and their summaries all come from the same paired
    posterior draws.  ``p_joint_grid`` is therefore
    ``E[prod_q p_q | data]`` rather than a product of posterior means.
    """

    component_names: tuple[str, ...]
    p_q_grid: NDArray[np.float64]
    p_q_interval: NDArray[np.float64]
    p_q_draws: NDArray[np.float64]
    p_joint_grid: NDArray[np.float64]
    p_joint_interval: NDArray[np.float64]
    p_joint_draws: NDArray[np.float64]
    fixed_coef_draws: NDArray[np.float64] | None
    fit: JointResult
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GBLKBayesianPosteriorState:
    """Immutable coefficient draws and prediction inputs for block projection."""

    component_names: tuple[str, ...]
    coefficient_draws: NDArray[np.float64]
    fixed_coef_draws: NDArray[np.float64] | None
    grid_model: NDArray[np.float64]
    grid_offsets: NDArray[np.float64]
    fixed_design_grid: NDArray[np.float64] | None
    lkinfo: Any
    fit: JointResult
    diagnostics: dict[str, Any]
    projected_probability_draws: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class GBLKBayesianDrawBlock:
    """One paired posterior block with an auditable logit decomposition."""

    draw_start: int
    draw_stop: int
    component_probability: NDArray[np.float64]
    prior_logit: NDArray[np.float64]
    evidence_logit: NDArray[np.float64]
    spatial_logit: NDArray[np.float64]


def build_paige_prior(config: GBLKBayesianConfig) -> Any:
    """Construct LatticeKrigX's published Paige ELK prior specification."""
    from latticekrigx.glk.bayes import PaigeELKPrior  # noqa: PLC0415

    return PaigeELKPrior(
        cor_scale_median=config.cor_scale_median,
        spatial_sd_u=config.spatial_sd_u,
        spatial_sd_alpha=config.spatial_sd_tail_probability,
        dirichlet_concentration=config.dirichlet_concentration,
    )


def build_lkinfo(  # noqa: PLR0913
    coords_spatial: NDArray[np.float64],
    *,
    nc: int = 6,
    nlevel: int = 1,
    a_wght: float = 4.5,
    overlap: float = 2.5,
    pad: float = 1e-3,
) -> Any:
    """Build a multiresolution 2-D or volumetric 3-D ``LKInfo``.

    Parameters
    ----------
    coords_spatial
        ``(n, D)`` coordinates with ``D`` equal to 2 or 3.
    nc
        Number of lattice centers per dimension.
    a_wght
        SAR center weight (``> 4`` for a positive-definite first-order rectangle
        precision).
    overlap
        Basis overlap factor.
    pad
        Domain padding added around the coordinate bounding box.

    Returns
    -------
    latticekrigx.model.config.LKInfo
        Single-level rectangular LatticeKrig configuration.
    """
    from latticekrigx.model.config import lk_setup  # noqa: PLC0415

    coords = np.asarray(coords_spatial, dtype=np.float64)
    if coords.ndim != _ARRAY_NDIM or coords.shape[1] not in _SPATIAL_DIMS:
        raise ValueError(
            f"coords_spatial must be (n, 2) or (n, 3); got {coords.shape}"
        )
    ndim = int(coords.shape[1])
    if not np.all(np.isfinite(coords)) or coords.shape[0] < 1:
        raise ValueError("coords_spatial must contain finite coordinates")
    if (
        isinstance(nlevel, bool | np.bool_)
        or not isinstance(nlevel, Integral)
        or nlevel < 1
    ):
        raise ValueError("nlevel must be a positive integer")
    if ndim == 3 and a_wght <= 6.0:  # noqa: PLR2004
        raise ValueError("a_wght must be greater than 6 for LKBox")
    lo = coords.min(axis=0) - pad
    hi = coords.max(axis=0) + pad
    return lk_setup(
        coords,
        nlevel=int(nlevel),
        nc=nc,
        geometry="LKRectangle" if ndim == _ARRAY_NDIM else "LKBox",
        a_wght=a_wght,
        normalize=False,
        overlap=overlap,
        range_domain=np.column_stack([lo, hi]).tolist(),
        m=0,
    )


def build_lkinfo_2d(  # noqa: PLR0913
    coords_xy: NDArray[np.float64],
    *,
    nc: int = 6,
    nlevel: int = 1,
    a_wght: float = 4.5,
    overlap: float = 2.5,
    pad: float = 1e-3,
) -> Any:
    """Build a 2-D ``LKInfo`` while preserving the existing public helper."""
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:  # noqa: PLR2004
        raise ValueError(f"coords_xy must be (n, 2); got {coords.shape}")
    return build_lkinfo(
        coords,
        nc=nc,
        nlevel=nlevel,
        a_wght=a_wght,
        overlap=overlap,
        pad=pad,
    )


def scale_spatial_coordinates(
    coords: NDArray[np.float64],
    grid: NDArray[np.float64],
    *,
    mode: str = "axis_range",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Apply one auditable coordinate transform to training and grid points."""
    train = np.asarray(coords, dtype=np.float64)
    prediction = np.asarray(grid, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] not in _SPATIAL_DIMS:  # noqa: PLR2004
        raise ValueError(f"coords must be (n, 2) or (n, 3); got {train.shape}")
    if prediction.ndim != 2 or prediction.shape[1] != train.shape[1]:  # noqa: PLR2004
        raise ValueError(
            "grid must have the same coordinate dimension as coords; "
            f"got {prediction.shape} and {train.shape}"
        )
    all_coords = np.vstack([train, prediction])
    if not np.all(np.isfinite(all_coords)):
        raise ValueError("coords and grid must contain only finite values")
    lower = all_coords.min(axis=0)
    raw_span = all_coords.max(axis=0) - lower
    if np.any(raw_span <= 0.0):
        raise ValueError(
            "each spatial coordinate axis must have positive span"
        )
    if mode == "axis_range":
        scale = raw_span
    elif mode == "physical_isotropic":
        common_scale = float(np.max(raw_span[:2]))
        scale = np.full(train.shape[1], common_scale, dtype=np.float64)
    else:
        raise ValueError(
            "coordinate scaling mode must be 'axis_range' or 'physical_isotropic'"
        )
    return (train - lower) / scale, (prediction - lower) / scale, lower, scale


def _prepare_joint_designs(
    fixed_effects: NDArray[np.float64] | None,
    fixed_effects_grid: NDArray[np.float64] | None,
    *,
    n_train: int,
    n_grid: int,
    n_components: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Translate geoPFA component-last designs to LatticeKrigX ordering."""
    if fixed_effects is None:
        if fixed_effects_grid is not None:
            raise ValueError("fixed_effects_grid requires fixed_effects")
        return None, None

    train = np.asarray(fixed_effects, dtype=np.float64)
    if train.ndim != _THREE_D or train.shape[0] != n_train:
        raise ValueError(
            "fixed_effects must have geoPFA shape "
            f"({n_train}, p, {n_components}); got {train.shape}"
        )
    p = int(train.shape[1])
    if (
        train.shape[2] != n_components
        or p < 1
        or not np.all(np.isfinite(train))
    ):
        raise ValueError(
            "fixed_effects must be finite with geoPFA shape "
            f"({n_train}, p, {n_components}) and p > 0"
        )
    prediction = np.asarray(fixed_effects_grid, dtype=np.float64)
    expected_prediction = (n_grid, p, n_components)
    if prediction.shape != expected_prediction or not np.all(
        np.isfinite(prediction)
    ):
        raise ValueError(
            "fixed_effects_grid must be finite with geoPFA shape "
            f"{expected_prediction}; got {prediction.shape}"
        )
    # geoPFA stores component last; LatticeKrigX accepts component-specific
    # designs as (n, Q, p).
    return np.moveaxis(train, 2, 1), np.moveaxis(prediction, 2, 1)


def _build_joint_coefficient_prior(
    fixed_design: NDArray[np.float64] | None,
    fixed_precision: NDArray[np.float64] | None,
    fixed_prior_mean: NDArray[np.float64] | None,
    *,
    n_components: int,
) -> Any | None:
    """Build the proper empirical-Bayes coefficient prior for a joint design."""
    import scipy.sparse as sp  # noqa: PLC0415

    from latticekrigx.glk import GaussianCoefficientPrior  # noqa: PLC0415

    if fixed_design is None:
        if fixed_precision is not None or fixed_prior_mean is not None:
            raise ValueError("coefficient-prior arrays require fixed_effects")
        return None

    p = int(fixed_design.shape[2])
    precision = np.asarray(fixed_precision, dtype=np.float64)
    mean = np.asarray(fixed_prior_mean, dtype=np.float64)
    expected_prior = (p, n_components)
    if (
        precision.shape != expected_prior
        or mean.shape != expected_prior
        or not np.all(np.isfinite(precision))
        or np.any(precision <= 0.0)
        or not np.all(np.isfinite(mean))
    ):
        raise ValueError(
            "fixed_precision and fixed_prior_mean must have shape "
            f"{expected_prior}; precisions must be positive and all values finite"
        )

    prior = GaussianCoefficientPrior(
        mean=mean.T.reshape(-1),
        precision=sp.diags(precision.T.reshape(-1), format="csc"),
    )
    return prior


def fit_gblk_bayesian_posterior_state(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    coords_spatial: NDArray[np.float64],
    labels: NDArray[np.float64],
    grid_spatial: NDArray[np.float64],
    *,
    component_names: tuple[str, ...] | list[str],
    bayes_config: GBLKBayesianConfig,
    labeled_mask: NDArray[np.bool_] | None = None,
    observed_mask: NDArray[np.bool_] | None = None,
    offsets: NDArray[np.float64] | None = None,
    grid_offsets: NDArray[np.float64] | None = None,
    fixed_effects: NDArray[np.float64] | None = None,
    fixed_effects_grid: NDArray[np.float64] | None = None,
    fixed_precision: NDArray[np.float64] | None = None,
    fixed_prior_mean: NDArray[np.float64] | None = None,
    nc: int = 6,
    nlevel: int = 1,
    a_wght: float = 4.5,
    coordinate_scaling: str = "axis_range",
) -> GBLKBayesianPosteriorState:
    """Fit geoPFA's canonical Bayesian GBLK model without a full draw cube.

    This is the array-level entry point shared by the config-driven workflow
    and manuscript simulations. It selects LatticeKrigX's public
    ``fit_joint(..., inference="inla")`` Paige model and summarizes the paired
    posterior coefficient draws returned by that model. Supplied alpha offsets
    are fixed. Grid probabilities are projected separately so publication runs
    can bound memory by a configured draw block.

    The LatticeKrigX Paige interface supports one or two components on either
    a 2-D ``LKRectangle`` or a 3-D ``LKBox``. Independent component-specific
    Gaussian coefficient priors are passed through the public joint interface.
    """
    import scipy.sparse as sp  # noqa: PLC0415

    from latticekrigx.basis.assembly import compute_basis  # noqa: PLC0415
    from latticekrigx.glk.families import Bernoulli  # noqa: PLC0415
    from latticekrigx.glk.joint import fit_joint  # noqa: PLC0415
    from latticekrigx.glk.links import LogitLink  # noqa: PLC0415
    from latticekrigx.model.config import lk_setup  # noqa: PLC0415

    coords = np.asarray(coords_spatial, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    grid = np.asarray(grid_spatial, dtype=np.float64)
    names = tuple(component_names)
    if y.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"labels must be 2-D (n, Q); got {y.shape}")
    n, n_components = y.shape
    if coords.ndim != _ARRAY_NDIM or coords.shape[1] not in _SPATIAL_DIMS:
        raise ValueError(
            "the canonical LatticeKrigX Paige/INLA model requires "
            f"coords_spatial with shape (n, 2) or (n, 3); got {coords.shape}"
        )
    if coords.shape[0] != n:
        raise ValueError(
            f"coords_spatial rows ({coords.shape[0]}) != labels rows ({n})"
        )
    if grid.ndim != 2 or grid.shape[1] != coords.shape[1]:  # noqa: PLR2004
        raise ValueError(
            "grid_spatial must have the same coordinate dimension as "
            f"coords_spatial; got {grid.shape} and {coords.shape}"
        )
    if len(names) != n_components:
        raise ValueError(
            f"component_names length ({len(names)}) != Q ({n_components})"
        )
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(grid)):
        raise ValueError("training and prediction coordinates must be finite")
    if n_components not in {1, 2}:
        raise ValueError(
            "the canonical LatticeKrigX Paige/INLA model supports one or two "
            f"components; got {n_components}"
        )
    if n_components == _MAX_PAIGE_COMPONENTS and (
        bayes_config.kleiber_r0 is None or bayes_config.kleiber_r1 is None
    ):
        raise ValueError(
            "bivariate Paige/INLA requires frozen kleiber_r0 and kleiber_r1 "
            "profile parameters"
        )
    spatial_dimension = int(coords.shape[1])
    stability_baseline = 2.0 * spatial_dimension
    if a_wght <= stability_baseline:
        raise ValueError(
            f"a_wght must be greater than {stability_baseline:g} for "
            f"{spatial_dimension}-D Paige inference"
        )

    if observed_mask is None:
        observed = np.ones_like(y, dtype=bool)
    else:
        observed = np.asarray(observed_mask)
        if observed.shape != y.shape or observed.dtype != np.bool_:
            raise ValueError(
                "observed_mask must be a boolean array with shape "
                f"{y.shape}; got {observed.shape}"
            )
        observed = observed.copy()
    if labeled_mask is not None:
        labeled = np.asarray(labeled_mask)
        if labeled.shape != (n,) or labeled.dtype != np.bool_:
            raise ValueError(
                f"labeled_mask must be a boolean vector with shape ({n},)"
            )
        observed &= labeled[:, np.newaxis]
    if np.any(observed.sum(axis=0) == 0):
        raise ValueError(
            "every component must have at least one observed label"
        )
    if not np.all(np.isfinite(y[observed])):
        raise ValueError("observed labels must be finite")

    if offsets is None:
        train_offsets = np.zeros_like(y)
    else:
        train_offsets = np.asarray(offsets, dtype=np.float64)
        if train_offsets.shape != y.shape or not np.all(
            np.isfinite(train_offsets)
        ):
            raise ValueError(f"offsets must be finite with shape {y.shape}")
    expected_grid_shape = (grid.shape[0], n_components)
    if grid_offsets is None:
        prediction_offsets = np.zeros(expected_grid_shape, dtype=np.float64)
    else:
        prediction_offsets = np.asarray(grid_offsets, dtype=np.float64)
        if prediction_offsets.shape != expected_grid_shape or not np.all(
            np.isfinite(prediction_offsets)
        ):
            raise ValueError(
                f"grid_offsets must be finite with shape {expected_grid_shape}"
            )

    coords_model, grid_model, lower, coordinate_scale = (
        scale_spatial_coordinates(coords, grid, mode=coordinate_scaling)
    )
    coordinate_span = np.ptp(np.vstack([coords, grid]), axis=0)
    domain = np.vstack([coords_model, grid_model])
    domain_lower = domain.min(axis=0) - 1e-3
    domain_upper = domain.max(axis=0) + 1e-3
    lkinfo = lk_setup(
        domain,
        nc=nc,
        nlevel=nlevel,
        nc_buffer=2,
        geometry=(
            "LKRectangle" if spatial_dimension == _ARRAY_NDIM else "LKBox"
        ),
        alpha=[1.0] * nlevel,
        a_wght=a_wght,
        normalize=False,
        overlap=2.5,
        range_domain=np.column_stack([domain_lower, domain_upper]).tolist(),
        m=2,
    )
    basis_train = compute_basis(coords_model, lkinfo, normalize=False)
    if not isinstance(basis_train, sp.csr_matrix):
        basis_train = basis_train.tocsr()

    fixed_design, fixed_design_grid = _prepare_joint_designs(
        fixed_effects,
        fixed_effects_grid,
        n_train=n,
        n_grid=grid.shape[0],
        n_components=n_components,
    )
    coefficient_prior = _build_joint_coefficient_prior(
        fixed_design,
        fixed_precision,
        fixed_prior_mean,
        n_components=n_components,
    )
    fit_kwargs: dict[str, Any] = {}
    if bayes_config.cluster_effect:
        basis_grid = compute_basis(grid_model, lkinfo, normalize=False)
        if not isinstance(basis_grid, sp.csr_matrix):
            basis_grid = basis_grid.tocsr()
        fit_kwargs = {
            "Phi_pred": basis_grid,
            "X_pred": fixed_design_grid,
            "offsets_pred": prediction_offsets,
        }
    fit = fit_joint(
        y,
        basis_train,
        lkinfo,
        X=fixed_design,
        labeled_mask=labeled_mask,
        observed_mask=observed,
        offsets=train_offsets,
        family=Bernoulli(),
        link=LogitLink(),
        coefficient_prior=coefficient_prior,
        inference="inla",
        prior=build_paige_prior(bayes_config),
        separate_ranges=bayes_config.separate_ranges,
        r_0=bayes_config.kleiber_r0,
        r_1=bayes_config.kleiber_r1,
        n_draw=bayes_config.n_draws,
        seed=bayes_config.seed,
        validate=bayes_config.validate_inla,
        cluster_effect=bayes_config.cluster_effect,
        **fit_kwargs,
    )
    fixed_draws_raw = fit.extra.get("fixed_draws")
    fixed_coef_draws = (
        None
        if fixed_draws_raw is None
        else np.asarray(fixed_draws_raw, dtype=np.float64)
    )
    coefficient_draws = np.asarray(fit.extra.get("c_draws"), dtype=np.float64)
    expected_coefficient_shape = (
        bayes_config.n_draws,
        basis_train.shape[1],
        n_components,
    )
    if coefficient_draws.shape != expected_coefficient_shape:
        raise RuntimeError(
            "canonical Bayesian posterior returned an unexpected "
            "latent-coefficient draw shape "
            f"{coefficient_draws.shape}; expected {expected_coefficient_shape}"
        )
    if not np.all(np.isfinite(coefficient_draws)):
        raise RuntimeError(
            "canonical Bayesian posterior returned nonfinite field draws"
        )
    if fixed_design_grid is None:
        if fixed_coef_draws is not None:
            raise RuntimeError(
                "canonical Bayesian posterior returned fixed-effect draws "
                "without a prediction design"
            )
    else:
        expected_fixed_shape = (
            bayes_config.n_draws,
            fixed_design_grid.shape[2],
            n_components,
        )
        if (
            fixed_coef_draws is None
            or fixed_coef_draws.shape != expected_fixed_shape
        ):
            fixed_shape = (
                None if fixed_coef_draws is None else fixed_coef_draws.shape
            )
            raise RuntimeError(
                "canonical Bayesian posterior returned an unexpected "
                f"fixed-effect draw shape {fixed_shape}; expected {expected_fixed_shape}"
            )
        if not np.all(np.isfinite(fixed_coef_draws)):
            raise RuntimeError(
                "canonical Bayesian posterior returned nonfinite fixed-effect draws"
            )
    projected_probability_draws = None
    if bayes_config.cluster_effect:
        projected_probability_draws = np.asarray(
            fit.extra.get("mean_draws"), dtype=np.float64
        )
        expected_prediction_shape = (
            bayes_config.n_draws,
            grid.shape[0],
            n_components,
        )
        if projected_probability_draws.shape != expected_prediction_shape:
            raise RuntimeError(
                "canonical Bayesian posterior returned an unexpected draw shape "
                f"{projected_probability_draws.shape}; expected "
                f"{expected_prediction_shape}"
            )
        if not np.all(np.isfinite(projected_probability_draws)) or np.any(
            (projected_probability_draws < 0.0)
            | (projected_probability_draws > 1.0)
        ):
            raise RuntimeError(
                "canonical Bayesian posterior returned invalid probability draws"
            )
        prediction_projection = "inla_global_projector"
    else:
        prediction_projection = "paired_coefficient_draws_chunked"

    diagnostics = {
        "backend": "gblk_bayesian",
        "estimator": "paige_inla",
        "posterior_scope": (
            "joint_fixed_effect_and_spatial_posterior"
            if fixed_effects is not None
            else "spatial_posterior"
        ),
        "fixed_effects_in_joint_likelihood": fixed_effects is not None,
        "fixed_effect_posterior_mean": (
            None
            if fixed_coef_draws is None
            else fixed_coef_draws.mean(axis=0).tolist()
        ),
        "n": int(n),
        "n_observed": int(observed.sum()),
        "n_components": int(n_components),
        "n_basis": int(basis_train.shape[1]),
        "n_grid": int(grid.shape[0]),
        "prediction_projection": prediction_projection,
        "prediction_chunk_size": (
            None
            if bayes_config.cluster_effect
            else _BAYESIAN_PREDICTION_CHUNK_SIZE
        ),
        "n_draws": int(bayes_config.n_draws),
        "ci_level": float(bayes_config.ci_level),
        "inference": fit.inference,
        "nlevel": int(lkinfo.nlevel),
        "spatial_dimension": int(coords.shape[1]),
        "coordinate_lower": lower.tolist(),
        "coordinate_span": coordinate_span.tolist(),
        "coordinate_scale": coordinate_scale.tolist(),
        "coordinate_transform": coordinate_scaling,
        "training_offsets_in_likelihood": True,
        "paige_prior": {
            "cor_scale_median": bayes_config.cor_scale_median,
            "spatial_sd_u": bayes_config.spatial_sd_u,
            "spatial_sd_tail_probability": (
                bayes_config.spatial_sd_tail_probability
            ),
            "dirichlet_concentration": bayes_config.dirichlet_concentration,
        },
        "kleiber_profile": {
            "r_0": bayes_config.kleiber_r0,
            "r_1": bayes_config.kleiber_r1,
        },
        "cluster_effect": bayes_config.cluster_effect,
        "inla_validation_outputs_requested": bayes_config.validate_inla,
    }
    return GBLKBayesianPosteriorState(
        component_names=names,
        coefficient_draws=coefficient_draws,
        fixed_coef_draws=fixed_coef_draws,
        grid_model=grid_model,
        grid_offsets=prediction_offsets,
        fixed_design_grid=fixed_design_grid,
        lkinfo=lkinfo,
        fit=fit,
        diagnostics=diagnostics,
        projected_probability_draws=projected_probability_draws,
    )


def project_gblk_bayesian_draw_block(
    state: GBLKBayesianPosteriorState,
    draw_start: int,
    draw_stop: int,
) -> GBLKBayesianDrawBlock:
    """Project one posterior draw block with prior/evidence/spatial logits."""
    from latticekrigx.basis.assembly import compute_basis  # noqa: PLC0415
    from scipy.special import expit  # noqa: PLC0415

    n_draws, n_basis, n_components = state.coefficient_draws.shape
    if not 0 <= draw_start < draw_stop <= n_draws:
        raise ValueError(
            f"draw range must satisfy 0 <= start < stop <= {n_draws}"
        )
    block_draws = draw_stop - draw_start
    n_grid = state.grid_model.shape[0]
    prior_logit = state.grid_offsets.copy()
    evidence_logit = np.zeros(
        (block_draws, n_grid, n_components), dtype=np.float64
    )
    spatial_logit = np.zeros_like(evidence_logit)

    if state.projected_probability_draws is not None:
        probability = state.projected_probability_draws[
            draw_start:draw_stop
        ].copy()
        eta = np.log(np.clip(probability, 1e-15, 1.0)) - np.log(
            np.clip(1.0 - probability, 1e-15, 1.0)
        )
        spatial_logit = eta - prior_logit[np.newaxis, :, :]
        if state.fixed_design_grid is not None:
            evidence_logit = np.einsum(
                "nqp,dpq->dnq",
                state.fixed_design_grid,
                state.fixed_coef_draws[draw_start:draw_stop],
                optimize=True,
            )
            spatial_logit -= evidence_logit
    else:
        coefficient_columns = (
            state.coefficient_draws[draw_start:draw_stop]
            .transpose(1, 0, 2)
            .reshape(n_basis, -1)
        )
        for cell_start in range(0, n_grid, _BAYESIAN_PREDICTION_CHUNK_SIZE):
            cell_stop = min(
                cell_start + _BAYESIAN_PREDICTION_CHUNK_SIZE, n_grid
            )
            basis_chunk = compute_basis(
                state.grid_model[cell_start:cell_stop],
                state.lkinfo,
                normalize=False,
            )
            latent = np.asarray(
                basis_chunk @ coefficient_columns, dtype=np.float64
            )
            spatial_logit[:, cell_start:cell_stop, :] = latent.reshape(
                cell_stop - cell_start, block_draws, n_components
            ).transpose(1, 0, 2)
            if state.fixed_design_grid is not None:
                evidence_logit[:, cell_start:cell_stop, :] = np.einsum(
                    "nqp,dpq->dnq",
                    state.fixed_design_grid[cell_start:cell_stop],
                    state.fixed_coef_draws[draw_start:draw_stop],
                    optimize=True,
                )
        probability = expit(
            prior_logit[np.newaxis, :, :] + evidence_logit + spatial_logit
        )
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise RuntimeError(
            "Bayesian block projection produced invalid probabilities"
        )
    return GBLKBayesianDrawBlock(
        draw_start=draw_start,
        draw_stop=draw_stop,
        component_probability=probability,
        prior_logit=prior_logit,
        evidence_logit=evidence_logit,
        spatial_logit=spatial_logit,
    )


def fit_gblk_bayesian_joint(  # noqa: PLR0913
    coords_spatial: NDArray[np.float64],
    labels: NDArray[np.float64],
    grid_spatial: NDArray[np.float64],
    *,
    component_names: tuple[str, ...] | list[str],
    bayes_config: GBLKBayesianConfig,
    labeled_mask: NDArray[np.bool_] | None = None,
    observed_mask: NDArray[np.bool_] | None = None,
    offsets: NDArray[np.float64] | None = None,
    grid_offsets: NDArray[np.float64] | None = None,
    fixed_effects: NDArray[np.float64] | None = None,
    fixed_effects_grid: NDArray[np.float64] | None = None,
    fixed_precision: NDArray[np.float64] | None = None,
    fixed_prior_mean: NDArray[np.float64] | None = None,
    nc: int = 6,
    nlevel: int = 1,
    a_wght: float = 4.5,
    coordinate_scaling: str = "axis_range",
) -> GBLKBayesianFitResult:
    """Fit and materialize all Bayesian draws through the existing API."""
    state = fit_gblk_bayesian_posterior_state(
        coords_spatial,
        labels,
        grid_spatial,
        component_names=component_names,
        bayes_config=bayes_config,
        labeled_mask=labeled_mask,
        observed_mask=observed_mask,
        offsets=offsets,
        grid_offsets=grid_offsets,
        fixed_effects=fixed_effects,
        fixed_effects_grid=fixed_effects_grid,
        fixed_precision=fixed_precision,
        fixed_prior_mean=fixed_prior_mean,
        nc=nc,
        nlevel=nlevel,
        a_wght=a_wght,
        coordinate_scaling=coordinate_scaling,
    )
    block = project_gblk_bayesian_draw_block(state, 0, bayes_config.n_draws)
    p_q_draws = block.component_probability
    p_joint_draws = np.prod(p_q_draws, axis=-1)
    tail = (1.0 - bayes_config.ci_level) / 2.0
    return GBLKBayesianFitResult(
        component_names=state.component_names,
        p_q_grid=p_q_draws.mean(axis=0),
        p_q_interval=np.quantile(p_q_draws, [tail, 1.0 - tail], axis=0),
        p_q_draws=p_q_draws,
        p_joint_grid=p_joint_draws.mean(axis=0),
        p_joint_interval=np.quantile(
            p_joint_draws, [tail, 1.0 - tail], axis=0
        ),
        p_joint_draws=p_joint_draws,
        fixed_coef_draws=state.fixed_coef_draws,
        fit=state.fit,
        diagnostics=state.diagnostics,
    )


def fit_gblk_joint(  # noqa: PLR0913, PLR0914
    coords_xy: NDArray[np.float64],
    labels: NDArray[np.float64],
    grid_xy: NDArray[np.float64],
    *,
    component_names: tuple[str, ...] | list[str],
    labeled_mask: NDArray[np.bool_] | None = None,
    observed_mask: NDArray[np.bool_] | None = None,
    offsets: NDArray[np.float64] | None = None,
    grid_offsets: NDArray[np.float64] | None = None,
    fixed_effects: NDArray[np.float64] | None = None,
    fixed_effects_grid: NDArray[np.float64] | None = None,
    fixed_precision: NDArray[np.float64] | None = None,
    fixed_prior_mean: NDArray[np.float64] | None = None,
    nc: int = 6,
    nlevel: int = 1,
    a_wght: float = 4.5,
    coordinate_scaling: str = "axis_range",
    max_outer_iter: int = 10,
    irls_max_iter: int = 15,
) -> GBLKFitResult:
    """Fit a joint multivariate Bernoulli-logit GBLK model and score a grid.

    Parameters
    ----------
    coords_xy
        ``(n, D)`` training coordinates, with ``D`` equal to 2 or 3.
    labels
        ``(n, Q)`` binary component labels.  Rows where ``labeled_mask`` is
        ``False`` are ignored by the likelihood.
    grid_xy
        ``(G, D)`` prediction-grid coordinates.
    component_names
        Ordered names for the ``Q`` components (columns of ``labels``).
    labeled_mask
        ``(n,)`` bool; ``True`` for labeled rows.  ``None`` means all labeled.
    offsets
        ``(n, Q)`` per-component logit-scale prior offsets at wells, or ``None``
        for zero offsets.
    grid_offsets
        ``(G, Q)`` per-component logit-scale prior offsets at the grid, or
        ``None`` for zero offsets.
    nc, a_wght
        LatticeKrig lattice knobs (see :func:`build_lkinfo_2d`).
    max_outer_iter, irls_max_iter
        Optimizer iteration budgets forwarded to ``fit_joint``.

    Returns
    -------
    GBLKFitResult
        Per-component and joint grid probabilities, estimated ``Omega``, and the
        raw fit.
    """
    import scipy.sparse as sp  # noqa: PLC0415

    from latticekrigx.basis.assembly import compute_basis  # noqa: PLC0415
    from latticekrigx.glk.joint import fit_joint  # noqa: PLC0415

    coords = np.asarray(coords_xy, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    grid = np.asarray(grid_xy, dtype=np.float64)
    names = tuple(component_names)

    if y.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"labels must be 2-D (n, Q); got {y.shape}")
    n, q = y.shape
    if coords.shape[0] != n:
        raise ValueError(
            f"coords_xy rows ({coords.shape[0]}) != labels rows ({n})"
        )
    if len(names) != q:
        raise ValueError(f"component_names length ({len(names)}) != Q ({q})")

    if coords.ndim != _ARRAY_NDIM or coords.shape[1] not in _SPATIAL_DIMS:
        raise ValueError(
            f"coords_xy must be (n, 2) or (n, 3); got {coords.shape}"
        )
    if grid.ndim != _ARRAY_NDIM or grid.shape[1] != coords.shape[1]:
        raise ValueError(
            "grid_xy must have the same coordinate dimension as coords_xy; "
            f"got {grid.shape} and {coords.shape}"
        )

    ndim = int(coords.shape[1])
    coords_model, grid_model, lower, coordinate_scale = (
        scale_spatial_coordinates(coords, grid, mode=coordinate_scaling)
    )
    coordinate_span = np.ptp(np.vstack([coords, grid]), axis=0)
    domain_coords = np.vstack([coords_model, grid_model])

    lkinfo = build_lkinfo(
        domain_coords,
        nc=nc,
        nlevel=nlevel,
        a_wght=a_wght,
    )

    basis = compute_basis(coords_model, lkinfo, normalize=False)
    if not isinstance(basis, sp.csr_matrix):
        basis = basis.tocsr()

    basis_grid = compute_basis(grid_model, lkinfo, normalize=False)
    if not isinstance(basis_grid, sp.csr_matrix):
        basis_grid = basis_grid.tocsr()

    fixed_design, fixed_design_grid = _prepare_joint_designs(
        fixed_effects,
        fixed_effects_grid,
        n_train=n,
        n_grid=grid.shape[0],
        n_components=q,
    )
    coefficient_prior = _build_joint_coefficient_prior(
        fixed_design,
        fixed_precision,
        fixed_prior_mean,
        n_components=q,
    )

    fit = fit_joint(
        y,
        basis,
        lkinfo,
        X=fixed_design,
        labeled_mask=labeled_mask,
        observed_mask=observed_mask,
        offsets=offsets,
        Phi_pred=basis_grid,
        X_pred=fixed_design_grid,
        offsets_pred=grid_offsets,
        coefficient_prior=coefficient_prior,
        max_outer_iter=max_outer_iter,
        irls_max_iter=irls_max_iter,
    )

    return GBLKFitResult(
        component_names=names,
        p_q_grid=np.asarray(fit.p_q, dtype=np.float64),
        p_joint_grid=np.asarray(fit.p_joint, dtype=np.float64),
        omega=np.asarray(fit.params.omega, dtype=np.float64),
        coef=np.asarray(fit.c_matrix, dtype=np.float64),
        fit=fit,
        diagnostics={
            "n": int(n),
            "n_labeled": (
                int(labeled_mask.sum()) if labeled_mask is not None else int(n)
            ),
            "n_components": int(q),
            "n_basis": int(basis.shape[1]),
            "n_grid": int(grid.shape[0]),
            "a_wght": float(a_wght),
            "nc": int(nc),
            "nlevel": int(nlevel),
            "spatial_dimension": ndim,
            "coordinate_lower": lower.tolist(),
            "coordinate_span": coordinate_span.tolist(),
            "coordinate_scale": coordinate_scale.tolist(),
            "coordinate_transform": coordinate_scaling,
            "fixed_effects_in_joint_likelihood": fixed_effects is not None,
            "fixed_effect_mode": (
                None
                if fit.fixed_coef is None
                else np.asarray(fit.fixed_coef, dtype=np.float64).tolist()
            ),
        },
    )
