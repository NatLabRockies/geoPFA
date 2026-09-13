"""Assembler that maps geoPFA data structures to ``fit_joint`` inputs.

Translates :class:`~geopfa.prob.pfa_grid.PFAGridAdapter`,
:class:`~geopfa.prob.labels.LoadedLabels`, and per-component
:class:`~geopfa.prob.alpha.AlphaCResult` objects into the array-level inputs consumed by
:func:`latticekrigx.glk.joint.fit_joint`:

* ``y (n, Q)``     — component responses at well locations.
* ``observed_mask`` — ``(n, Q)`` bool preserving componentwise label availability.
* ``labeled_mask`` — ``(n,)`` bool; ``True`` only for complete component rows.
* ``offsets (n, Q)`` — alpha logit offsets at well locations, snapped from
  the per-component grid offsets in every declared coordinate dimension.
* ``grid_offsets (G, Q)`` — alpha logit offsets at grid cells.
* ``evidence`` — per-component ``(n, L_q)`` evidence design matrix built
  from the processed layer values at well locations.
* ``well_coords (n, D)`` / ``grid_coords (G, D)`` — spatial coordinates.

Component ordering is fixed (sorted alphabetically) and reproducible.

Change-of-support helpers
-------------------------
:func:`make_point_support`, :func:`make_interval_support`, and
:func:`make_areal_support` build :class:`latticekrigx.glk.support.Support`
quadrature objects from geoPFA observation metadata. Pass a list of such
objects to :func:`build_observation_basis` to obtain the ``(n, M)`` sparse
averaged-basis-row matrix consumed by the GBLK fitter.

Regional pooling
----------------
:func:`pool_regional_coefficients` maps per-region coefficient estimates
``d_hat`` and variances ``d_var`` onto
:func:`latticekrigx.glk.hierarchy.eb_pool_coefficients` using play type
(from :mod:`geopfa.prob.play_types`) as the pooling group.  Regions with the
same play type share a common prior mean and between-region variance.

Notes
-----
Grid offsets for wells are obtained by dimension-aware nearest-neighbour
snapping onto the first component's ``pr_norm`` grid. Rectilinear component
grids are linearly resampled to that canonical support; incomplete,
duplicated, or non-covering grids fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import scipy.sparse
from latticekrigx.glk.hierarchy import EBPoolResult, eb_pool_coefficients
from latticekrigx.glk.support import Support, averaged_basis_row
from latticekrigx.model.config import LKInfo
from numpy.typing import NDArray
from scipy.special import expit

from geopfa.exceptions import GEOPFAValueError
from geopfa import transformation
from geopfa.prob.alpha import AlphaCResult
from geopfa.prob.config import EvidenceConfig, LabelsConfig
from geopfa.prob.fitting import _is_sparse_binary
from geopfa.prob.labels import LoadedLabels, _coerce_component_labels
from geopfa.prob.pfa_grid import PFAGridAdapter
from geopfa.prob.spatial_alignment import (
    align_to_grid_crs,
    extract_coordinates,
    grid_values_on_reference,
    sample_layer_at_points,
    snap_to_grid_indices,
)


@dataclass(frozen=True)
class AssembledInputs:
    """Assembled ``fit_joint`` inputs built from geoPFA data structures.

    Attributes
    ----------
    component_names
        Ordered component names (alphabetically sorted), length ``Q``.
    y
        Binary labels at well locations, shape ``(n, Q)``. Entries are
        ``0.0`` where no finite label was available; check
        ``observed_mask`` to distinguish from a genuine negative label.
    observed_mask
        Component-level label availability, shape ``(n, Q)``.
    labeled_mask
        Complete-case label availability, shape ``(n,)``. ``True`` only
        when every fitted component has a finite label for that well.
    well_offsets
        Alpha logit offsets at well locations, shape ``(n, Q)``. Each
        well is assigned the offset of its nearest grid cell.
    grid_offsets
        Alpha logit offsets at grid cells, shape ``(G, Q)``.
    well_coords
        Well spatial coordinates, shape ``(n, 2)`` for 2-D or
        ``(n, 3)`` for 3-D.
    grid_coords
        Grid cell spatial coordinates, shape ``(G, 2)`` or ``(G, 3)``.
    evidence
        Per-component evidence design matrices keyed by component name.
        Each value has shape ``(n, L_q)`` where ``L_q`` is the number of
        non-alpha evidence layers for component ``q``.
    grid_evidence
        Matching per-component evidence matrices at prediction cells, each
        with shape ``(G, L_q)``.
    layer_names
        Ordered layer names for each component's evidence matrix.
    prior_probability_grid, prior_probability_well
        Configured event probability before the outcome update, retained on
        the grid and at observation locations for predictive stacking.
    prior_response_mean_grid, prior_response_mean_well,
    prior_response_sd_grid, prior_response_sd_well
        Optional continuous prior predictive moments, retained when the alpha
        construction supplies both a latent mean and standard deviation.
    """

    component_names: tuple[str, ...]
    y: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    labeled_mask: NDArray[np.bool_]
    well_offsets: NDArray[np.float64]
    grid_offsets: NDArray[np.float64]
    well_coords: NDArray[np.float64]
    grid_coords: NDArray[np.float64]
    evidence: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    grid_evidence: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    layer_names: dict[str, list[str]] = field(default_factory=dict)
    prior_probability_grid: NDArray[np.float64] | None = None
    prior_probability_well: NDArray[np.float64] | None = None
    prior_response_mean_grid: NDArray[np.float64] | None = None
    prior_response_mean_well: NDArray[np.float64] | None = None
    prior_response_sd_grid: NDArray[np.float64] | None = None
    prior_response_sd_well: NDArray[np.float64] | None = None

    @property
    def n(self) -> int:
        """int: Number of observation (well) rows."""
        return int(self.y.shape[0])

    @property
    def n_components(self) -> int:
        """int: Number of components."""
        return int(self.y.shape[1])

    @property
    def n_grid(self) -> int:
        """int: Number of grid cells."""
        return int(self.grid_coords.shape[0])


def _transformed_layer_model(
    adapter: PFAGridAdapter,
    component: str,
    layer_name: str,
) -> tuple[gpd.GeoDataFrame, str]:
    """Return a copy carrying the configured PFA evidence transformation."""
    layer_data = adapter.layer_data(component, layer_name)
    value_col = layer_data.get("model_data_col", "value_interpolated")
    model = layer_data["model"]
    if value_col not in model.columns:
        raise GEOPFAValueError(
            f"component {component!r} evidence layer {layer_name!r} "
            f"is missing declared value column {value_col!r}"
        )
    try:
        values = model[value_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise GEOPFAValueError(
            f"component {component!r} evidence layer {layer_name!r} "
            "must contain numeric values"
        ) from exc
    method = str(layer_data.get("transformation_method", "none"))
    try:
        transformed = np.asarray(
            transformation.transform(values, method), dtype=np.float64
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise GEOPFAValueError(
            f"component {component!r} evidence layer {layer_name!r} "
            f"failed transformation {method!r}"
        ) from exc
    if transformed.shape != values.shape:
        raise GEOPFAValueError(
            f"component {component!r} evidence layer {layer_name!r} "
            "transformation changed the evidence shape"
        )
    transformed_model = model.copy()
    transformed_model[value_col] = transformed
    return transformed_model, value_col


def _build_labels_array(
    wells_gdf: gpd.GeoDataFrame,
    component_names: tuple[str, ...],
    labels_config: LabelsConfig,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Build labels plus componentwise and complete-case observation masks."""
    n = len(wells_gdf)
    Q = len(component_names)
    y_raw = np.full((n, Q), np.nan, dtype=np.float64)
    for q_idx, comp in enumerate(component_names):
        col = labels_config.label_columns.get(comp)
        if col is None:
            raise GEOPFAValueError(
                f"component {comp!r} has no declared label column"
            )
        if col not in wells_gdf.columns:
            raise GEOPFAValueError(
                f"component {comp!r} label column {col!r} is missing"
            )
        y_raw[:, q_idx] = _coerce_component_labels(
            wells_gdf[col],
            label_column=col,
            family=labels_config.observation_model_for(comp).family,
        ).to_numpy(dtype=float)
    observed_mask = np.isfinite(y_raw)
    labeled_mask = np.all(observed_mask, axis=1)
    y = np.where(np.isfinite(y_raw), y_raw, 0.0)
    return y, observed_mask, labeled_mask


def select_component_evidence_layers(
    adapter: PFAGridAdapter,
    component: str,
    alpha_results: dict[str, AlphaCResult],
    evidence_config: EvidenceConfig,
) -> list[str]:
    """Return the canonical active evidence-layer order for one component."""
    excluded = set(alpha_results[component].excluded_layer_names) | set(
        evidence_config.exclude_layers
    )
    included = (
        None
        if evidence_config.include_layers is None
        else set(evidence_config.include_layers)
    )
    active: list[str] = []
    for layer_name in adapter.layers(component):
        if included is not None and layer_name not in included:
            continue
        if layer_name in excluded:
            continue
        layer_data = adapter.layer_data(component, layer_name)
        value_col = layer_data.get("model_data_col", "value_interpolated")
        if value_col in evidence_config.coordinate_blacklist:
            continue
        model, value_col = _transformed_layer_model(
            adapter, component, layer_name
        )
        values = model[value_col].to_numpy(dtype=float)
        if _is_sparse_binary(
            values,
            threshold=evidence_config.sparse_binary_threshold,
        ):
            continue
        active.append(layer_name)
    return active


def _build_evidence(
    wells_gdf: gpd.GeoDataFrame,
    adapter: PFAGridAdapter,
    component_names: tuple[str, ...],
    alpha_results: dict[str, AlphaCResult],
    evidence_config: EvidenceConfig,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, list[str]]]:
    """Build per-component evidence design matrices and layer name lists."""
    n = len(wells_gdf)
    evidence: dict[str, NDArray[np.float64]] = {}
    layer_names_map: dict[str, list[str]] = {}
    for comp in component_names:
        active = select_component_evidence_layers(
            adapter, comp, alpha_results, evidence_config
        )
        layer_names_map[comp] = active
        if not active:
            evidence[comp] = np.zeros((n, 0), dtype=np.float64)
            continue
        cols = []
        for lname in active:
            model, value_col = _transformed_layer_model(adapter, comp, lname)
            vals = sample_layer_at_points(
                wells_gdf,
                model,
                value_col,
            )
            cols.append(vals)
        evidence[comp] = np.column_stack(cols).astype(np.float64)
    return evidence, layer_names_map


def _build_grid_evidence(
    adapter: PFAGridAdapter,
    component_names: tuple[str, ...],
    layer_names_map: dict[str, list[str]],
    reference_grid: gpd.GeoDataFrame,
) -> dict[str, NDArray[np.float64]]:
    """Build prediction-grid evidence in the same column order as well evidence."""
    grid_evidence: dict[str, NDArray[np.float64]] = {}
    for comp in component_names:
        columns: list[NDArray[np.float64]] = []
        for lname in layer_names_map[comp]:
            layer_grid, value_col = _transformed_layer_model(
                adapter, comp, lname
            )
            columns.append(
                grid_values_on_reference(
                    reference_grid,
                    layer_grid,
                    value_col,
                    context=f"component {comp!r} layer {lname!r}",
                )
            )
        if columns:
            grid_evidence[comp] = np.column_stack(columns).astype(np.float64)
        else:
            grid_evidence[comp] = np.zeros(
                (len(reference_grid), 0), dtype=np.float64
            )
    return grid_evidence


def _component_values_on_reference(
    reference_grid: gpd.GeoDataFrame,
    component_grid: gpd.GeoDataFrame,
    values: NDArray[np.float64],
    *,
    context: str,
) -> NDArray[np.float64]:
    """Resample one component-grid vector onto the canonical support."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(component_grid),):
        raise GEOPFAValueError(
            f"{context} has {vector.size} values; expected {len(component_grid)}"
        )
    value_col = "__geopfa_component_value__"
    field = component_grid[[component_grid.geometry.name]].copy()
    field[value_col] = vector
    return grid_values_on_reference(
        reference_grid,
        field,
        value_col,
        context=context,
    )


def build_component_grid_evidence(
    adapter: PFAGridAdapter,
    component: str,
    alpha_result: AlphaCResult,
    evidence_config: EvidenceConfig,
) -> tuple[NDArray[np.float64], list[str]]:
    """Build canonical grid evidence without requiring outcome locations."""
    layer_names = select_component_evidence_layers(
        adapter,
        component,
        {component: alpha_result},
        evidence_config,
    )
    reference = adapter.pr_norm(component)
    evidence = _build_grid_evidence(
        adapter,
        (component,),
        {component: layer_names},
        reference,
    )
    return evidence[component], layer_names


def _validated_component_names(
    adapter: PFAGridAdapter,
    loaded_labels: LoadedLabels,
    alpha_results: dict[str, AlphaCResult],
) -> tuple[str, ...]:
    """Validate fitted component declarations and return adapter ordering."""
    adapter_components = adapter.components()
    label_components = set(loaded_labels.config.label_columns)
    alpha_components = set(alpha_results)
    missing_from_adapter = (label_components | alpha_components) - set(
        adapter_components
    )
    if missing_from_adapter:
        raise ValueError(
            "configured components are absent from the PFA adapter: "
            + ", ".join(sorted(missing_from_adapter))
        )
    if label_components != alpha_components:
        details = []
        if missing_alpha := label_components - alpha_components:
            details.append(
                "missing alpha: " + ", ".join(sorted(missing_alpha))
            )
        if missing_labels := alpha_components - label_components:
            details.append(
                "missing label contract: " + ", ".join(sorted(missing_labels))
            )
        raise ValueError(
            "GBLK assembly requires identical fitted alpha and label component "
            "sets; " + "; ".join(details)
        )
    return tuple(
        component
        for component in adapter_components
        if component in alpha_components
    )


def assemble_gblk_inputs(  # noqa: PLR0914
    adapter: PFAGridAdapter,
    loaded_labels: LoadedLabels,
    alpha_results: dict[str, AlphaCResult],
    *,
    evidence_config: EvidenceConfig,
    prior_probability_results: dict[str, AlphaCResult] | None = None,
) -> AssembledInputs:
    """Build ``fit_joint`` inputs from geoPFA data structures.

    Parameters
    ----------
    adapter
        Bound :class:`~geopfa.prob.pfa_grid.PFAGridAdapter` wrapping the PFA
        dict.
    loaded_labels
        Labelled wells returned by
        :func:`~geopfa.prob.labels.load_labels`.
    alpha_results
        Per-component alpha build results from
        :func:`~geopfa.prob.alpha.build_alpha_c`, keyed by component
        name. Its keys must exactly match the fitted label-component keys.
    evidence_config
        Canonical evidence allowlist, denylist, coordinate screening, and
        sparse-layer screening contract.
    prior_probability_results
        Optional alpha results that retain the configured event-probability
        logits when ``alpha_results`` has been transformed to another
        likelihood scale, such as a Gaussian response mean.

    Returns
    -------
    AssembledInputs
        All arrays needed to call ``fit_joint``.

    Raises
    ------
    ValueError
        If fitted alpha, label, and PFA component declarations disagree.
    """
    component_names = _validated_component_names(
        adapter, loaded_labels, alpha_results
    )
    probability_results = (
        alpha_results
        if prior_probability_results is None
        else prior_probability_results
    )
    if set(probability_results) != set(component_names):
        raise ValueError(
            "prior_probability_results must match the fitted component set"
        )
    if not component_names:
        raise ValueError(
            "no components found in both the adapter and alpha_results; "
            f"adapter has {adapter.components()}, alpha_results has "
            f"{sorted(alpha_results)}"
        )

    grid_gdf = adapter.pr_norm(component_names[0])
    grid_coords = extract_coordinates(grid_gdf)
    aligned_offsets = [
        _component_values_on_reference(
            grid_gdf,
            adapter.pr_norm(component),
            alpha_results[component].grid_offset,
            context=f"component {component!r} alpha grid",
        )
        for component in component_names
    ]
    grid_offsets = np.column_stack(aligned_offsets).astype(np.float64)
    prior_probability_grid = expit(
        np.column_stack(
            [
                _component_values_on_reference(
                    grid_gdf,
                    adapter.pr_norm(component),
                    probability_results[component].grid_offset,
                    context=(
                        f"component {component!r} prior-probability grid"
                    ),
                )
                for component in component_names
            ]
        ).astype(np.float64)
    )
    has_response_moments = [
        probability_results[component].latent_mean is not None
        and probability_results[component].latent_sd is not None
        for component in component_names
    ]
    prior_response_mean_grid = None
    prior_response_sd_grid = None
    if any(has_response_moments):
        prior_response_mean_grid = np.column_stack(
            [
                _component_values_on_reference(
                    grid_gdf,
                    adapter.pr_norm(component),
                    probability_results[component].latent_mean,
                    context=f"component {component!r} prior response mean",
                )
                if available
                else np.full(len(grid_gdf), np.nan)
                for component, available in zip(
                    component_names, has_response_moments, strict=True
                )
            ]
        ).astype(np.float64)
        prior_response_sd_grid = np.column_stack(
            [
                _component_values_on_reference(
                    grid_gdf,
                    adapter.pr_norm(component),
                    probability_results[component].latent_sd,
                    context=f"component {component!r} prior response SD",
                )
                if available
                else np.full(len(grid_gdf), np.nan)
                for component, available in zip(
                    component_names, has_response_moments, strict=True
                )
            ]
        ).astype(np.float64)

    wells_gdf = align_to_grid_crs(loaded_labels.gdf, grid_gdf)
    well_coords = extract_coordinates(wells_gdf)
    well_grid_indices = snap_to_grid_indices(wells_gdf, grid_gdf)
    well_offsets = grid_offsets[well_grid_indices]
    prior_probability_well = prior_probability_grid[well_grid_indices]
    prior_response_mean_well = (
        None
        if prior_response_mean_grid is None
        else prior_response_mean_grid[well_grid_indices]
    )
    prior_response_sd_well = (
        None
        if prior_response_sd_grid is None
        else prior_response_sd_grid[well_grid_indices]
    )

    y, observed_mask, labeled_mask = _build_labels_array(
        wells_gdf,
        component_names,
        loaded_labels.config,
    )

    evidence, layer_names_map = _build_evidence(
        wells_gdf,
        adapter,
        component_names,
        alpha_results,
        evidence_config,
    )
    grid_evidence = _build_grid_evidence(
        adapter, component_names, layer_names_map, grid_gdf
    )

    return AssembledInputs(
        component_names=component_names,
        y=y,
        observed_mask=observed_mask,
        labeled_mask=labeled_mask,
        well_offsets=well_offsets,
        grid_offsets=grid_offsets,
        well_coords=well_coords,
        grid_coords=grid_coords,
        evidence=evidence,
        grid_evidence=grid_evidence,
        layer_names=layer_names_map,
        prior_probability_grid=prior_probability_grid,
        prior_probability_well=prior_probability_well,
        prior_response_mean_grid=prior_response_mean_grid,
        prior_response_mean_well=prior_response_mean_well,
        prior_response_sd_grid=prior_response_sd_grid,
        prior_response_sd_well=prior_response_sd_well,
    )


def make_point_support(x: NDArray[np.float64]) -> Support:
    """Build a point support for a single well location.

    Parameters
    ----------
    x : np.ndarray
        Spatial location, shape ``(ndim,)`` or ``(1, ndim)``.

    Returns
    -------
    latticekrigx.glk.support.Support
        Point support with unit weight at ``x``.
    """
    return Support.from_point(np.asarray(x, dtype=float))


def make_interval_support(
    x_start: NDArray[np.float64],
    x_end: NDArray[np.float64],
    n_quad: int = 10,
) -> Support:
    """Build a support for a depth interval or line segment.

    Quadrature points are placed at ``n_quad`` evenly-spaced locations
    along the straight segment from ``x_start`` to ``x_end``, each with
    equal weight.

    Parameters
    ----------
    x_start : np.ndarray
        Start location, shape ``(ndim,)``.
    x_end : np.ndarray
        End location, shape ``(ndim,)``.
    n_quad : int, default=10
        Number of quadrature points along the segment.

    Returns
    -------
    latticekrigx.glk.support.Support
        Interval support with ``n_quad`` equal-weight quadrature points.

    Raises
    ------
    geopfa.exceptions.GEOPFAValueError
        If ``n_quad`` is less than 1.
    """
    if n_quad < 1:
        raise GEOPFAValueError(f"n_quad must be >= 1, got {n_quad}")
    ts = np.linspace(0.0, 1.0, n_quad)
    a = np.asarray(x_start, dtype=float).ravel()
    b = np.asarray(x_end, dtype=float).ravel()
    pts = a + ts[:, np.newaxis] * (b - a)
    weights = np.ones(n_quad)
    return Support(x=pts, w=weights, kind="interval")


def make_areal_support(
    pts: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> Support:
    """Build a support for an areal (polygon) observation.

    Parameters
    ----------
    pts : np.ndarray
        Quadrature locations, shape ``(n_pts, ndim)``.
    weights : np.ndarray, optional
        Non-negative quadrature weights, shape ``(n_pts,)``. Equal
        weights are used when ``None``.

    Returns
    -------
    latticekrigx.glk.support.Support
        Areal support with the given quadrature points and weights.
    """
    pts_arr = np.asarray(pts, dtype=float)
    if pts_arr.ndim == 1:
        pts_arr = pts_arr[np.newaxis, :]
    n_pts = pts_arr.shape[0]
    w = np.ones(n_pts) if weights is None else np.asarray(weights, dtype=float)
    return Support(x=pts_arr, w=w, kind="areal")


def pool_regional_coefficients(
    d_hat: NDArray[np.float64],
    d_var: NDArray[np.float64],
    play_type_per_region: list[str] | NDArray,
) -> EBPoolResult:
    """Partial-pool per-region evidence coefficients grouped by play type.

    Wraps :func:`latticekrigx.glk.hierarchy.eb_pool_coefficients` with
    the geoPFA convention that **play type is the pooling group**.  Each
    region belongs to exactly one play type; regions within the same play
    type share a common prior mean and between-region variance.

    The shrinkage factor for region ``r`` and coefficient ``j`` is::

        B[r, j] = sigma2[r, j] / (tau2[t(r), j] + sigma2[r, j])

    *  **Data-rich region** (``sigma2 ≈ 0``): ``B ≈ 0``; posterior ≈
       unpooled estimate ``d_hat``.
    *  **Data-poor region** (``sigma2 ≫ tau2``): ``B ≈ 1``; posterior
       shrinks toward the play-type mean ``mu_{t(r)}``.
    *  **Single region** (``R = 1``): ``tau2`` is estimated as 0, so
       ``B = 1`` and ``d_post = mu = d_hat``.

    Parameters
    ----------
    d_hat : np.ndarray
        Per-region raw (OLS/MLE) coefficient estimates, shape ``(R,)``
        or ``(R, p)``.
    d_var : np.ndarray
        Sampling variances for each coefficient, same shape as ``d_hat``.
        All values must be non-negative.
    play_type_per_region : list[str] or np.ndarray
        Play-type label for each of the ``R`` regions, shape ``(R,)``.
        Regions with the same play type are pooled together.

    Returns
    -------
    latticekrigx.glk.hierarchy.EBPoolResult
        Shrinkage-pooled coefficients and diagnostics.  ``d_pooled`` has
        shape ``(R, p)`` (or ``(R, 1)`` for scalar input).

    Raises
    ------
    geopfa.exceptions.GEOPFAValueError
        If ``play_type_per_region`` has a different length from ``d_hat``,
        or if any ``d_var`` value is negative.
    """
    d_hat_arr = np.asarray(d_hat, dtype=np.float64)
    d_var_arr = np.asarray(d_var, dtype=np.float64)
    region_to_group = np.asarray(play_type_per_region)

    R = d_hat_arr.shape[0]
    if region_to_group.shape[0] != R:
        raise GEOPFAValueError(
            f"play_type_per_region length {region_to_group.shape[0]} does "
            f"not match d_hat row count {R}"
        )
    if np.any(d_var_arr < 0):
        raise GEOPFAValueError(
            "d_var must be non-negative; found negative values"
        )

    return eb_pool_coefficients(d_hat_arr, d_var_arr, region_to_group)


def build_observation_basis(
    supports: list[Support],
    lkinfo: LKInfo,
) -> scipy.sparse.csr_matrix:
    """Build the ``(n, M)`` averaged-basis-row matrix for ``n`` observations.

    Each observation is represented by a :class:`~latticekrigx.glk.support.Support`
    quadrature rule. A point observation produces a row equal to the plain
    basis evaluation; an interval or areal observation produces a row equal
    to the weighted mean of the member basis rows.

    Parameters
    ----------
    supports : list of latticekrigx.glk.support.Support
        One support per observation, length ``n``.
    lkinfo : latticekrigx.model.config.LKInfo
        LatticeKrig configuration defining the basis.

    Returns
    -------
    scipy.sparse.csr_matrix
        Averaged basis matrix, shape ``(n, M)``.

    Raises
    ------
    geopfa.exceptions.GEOPFAValueError
        If ``supports`` is empty.
    """
    if not supports:
        raise GEOPFAValueError("supports list must not be empty")
    rows = [averaged_basis_row(s, lkinfo) for s in supports]
    return scipy.sparse.vstack(rows, format="csr")


__all__ = [
    "AssembledInputs",
    "assemble_gblk_inputs",
    "build_observation_basis",
    "make_areal_support",
    "make_interval_support",
    "make_point_support",
    "pool_regional_coefficients",
]
