"""Adapters that translate probabilistic config blocks into fitter inputs.

The runner consumes a
:class:`~geopfa.prob.config.ProbabilisticConfig` and dispatches to the
appropriate fitter. The fitters themselves keep their existing typed
kwarg-based API (so direct programmatic use stays simple). This module
bridges the two by translating config sub-blocks into the kwargs the
:func:`~geopfa.prob.fitting.fit_component_probability` function accepts,
without changing its signature.

Each helper here is intentionally thin and side-effect-free so it can be
unit-tested without spinning up a full fit.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from .alpha import build_alpha_c
from .config import AlphaModeConfig, EvidenceConfig, SpatialFieldConfig
from .fitting import ComponentProbability, fit_component_probability
from .pfa_grid import component_grid


def build_fit_kwargs(  # noqa: PLR0913
    *,
    alpha_config: AlphaModeConfig,
    evidence_config: EvidenceConfig,
    spatial_field_config: SpatialFieldConfig,
    pu_mode: str = "off",
    pu_class_prior: float | None = None,
    min_wells: int = 4,
) -> dict[str, Any]:
    """Translate config sub-blocks into kwargs for ``fit_component_probability``.

    Parameters
    ----------
    alpha_config
        Component's ``AlphaModeConfig`` (per-component prior choice).
    evidence_config
        Top-level ``EvidenceConfig`` (regression knobs).
    spatial_field_config
        Top-level ``SpatialFieldConfig`` (``u_c`` configuration).
    pu_mode
        Positive-Unlabeled correction mode forwarded from
        ``LabelsConfig.pu_mode``. One of ``"off"``,
        ``"naive_pseudo_absence"``, or ``"nnpu"``.

    Returns
    -------
    dict
        Kwargs that can be ``**``-spread into
        ``fit_component_probability(component_data, ...)``.
    """
    excluded = set(evidence_config.exclude_layers)
    prior_layer_name: str | None = None
    if alpha_config.mode == "layer_logit":
        prior_layer_name = alpha_config.layer
    elif alpha_config.mode == "multi_layer":
        excluded.update(alpha_config.layers)

    # Explicit proper Gaussian coefficient priors from config. The newer
    # ``prior_precisions`` spelling takes precedence when both mappings name
    # the same feature.
    explicit_weights = dict(evidence_config.regularization.per_feature_weights)
    explicit_weights.update(evidence_config.regularization.prior_precisions)
    explicit_means = dict(evidence_config.regularization.prior_means)

    return {
        "prior_probability": alpha_config.scalar_fallback_pr0,
        "prior_layer_name": prior_layer_name,
        "prior_p_min": alpha_config.p_min,
        "prior_p_max": alpha_config.p_max,
        "include_spatial": spatial_field_config.enabled,
        "included_layer_names": evidence_config.include_layers,
        "excluded_layer_names": tuple(sorted(excluded)),
        "spatial_backend": spatial_field_config.backend,
        "sparse_binary_threshold": evidence_config.sparse_binary_threshold,
        "coordinate_blacklist": tuple(evidence_config.coordinate_blacklist),
        "force_prior_predictive": alpha_config.force_prior_predictive,
        "per_feature_weights": explicit_weights or None,
        "prior_means": explicit_means or None,
        "pu_mode": pu_mode,
        "pu_class_prior": pu_class_prior,
        "min_wells": min_wells,
    }


def fit_component_from_config(  # noqa: PLR0913
    component_data: dict,
    *,
    alpha_config: AlphaModeConfig,
    evidence_config: EvidenceConfig,
    spatial_field_config: SpatialFieldConfig,
    labeled_wells: gpd.GeoDataFrame | None = None,
    label_column: str | None = None,
) -> ComponentProbability:
    """Run :func:`~geopfa.prob.fitting.fit_component_probability` from config.

    Parameters
    ----------
    component_data
        The per-component dict from the PFA tree (with ``layers``,
        ``pr_norm``, etc.).
    alpha_config, evidence_config, spatial_field_config
        Config sub-blocks; see :func:`build_fit_kwargs`.
    labeled_wells
        Labelled-well GeoDataFrame. It is required unless the alpha config
        explicitly requests a prior-predictive result.
    label_column
        Column on ``labeled_wells`` holding 0/1 labels for this component.

    Returns
    -------
    ComponentProbability
        The fitted-surface object from
        :func:`geopfa.prob.fitting.fit_component_probability`.
    """
    kwargs = build_fit_kwargs(
        alpha_config=alpha_config,
        evidence_config=evidence_config,
        spatial_field_config=spatial_field_config,
    )
    grid_gdf = component_grid(component_data)
    if grid_gdf is None:
        raise ValueError(
            "component has no usable grid: pr_norm is absent and no layer "
            "model GeoDataFrame is available"
        )
    alpha_result = build_alpha_c(
        component_data,
        alpha_config,
        grid_gdf=grid_gdf,
    )
    excluded = set(kwargs["excluded_layer_names"])
    excluded.update(alpha_result.excluded_layer_names)
    kwargs["excluded_layer_names"] = tuple(sorted(excluded))
    kwargs["alpha_offset"] = alpha_result.grid_offset
    kwargs["prior_layer_name"] = None
    return fit_component_probability(
        component_data,
        labeled_wells=labeled_wells,
        label_column=label_column,
        **kwargs,
    )


__all__ = ["build_fit_kwargs", "fit_component_from_config"]
