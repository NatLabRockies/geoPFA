"""Sequential inference backend.

Wraps the existing :func:`geopfa.prob.fitting.fit_component_probability`
(scipy BFGS logistic regression followed by a standalone GP spatial field)
into the :class:`~geopfa.prob.inference.Fitter` protocol.

This is a deprecated, explicitly selected non-Bayesian backend. GBLK is the
default production backend.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np

from geopfa.prob.alpha import AlphaCResult
from geopfa.prob.config import AlphaModeConfig
from geopfa.prob.config import EvidenceConfig, SpatialFieldConfig
from geopfa.prob.fit_dispatch import build_fit_kwargs
from geopfa.prob.fitting import fit_component_probability
from . import ComponentFitResult


class SequentialFitter:
    """Sequential backend: L2-regularised logistic regression + standalone GP.

    Uses :func:`geopfa.prob.fitting.fit_component_probability` under the
    hood. This backend is retained for explicitly supported diagnostics such
    as nnPU; it is not a Bayesian implementation.
    """

    def fit(  # noqa: PLR0913, PLR0917, PLR6301
        self,
        component_data: dict,
        alpha_result: AlphaCResult,
        evidence_config: EvidenceConfig,
        labels: gpd.GeoDataFrame,
        label_column: str,
        spatial_config: SpatialFieldConfig,
        *,
        per_feature_weights: dict[str, float] | None = None,
        prior_means: dict[str, float] | None = None,
    ) -> ComponentFitResult:
        """Fit one component with the sequential approximation.

        Parameters
        ----------
        component_data, alpha_result, evidence_config, labels,
        label_column, spatial_config
            Standard per-component inputs (see protocol).
        per_feature_weights, prior_means
            Optional explicit per-feature Gaussian-prior parameters.
        """
        # Build a minimal AlphaModeConfig just to call build_fit_kwargs.
        alpha_mode_cfg = AlphaModeConfig(
            mode="scalar",
            scalar_fallback_pr0=float(
                np.clip(
                    1.0 / (1.0 + np.exp(-alpha_result.scalar_fallback)),
                    1e-4,
                    1.0 - 1e-4,
                )
            ),
        )
        kwargs = build_fit_kwargs(
            alpha_config=alpha_mode_cfg,
            evidence_config=evidence_config,
            spatial_field_config=spatial_config,
        )
        # Inject the pre-built alpha offset.
        kwargs["alpha_offset"] = alpha_result.grid_offset
        kwargs["prior_layer_name"] = None
        # Exclude alpha-excluded layers.
        extra_excluded = set(alpha_result.excluded_layer_names) | set(
            kwargs.get("excluded_layer_names", ())
        )
        kwargs["excluded_layer_names"] = tuple(sorted(extra_excluded))
        if per_feature_weights is not None:
            kwargs["per_feature_weights"] = per_feature_weights
        if prior_means is not None:
            kwargs["prior_means"] = prior_means

        comp_prob = fit_component_probability(
            component_data,
            labeled_wells=labels,
            label_column=label_column,
            **kwargs,
        )

        is_prior_only = comp_prob.model is None
        beta_posterior: dict[str, Any] = {}
        n_train = 0
        if not is_prior_only:
            coefficients = np.asarray(comp_prob.model.x, dtype=float)
            if coefficients.ndim != 1 or not np.all(np.isfinite(coefficients)):
                raise RuntimeError(
                    "sequential fit returned invalid evidence coefficients"
                )
            beta_posterior = {"mean": coefficients}
            reported_n_train = comp_prob.diagnostics.get("n_train")
            if (
                isinstance(reported_n_train, bool)
                or not isinstance(reported_n_train, int | np.integer)
                or reported_n_train < 1
            ):
                raise RuntimeError(
                    "sequential fit did not report a valid training count"
                )
            n_train = int(reported_n_train)

        spatial_mean = comp_prob.spatial_field
        spatial_std: np.ndarray | None = None
        gdf = comp_prob.probability
        if "spatial_u_std" in gdf.columns:
            spatial_std = gdf["spatial_u_std"].to_numpy(dtype=float)

        return ComponentFitResult(
            probability=gdf,
            beta_posterior=beta_posterior,
            feature_names=comp_prob.feature_names,
            alpha_used=alpha_result.grid_offset,
            spatial_mean=spatial_mean,
            spatial_std=spatial_std,
            n_train=n_train,
            diagnostics={**comp_prob.diagnostics, "backend": "sequential"},
            prior_predictive_only=is_prior_only,
        )


__all__ = ["SequentialFitter"]
