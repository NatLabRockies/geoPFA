"""Componentwise inference protocol for the probabilistic method.

The sequential component fitter implements :class:`Fitter` and returns a
:class:`ComponentFitResult`. Joint Bayesian spatial inference is exposed only
through the GBLK runner and therefore does not implement this componentwise
protocol.

The protocol is intentionally thin: the only contract is that ``fit``
accepts the standard per-component inputs and returns a
``ComponentFitResult``.  Backend-specific knobs live in the config and
in the concrete Fitter constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import geopandas as gpd
import numpy as np


@dataclass(frozen=True)
class ComponentFitResult:
    """Unified output of a per-component fit across backends.

    Attributes
    ----------
    probability
        Grid GeoDataFrame with a ``probability`` column (and optionally
        ``spatial_u``, ``spatial_u_std``, ``prior_probability_spatial``).
    beta_posterior
        Fitted regression coefficients: ``{"mean": np.ndarray}`` for the
        sequential backend.
    feature_names
        Names of the evidence features used in the regression matrix.
    alpha_used
        The per-cell prior offset array ``alpha_c(s, z)`` that was applied.
    spatial_mean
        Per-cell posterior mean of ``u_c(s)`` (``None`` when disabled).
    spatial_std
        Per-cell posterior std of ``u_c(s)`` (``None`` when disabled).
    n_train
        Number of labelled wells used for fitting.
    diagnostics
        Free-form dict with backend-specific diagnostic info.
    prior_predictive_only
        ``True`` when the fit was skipped due to insufficient labels and
        the surface is the prior-predictive ``sigma(alpha_c)`` only.
    """

    probability: gpd.GeoDataFrame
    beta_posterior: dict[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    alpha_used: np.ndarray | None = None
    spatial_mean: np.ndarray | None = None
    spatial_std: np.ndarray | None = None
    n_train: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    prior_predictive_only: bool = False


class Fitter(Protocol):
    """Backend-agnostic per-component fitting protocol."""

    def fit(  # noqa: PLR0913, PLR0917
        self,
        component_data: dict,
        alpha_result: Any,
        evidence_config: Any,
        labels: gpd.GeoDataFrame,
        label_column: str,
        spatial_config: Any,
    ) -> ComponentFitResult:
        """Fit the probabilistic model for one component."""
        ...


__all__ = ["ComponentFitResult", "Fitter"]
