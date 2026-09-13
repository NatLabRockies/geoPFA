"""Spatial random field ``u_c(s)`` via the existing geopfa.extrapolation GP.

Thin wrapper around :func:`geopfa.extrapolation.build_and_fit_gp` and
:func:`geopfa.extrapolation.get_predictions` so the probabilistic engine
gets a proper LatticeKrigX field with ARD lengthscale bounds from the
coordinate radius, MLE optimisation, and predictive ``(mean, std)`` — all
already implemented in the existing extrapolation module.

The regression backend is generic over feature dimension ``D``, so the same
code path serves 2D ``u_c(s) = u_c(x, y)`` and 3D ``u_c(x, y, z)`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geopfa.extrapolation import (
    build_and_fit_gp,
    estimate_variance,
    get_predictions,
    standardize_xy,
)


# Tight numerical clip on the returned mean: u_c is a logit-space correction,
# so values outside ±3 logit units would imply post-sigmoid probabilities
# pinned at the asymptotes and are almost always pathological in this regime.
_U_CLIP = 3.0
_MIN_TRAINING_POINTS = 4
_PREDICTION_CHUNK_SIZE = 6_000


@dataclass(frozen=True)
class SpatialFieldResult:
    """Output of a GP fit + grid prediction for the spatial random field.

    Attributes
    ----------
    u_mean
        Per-grid-cell posterior mean of ``u_c(s)`` (clipped to ±3 logit).
    u_std
        Per-grid-cell posterior standard deviation. Useful for uncertainty
        bands on the final probability map.
    model
        The fitted spatial model. This is ``None`` only when the supplied
        residual response is exactly constant.
    diagnostics
        Free-form dict with kernel summary, hyperparameters, and any
        notes from the underlying ``build_and_fit_gp`` call.
    """

    u_mean: np.ndarray
    u_std: np.ndarray
    model: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _constant_field(
    n_grid: int,
    value: float,
    *,
    n_train: int,
    reason: str,
) -> SpatialFieldResult:
    """Return an identified constant logit correction without a GP fit."""
    return SpatialFieldResult(
        u_mean=np.full(n_grid, np.clip(value, -_U_CLIP, _U_CLIP), dtype=float),
        u_std=np.zeros(n_grid, dtype=float),
        model=None,
        diagnostics={
            "degenerate": reason,
            "kernel": None,
            "n_train": n_train,
        },
    )


def fit_spatial_field_gp(  # noqa: PLR0914
    train_coords: np.ndarray,
    train_residuals: np.ndarray,
    grid_coords: np.ndarray,
) -> SpatialFieldResult:
    """Fit ``u_c(s)`` on ``(train_coords, train_residuals)`` and predict on grid.

    Parameters
    ----------
    train_coords
        Training coordinates, shape ``(N, D)`` where ``D`` is 2 (surface
        runs) or 3 (3D runs with a per-voxel residual). The wrapper does
        not check ``D``; it passes the array straight through to the
        underlying regression backend.
    train_residuals
        Training residuals in logit space, shape ``(N,)``.
    grid_coords
        Prediction coordinates, shape ``(M, D)``.
    Returns
    -------
    SpatialFieldResult

    Every supplied coordinate dimension enters the model. A 3-D call thus
    estimates a volumetric field rather than broadcasting a surface field
    over depth.
    """
    train_coords = np.asarray(train_coords, dtype=float)
    train_residuals = np.asarray(train_residuals, dtype=float)
    grid_coords = np.asarray(grid_coords, dtype=float)

    if train_coords.ndim != 2 or grid_coords.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            "train_coords and grid_coords must both be 2-D arrays of shape "
            "(N, D)",
        )
    if train_coords.shape[1] != grid_coords.shape[1]:
        raise ValueError(
            "train_coords and grid_coords must share their D-axis "
            f"(got {train_coords.shape[1]} vs {grid_coords.shape[1]})",
        )
    if train_residuals.shape != (train_coords.shape[0],):
        raise ValueError(
            "train_residuals must be a one-dimensional vector aligned with "
            f"train_coords; got {train_residuals.shape} for "
            f"{train_coords.shape[0]} coordinate rows"
        )
    if not (
        np.all(np.isfinite(train_coords))
        and np.all(np.isfinite(grid_coords))
        and np.all(np.isfinite(train_residuals))
    ):
        raise ValueError(
            "train_coords, grid_coords, and train_residuals must be finite"
        )

    if len(train_coords) < _MIN_TRAINING_POINTS:
        raise ValueError(
            "too few training points to fit a GP spatial field "
            f"({len(train_coords)} < {_MIN_TRAINING_POINTS})"
        )

    # Standardise coordinates using training-set statistics so the GP
    # operates on a well-conditioned space.
    X_train_std, X_grid_std, _x_mean, _x_std = standardize_xy(
        train_coords, grid_coords
    )

    # Standardise residuals too so the GP's variance prior is well-scaled.
    y_mean = float(np.mean(train_residuals))
    y_scale = float(np.std(train_residuals))
    if y_scale < 1e-10:  # noqa: PLR2004
        # Spatial variation is absent, but the constant logit correction is
        # still identified and must not be silently replaced by zero.
        return _constant_field(
            len(grid_coords),
            y_mean,
            n_train=len(train_coords),
            reason="constant_residuals",
        )
    Y_train_std = ((train_residuals - y_mean) / y_scale).reshape(-1, 1)

    try:
        model, kernel_info = build_and_fit_gp(
            X_train_std,
            Y_train_std,
        )
        # Batch predictions for large grids to stay within memory budget.
        n_grid = len(X_grid_std)
        if n_grid <= _PREDICTION_CHUNK_SIZE:
            Y_pred_std, Y_std_std = get_predictions(
                model, X_grid_std, Y_mean=y_mean, Y_std=y_scale
            )
        else:
            preds = np.empty(n_grid, dtype=float)
            stds = np.empty(n_grid, dtype=float)
            for start in range(0, n_grid, _PREDICTION_CHUNK_SIZE):
                chunk = X_grid_std[start : start + _PREDICTION_CHUNK_SIZE]
                p, s = get_predictions(
                    model, chunk, Y_mean=y_mean, Y_std=y_scale
                )
                preds[start : start + _PREDICTION_CHUNK_SIZE] = np.asarray(
                    p
                ).ravel()
                stds[start : start + _PREDICTION_CHUNK_SIZE] = np.asarray(
                    s
                ).ravel()
            Y_pred_std, Y_std_std = preds, stds
    except (
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
        MemoryError,
    ) as exc:
        raise RuntimeError(
            f"GP spatial-field fit failed ({type(exc).__name__}: {exc})"
        ) from exc

    u_mean = np.clip(np.asarray(Y_pred_std).ravel(), -_U_CLIP, _U_CLIP)
    u_std = np.asarray(Y_std_std).ravel()

    diagnostics: dict[str, Any] = {
        "kernel": str(kernel_info.get("kernel_components"))
        if isinstance(kernel_info, dict)
        else None,
        "n_train": len(train_coords),
        "residual_variance_estimate": float(
            estimate_variance(train_residuals)
        ),
    }
    return SpatialFieldResult(
        u_mean=u_mean,
        u_std=u_std,
        model=model,
        diagnostics=diagnostics,
    )


__all__ = ["SpatialFieldResult", "fit_spatial_field_gp"]
