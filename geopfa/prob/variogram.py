"""Empirical variogram range estimator for automatic spatial-block-CV sizing.

Used by the CV splitter when ``cross_validation.block_size_km`` is ``None``.
The range is estimated from the semivariogram of the residuals of a pilot
logistic-regression fit (without the spatial field), then the recommended
block size is 1.5x the estimated range.  This ensures the spatial-block
hold-out zones are large enough to prevent autocorrelation leakage.
"""

from __future__ import annotations

import numpy as np


def _validated_variogram_inputs(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int,
    max_distance_frac: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Validate geometry before it can determine a CV holdout policy."""
    coordinates = np.asarray(coords, dtype=float)
    observations = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:  # noqa: PLR2004
        raise ValueError("coords must have shape (N, 2)")
    if observations.ndim != 1 or observations.shape[0] != coordinates.shape[0]:
        raise ValueError("values must be a row-aligned one-dimensional array")
    if coordinates.shape[0] < 4:  # noqa: PLR2004
        raise ValueError("need at least 4 points for variogram estimation")
    if not np.all(np.isfinite(coordinates)) or not np.all(
        np.isfinite(observations)
    ):
        raise ValueError("coords and values must contain only finite values")
    if isinstance(n_bins, bool | np.bool_) or not isinstance(
        n_bins, int | np.integer
    ):
        raise TypeError("n_bins must be an integer")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if (
        isinstance(max_distance_frac, bool | np.bool_)
        or np.iscomplexobj(max_distance_frac)
        or not np.isscalar(max_distance_frac)
    ):
        raise ValueError("max_distance_frac must be a finite fraction in (0, 1]")
    fraction = float(max_distance_frac)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("max_distance_frac must be a finite fraction in (0, 1]")
    span = np.ptp(coordinates, axis=0)
    if not np.any(span > 0.0):
        raise ValueError("variogram coordinates must have positive spatial span")
    return coordinates, observations, int(n_bins), fraction


def empirical_variogram(  # noqa: PLR0914
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 12,
    max_distance_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an empirical (omnidirectional) semivariogram.

    Parameters
    ----------
    coords
        ``(N, 2)`` coordinate array in projected units.
    values
        ``(N,)`` residual / value array.
    n_bins
        Number of lag distance bins.
    max_distance_frac
        Maximum lag as a fraction of the overall coordinate span.

    Returns
    -------
    (lag_midpoints, gamma)
        ``lag_midpoints`` — bin centre distances; ``gamma`` — mean
        squared half-differences per bin.
    """
    coords, values, n_bins, max_distance_frac = _validated_variogram_inputs(
        coords,
        values,
        n_bins=n_bins,
        max_distance_frac=max_distance_frac,
    )

    span = float(
        np.sqrt(
            (coords[:, 0].max() - coords[:, 0].min()) ** 2
            + (coords[:, 1].max() - coords[:, 1].min()) ** 2
        )
    )
    max_lag = span * max_distance_frac

    # Sub-sample to at most 300 points to keep O(n^2) tractable.
    n = len(coords)
    max_pts = 300
    if n > max_pts:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_pts, replace=False)
        coords = coords[idx]
        values = values[idx]

    # Compute all pairwise distances and squared half-differences.
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    val_diff = (values[:, None] - values[None, :]) ** 2 / 2.0

    # Upper triangle only (skip diagonal).
    triu_i, triu_j = np.triu_indices(len(coords), k=1)
    flat_dists = dists[triu_i, triu_j]
    flat_gamma = val_diff[triu_i, triu_j]

    # Filter to max_lag.
    mask = flat_dists <= max_lag
    flat_dists = flat_dists[mask]
    flat_gamma = flat_gamma[mask]

    if len(flat_dists) == 0:
        return np.array([max_lag / 2.0]), np.array([float(np.nanvar(values))])

    edges = np.linspace(0.0, max_lag, n_bins + 1)
    gamma = np.zeros(n_bins)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    for i in range(n_bins):
        in_bin = (flat_dists >= edges[i]) & (flat_dists < edges[i + 1])
        gamma[i] = (
            float(np.nanmean(flat_gamma[in_bin])) if in_bin.any() else np.nan
        )
    return midpoints, gamma


def estimate_variogram_range(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 12,
    max_distance_frac: float = 0.5,
) -> float:
    """Estimate the practical variogram range.

    The range is approximated as the lag at which the semivariogram first
    reaches 95% of its observed sill. If no finite bin reaches that threshold,
    the largest evaluated lag is returned. Invalid or insufficient geometry
    raises instead of silently selecting a cross-validation policy.

    Parameters
    ----------
    coords, values, n_bins, max_distance_frac
        See :func:`empirical_variogram`.

    Returns
    -------
    float
        Estimated range in the same units as ``coords``.
    """
    lags, gamma = empirical_variogram(
        coords, values, n_bins=n_bins, max_distance_frac=max_distance_frac
    )

    finite = np.isfinite(gamma)
    if not finite.any():
        return float(lags[-1])

    sill = float(np.nanmax(gamma[finite]))
    threshold = 0.95 * sill
    above = finite & (gamma >= threshold)
    if above.any():
        return float(lags[above.argmax()])
    return float(lags[-1])


def recommend_block_size_km(  # noqa: PLR0913
    coords: np.ndarray,
    values: np.ndarray,
    *,
    multiplier: float = 1.5,
    min_km: float = 10.0,
    max_km: float = 200.0,
    assumed_crs_unit_m: bool = True,
) -> float:
    """Recommend a spatial-block CV block size from residual variogram range.

    Parameters
    ----------
    coords
        ``(N, 2)`` coordinate array.  Assumed to be in metres when
        ``assumed_crs_unit_m=True``.
    values
        Residuals from a pilot fit.
    multiplier
        Recommended block size = ``multiplier * range`` (default 1.5x).
    min_km, max_km
        Clip the recommendation to this range.
    assumed_crs_unit_m
        If ``True``, convert the range from metres to km for the output.
    """
    def positive_scalar(value: float, name: str) -> float:
        if (
            isinstance(value, bool | np.bool_)
            or np.iscomplexobj(value)
            or not np.isscalar(value)
        ):
            raise TypeError(f"{name} must be a positive finite scalar")
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{name} must be a positive finite scalar")
        return parsed

    multiplier_value = positive_scalar(multiplier, "multiplier")
    min_value = positive_scalar(min_km, "min_km")
    max_value = positive_scalar(max_km, "max_km")
    if min_value > max_value:
        raise ValueError("min_km must be less than or equal to max_km")
    if not isinstance(assumed_crs_unit_m, bool | np.bool_):
        raise TypeError("assumed_crs_unit_m must be boolean")

    range_units = estimate_variogram_range(coords, values)
    if not np.isfinite(range_units) or range_units <= 0.0:
        raise RuntimeError("variogram estimator returned a non-positive range")

    range_km = range_units / 1000.0 if assumed_crs_unit_m else range_units
    return float(np.clip(multiplier_value * range_km, min_value, max_value))


__all__ = [
    "empirical_variogram",
    "estimate_variogram_range",
    "recommend_block_size_km",
]
