"""Spatial block cross-validation splitter for the probabilistic method.

Provides:

* :func:`spatial_block_cv` — generator yielding ``(train_mask, test_mask)``
  boolean pairs, one per fold. Two block-partitioning strategies:
  ``"grid"`` (uniform square blocks) and ``"kmeans"`` (geographic clusters).
* :class:`SpatialBlockKFold` — an sklearn-compatible ``BaseCrossValidator``
  wrapper that lets the splitter drop straight into
  ``cross_val_score`` / ``cross_val_predict``.

The strategies reuse the existing block-id machinery in
:mod:`geopfa.prob.scenario` coordinate-trend diagnostic so behaviour stays consistent with the
already-merged Stage-1 spatial-block hold-out helper.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .scenario import _balanced_block_fold_ids, _block_ids

_MIN_N_FOLDS = 2


def _grid_blocks(coords: np.ndarray, grid_size: int) -> np.ndarray:
    return _block_ids(coords[:, :2], grid_size=grid_size)


def _kmeans_blocks(coords: np.ndarray, n_blocks: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans  # noqa: PLC0415 — heavy import on demand

    km = KMeans(n_clusters=n_blocks, n_init="auto", random_state=seed)
    return km.fit_predict(coords)


def _fixed_width_blocks(coords: np.ndarray, block_width: float) -> np.ndarray:
    """Assign 2-D points to fixed-width square blocks."""
    if not np.isfinite(block_width) or block_width <= 0.0:
        raise ValueError("block_size_km must be positive and finite")
    origin = np.min(coords, axis=0)
    indices = np.floor((coords - origin) / block_width).astype(np.int64)
    _, block_ids = np.unique(indices, axis=0, return_inverse=True)
    return block_ids


def spatial_block_cv(  # noqa: PLR0912, PLR0913, PLR0915
    coords: np.ndarray,
    *,
    n_folds: int = 5,
    block_type: str = "grid",
    grid_size: int = 4,
    seed: int = 0,
    residuals: np.ndarray | None = None,
    block_size_km: float | None = None,
    buffer_distance: float = 0.0,
    dims: tuple[int, ...] = (0, 1),
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_mask, test_mask)`` pairs for spatial-block CV.

    Parameters
    ----------
    coords
        ``(N, D)`` coordinate array.
    n_folds
        Number of folds to yield.
    block_type
        ``"grid"`` uses a ``grid_size x grid_size`` uniform partition.
        ``"kmeans"`` uses K-means on the (x, y) projection.
    grid_size
        For ``"grid"``: side length of the square block grid. For
        ``"kmeans"``: cluster-count multiplier.
    seed
        RNG seed for the K-means initialisation.
    residuals
        Optional pilot-fit residuals used to auto-size the block when
        ``block_size_km`` is ``None``.  If both are ``None``, the default
        ``grid_size`` is used as-is.
    block_size_km
        Override the block size in km (used for grid-type only).  When
        ``None`` and ``residuals`` are provided, the block size is
        estimated from the empirical variogram range.
    buffer_distance
        Non-negative distance excluded from training around each test fold,
        in the coordinate units of ``dims``.
    dims
        Coordinate dimensions used both for blocking and buffering. The
        default ``(0, 1)`` holds out whole surface sites, including all depths.

    Yields
    ------
    (train_mask, test_mask)
        Boolean arrays of length ``N``. Train/test are disjoint; rows inside
        the exclusion buffer belong to neither set for that fold.

    Raises
    ------
    ValueError
        If ``block_type`` is not one of ``"grid"`` / ``"kmeans"``.

    Notes
    -----
    For volumetric models the default surface dimensions intentionally keep
    every depth from one site in the same fold. Callers can select all three
    dimensions for a fully volumetric K-means holdout.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:  # noqa: PLR2004
        raise ValueError("coords must be a 2-D array of shape (N, >=2)")
    if coords.shape[0] < 2:  # noqa: PLR2004
        raise ValueError("coords must contain at least two observations")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite values")
    if (
        isinstance(n_folds, bool | np.bool_)
        or not isinstance(n_folds, int | np.integer)
        or n_folds < _MIN_N_FOLDS
    ):
        raise ValueError("n_folds must be an integer >= 2")
    if n_folds > coords.shape[0]:
        raise ValueError("n_folds cannot exceed the number of observations")
    if (
        isinstance(grid_size, bool | np.bool_)
        or not isinstance(grid_size, int | np.integer)
        or grid_size < 1
    ):
        raise ValueError("grid_size must be a positive integer")
    if block_type not in {"grid", "kmeans"}:
        raise ValueError(
            f"block_type must be 'grid' or 'kmeans' (got {block_type!r})"
        )
    if block_size_km is not None:
        if (
            isinstance(block_size_km, bool | np.bool_)
            or np.iscomplexobj(block_size_km)
            or not np.isscalar(block_size_km)
            or not np.isfinite(block_size_km)
            or float(block_size_km) <= 0.0
        ):
            raise ValueError("block_size_km must be positive and finite")
        if block_type != "grid":
            raise ValueError(
                "block_size_km is only defined for block_type='grid'"
            )
    if residuals is not None:
        residual_array = np.asarray(residuals)
        if (
            np.iscomplexobj(residual_array)
            or residual_array.shape != (coords.shape[0],)
            or not np.all(np.isfinite(residual_array))
        ):
            raise ValueError(
                "residuals must be a finite real vector aligned with coords"
            )
    if (
        not isinstance(dims, tuple)
        or not dims
        or len(set(dims)) != len(dims)
        or any(
            isinstance(dim, bool | np.bool_)
            or not isinstance(dim, int | np.integer)
            or dim < 0
            or dim >= coords.shape[1]
            for dim in dims
        )
    ):
        raise ValueError("dims must be unique coordinate-column indices")
    if (
        isinstance(buffer_distance, bool | np.bool_)
        or np.iscomplexobj(buffer_distance)
        or not np.isscalar(buffer_distance)
    ):
        raise ValueError("buffer_distance must be non-negative and finite")
    buffer_value = float(buffer_distance)
    if not np.isfinite(buffer_value) or buffer_value < 0.0:
        raise ValueError("buffer_distance must be non-negative and finite")

    # Auto-estimate block size from variogram when residuals are available.
    effective_grid_size = grid_size
    if (
        block_type == "grid"
        and block_size_km is None
        and residuals is not None
    ):
        from .variogram import recommend_block_size_km  # noqa: PLC0415

        rec_km = recommend_block_size_km(coords[:, :2], np.asarray(residuals))
        span_km = (
            float(
                np.sqrt(
                    (coords[:, 0].max() - coords[:, 0].min()) ** 2
                    + (coords[:, 1].max() - coords[:, 1].min()) ** 2
                )
            )
            / 1000.0
        )
        if span_km > 0:
            effective_grid_size = max(2, int(np.ceil(span_km / rec_km)))

    blocking_coords = coords[:, list(dims)]
    if block_type == "grid":
        if blocking_coords.shape[1] != 2:  # noqa: PLR2004
            raise ValueError(
                "grid block_type currently requires exactly two dims"
            )
        if block_size_km is not None:
            blocks = _fixed_width_blocks(
                blocking_coords, float(block_size_km) * 1000.0
            )
        else:
            blocks = _grid_blocks(
                blocking_coords, grid_size=effective_grid_size
            )
    elif block_type == "kmeans":
        if n_folds * grid_size > coords.shape[0]:
            raise ValueError(
                "n_folds * grid_size cannot exceed the number of observations "
                "for block_type='kmeans'"
            )
        blocks = _kmeans_blocks(
            blocking_coords, n_blocks=n_folds * grid_size, seed=seed
        )

    fold_of = _balanced_block_fold_ids(blocks, n_folds)
    for fold in range(n_folds):
        test_mask = fold_of == fold
        train_mask = ~test_mask
        if not test_mask.any():
            raise ValueError(f"spatial fold {fold} contains zero test rows")
        if buffer_value > 0.0:
            tree = cKDTree(blocking_coords[test_mask])
            candidate_indices = np.flatnonzero(train_mask)
            distances, _ = tree.query(blocking_coords[candidate_indices], k=1)
            train_mask[candidate_indices[distances < buffer_value]] = False
        if not train_mask.any():
            raise ValueError(
                f"spatial fold {fold} contains zero training rows after "
                f"buffer_distance={buffer_value:g}"
            )
        yield train_mask, test_mask


@dataclass
class SpatialBlockKFold:
    """sklearn-compatible ``BaseCrossValidator`` for spatial block CV.

    Designed to drop into ``cross_val_score`` / ``cross_val_predict`` with
    pre-computed coords. The ``X`` / ``y`` arrays passed to ``split`` are
    only used to compute ``n_samples`` — the actual partitioning uses the
    coordinates supplied at construction.
    """

    coords: np.ndarray
    n_splits: int = 5
    block_type: str = "grid"
    grid_size: int = 4
    block_size_km: float | None = None
    seed: int = 0
    buffer_distance: float = 0.0
    dims: tuple[int, ...] = (0, 1)

    def get_n_splits(
        self,
        X: np.ndarray | None = None,  # noqa: ARG002
        y: np.ndarray | None = None,  # noqa: ARG002
        groups: np.ndarray | None = None,  # noqa: ARG002
    ) -> int:
        """Return the number of folds."""
        return int(self.n_splits)

    def split(
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray | None = None,  # noqa: ARG002
        groups: np.ndarray | None = None,  # noqa: ARG002
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` for each fold."""
        for train_mask, test_mask in spatial_block_cv(
            self.coords,
            n_folds=self.n_splits,
            block_type=self.block_type,
            grid_size=self.grid_size,
            block_size_km=self.block_size_km,
            seed=self.seed,
            buffer_distance=self.buffer_distance,
            dims=self.dims,
        ):
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


__all__ = ["SpatialBlockKFold", "spatial_block_cv"]
