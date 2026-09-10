"""Dimension-aware spatial alignment for probabilistic PFA inputs.

GeoPandas nearest-neighbour operations are planar and therefore ignore point
Z coordinates.  The probabilistic workflow uses these helpers so 3-D wells,
voxel priors, and evidence layers are aligned in every declared coordinate
dimension.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from geopfa.exceptions import GEOPFAValueError


def extract_coordinates(gdf: gpd.GeoDataFrame) -> NDArray[np.float64]:
    """Extract a finite two- or three-dimensional coordinate array."""
    if len(gdf) == 0:
        raise GEOPFAValueError(
            "coordinate table must contain at least one geometry"
        )
    has_z = gdf.geometry.has_z.to_numpy(dtype=bool)
    if has_z.any() and not has_z.all():
        raise GEOPFAValueError("geometry column cannot mix 2-D and 3-D points")
    columns = [
        gdf.geometry.x.to_numpy(dtype=float),
        gdf.geometry.y.to_numpy(dtype=float),
    ]
    if has_z.all():
        columns.append(gdf.geometry.z.to_numpy(dtype=float))
    coordinates = np.column_stack(columns)
    if not np.all(np.isfinite(coordinates)):
        raise GEOPFAValueError("point geometries must have finite coordinates")
    return coordinates


def align_to_grid_crs(
    points_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Return point observations in the prediction grid's CRS."""
    points_crs = points_gdf.crs
    grid_crs = grid_gdf.crs
    if points_crs is None or grid_crs is None:
        if points_crs != grid_crs:
            raise GEOPFAValueError(
                "point and prediction-grid geometries must both declare a CRS"
            )
        return points_gdf
    if points_crs == grid_crs:
        return points_gdf
    return points_gdf.to_crs(grid_crs)


def snap_to_grid_indices(
    points_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    *,
    max_distance: float | None = None,
) -> NDArray[np.int64]:
    """Return nearest grid rows using every declared coordinate dimension."""
    aligned = align_to_grid_crs(points_gdf, grid_gdf)
    points = extract_coordinates(aligned)
    grid = extract_coordinates(grid_gdf)
    if points.shape[1] != grid.shape[1]:
        raise GEOPFAValueError(
            "point and grid geometries must have the same coordinate dimension; "
            f"got {points.shape[1]} and {grid.shape[1]}"
        )
    for dimension in range(grid.shape[1]):
        lower, upper = _axis_footprint_bounds(grid[:, dimension])
        outside = (points[:, dimension] < lower) | (
            points[:, dimension] > upper
        )
        if np.any(outside):
            raise GEOPFAValueError(
                "point coordinates fall outside the grid cell footprint in "
                f"dimension {dimension}; supported interval is "
                f"[{lower:.6g}, {upper:.6g}]"
            )
    distances, indices = cKDTree(grid).query(points, k=1)
    if max_distance is not None:
        distance_limit = float(max_distance)
        if not np.isfinite(distance_limit) or distance_limit < 0.0:
            raise GEOPFAValueError(
                "max_distance must be non-negative and finite"
            )
        if np.any(distances > distance_limit):
            raise GEOPFAValueError(
                "nearest grid match exceeds max_distance; maximum observed "
                f"distance={float(np.max(distances)):.6g}, "
                f"limit={distance_limit:.6g}"
            )
    return np.asarray(indices, dtype=np.int64)


def grid_footprint_mask(
    points_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> NDArray[np.bool_]:
    """Identify points inside every dimension of a grid's cell footprint.

    The returned mask is suitable for defining a covariate-support cohort
    before outcomes are fitted. Coordinates are reprojected to the grid CRS,
    and no out-of-support point is clipped or snapped to an edge cell.
    """
    aligned = align_to_grid_crs(points_gdf, grid_gdf)
    points = extract_coordinates(aligned)
    grid = extract_coordinates(grid_gdf)
    if points.shape[1] != grid.shape[1]:
        raise GEOPFAValueError(
            "point and grid geometries must have the same coordinate dimension; "
            f"got {points.shape[1]} and {grid.shape[1]}"
        )
    inside = np.ones(points.shape[0], dtype=bool)
    for dimension in range(grid.shape[1]):
        lower, upper = _axis_footprint_bounds(grid[:, dimension])
        inside &= (points[:, dimension] >= lower) & (
            points[:, dimension] <= upper
        )
    return inside


def sample_layer_at_points(
    points_gdf: gpd.GeoDataFrame,
    layer_gdf: gpd.GeoDataFrame,
    value_col: str,
) -> NDArray[np.float64]:
    """Sample one gridded layer at points using dimension-aware snapping."""
    if value_col not in layer_gdf.columns:
        raise GEOPFAValueError(
            f"evidence layer is missing value column {value_col!r}"
        )
    indices = snap_to_grid_indices(points_gdf, layer_gdf)
    return layer_gdf.iloc[indices][value_col].to_numpy(dtype=float)


def require_same_grid(
    reference: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    *,
    context: str,
) -> None:
    """Require identical coordinate support and row order for grid arrays."""
    if len(reference) != len(candidate):
        raise GEOPFAValueError(
            f"{context} has {len(candidate)} rows; expected {len(reference)}"
        )
    aligned = align_to_grid_crs(candidate, reference)
    reference_coords = extract_coordinates(reference)
    candidate_coords = extract_coordinates(aligned)
    if reference_coords.shape != candidate_coords.shape or not np.allclose(
        reference_coords,
        candidate_coords,
        rtol=0.0,
        atol=1e-8,
    ):
        raise GEOPFAValueError(
            f"{context} is not aligned to the reference grid in row order"
        )


def _axis_boundary_tolerance(axis: NDArray[np.float64]) -> float:
    """Return a sub-per-mille cell tolerance for coordinate serialization."""
    scale = max(float(np.ptp(axis)), 1.0)
    roundoff = 32.0 * np.finfo(float).eps * scale
    cell_fraction = (
        0.001 * float(np.min(np.diff(axis))) if axis.size > 1 else 0.0
    )
    return max(roundoff, cell_fraction)


def _axis_footprint_bounds(
    coordinates: NDArray[np.float64],
) -> tuple[float, float]:
    """Return outer cell-edge bounds for one grid coordinate dimension."""
    axis = np.unique(coordinates)
    tolerance = _axis_boundary_tolerance(axis)
    if axis.size == 1:
        return float(axis[0] - tolerance), float(axis[0] + tolerance)
    return (
        float(axis[0] - 0.5 * (axis[1] - axis[0]) - tolerance),
        float(axis[-1] + 0.5 * (axis[-1] - axis[-2]) + tolerance),
    )


def grid_values_on_reference(
    reference: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    value_col: str,
    *,
    context: str,
) -> NDArray[np.float64]:
    """Return candidate values on a 2-D or 3-D reference grid.

    Exact row-aligned grids pass through. Different supports require a
    complete rectilinear candidate grid that covers the reference, in which
    case values are linearly interpolated in every coordinate dimension.
    """
    if value_col not in candidate.columns:
        raise GEOPFAValueError(f"{context} is missing {value_col!r}")
    aligned = align_to_grid_crs(candidate, reference)
    reference_coords = extract_coordinates(reference)
    candidate_coords = extract_coordinates(aligned)
    if reference_coords.shape[1] != candidate_coords.shape[1]:
        raise GEOPFAValueError(
            f"{context} has coordinate dimension {candidate_coords.shape[1]}; "
            f"expected {reference_coords.shape[1]}"
        )
    values = aligned[value_col].to_numpy(dtype=float)
    if reference_coords.shape == candidate_coords.shape and np.allclose(
        reference_coords,
        candidate_coords,
        rtol=0.0,
        atol=1e-8,
    ):
        return values

    axes = tuple(
        np.unique(candidate_coords[:, dimension])
        for dimension in range(candidate_coords.shape[1])
    )
    expected_rows = int(np.prod([axis.size for axis in axes], dtype=np.int64))
    if expected_rows != candidate_coords.shape[0]:
        raise GEOPFAValueError(
            f"{context} is not a complete rectilinear grid; explicit "
            "resampling is required upstream"
        )
    axis_indices = tuple(
        np.searchsorted(axis, candidate_coords[:, dimension])
        for dimension, axis in enumerate(axes)
    )
    flat_indices = np.ravel_multi_index(
        axis_indices,
        tuple(axis.size for axis in axes),
    )
    if np.unique(flat_indices).size != candidate_coords.shape[0]:
        raise GEOPFAValueError(
            f"{context} contains duplicate grid coordinates"
        )

    interpolation_values = np.empty(
        tuple(axis.size for axis in axes), dtype=float
    )
    interpolation_values[axis_indices] = values
    evaluation_coords = reference_coords.copy()
    for dimension, axis in enumerate(axes):
        tolerance = _axis_boundary_tolerance(axis)
        if (
            float(np.min(evaluation_coords[:, dimension]))
            < axis[0] - tolerance
            or float(np.max(evaluation_coords[:, dimension]))
            > axis[-1] + tolerance
        ):
            raise GEOPFAValueError(
                f"{context} does not cover the reference grid in coordinate "
                f"dimension {dimension}"
            )
        evaluation_coords[:, dimension] = np.clip(
            evaluation_coords[:, dimension], axis[0], axis[-1]
        )
    interpolator = RegularGridInterpolator(
        axes,
        interpolation_values,
        method="linear",
        bounds_error=True,
    )
    return np.asarray(interpolator(evaluation_coords), dtype=float)


__all__ = [
    "align_to_grid_crs",
    "extract_coordinates",
    "grid_footprint_mask",
    "grid_values_on_reference",
    "require_same_grid",
    "sample_layer_at_points",
    "snap_to_grid_indices",
]
