"""Output writers for the probabilistic-method runner.

Supports GeoTIFF (2D rasters via point-grid rasterisation), Parquet (long-form
per-cell tables), and VTK (3D point clouds via PyVista). All writers operate
on per-component GeoDataFrames with a ``probability`` column (and optional
uncertainty / spatial-residual columns) — the same shape the runner produces.

A :func:`write_manifest` helper records every file written, the config hash,
and a run id so downstream tooling can verify reproducibility.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess  # noqa: S404 -- fixed Git commands capture source provenance
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import (
    ALLOWED_OUTPUT_FORMATS,
    ProbabilisticConfig,
    validate_surface_names,
)
from geopfa.exceptions import GEOPFAValueError

_MIN_POINTS_FOR_SPACING_CHECK = 3
_DRAW_ARRAY_DIMENSIONS = 2
_SUMMARY_CELL_CHUNK_SIZE = 2_048
_INCREMENTAL_DRAW_SCHEMA_VERSION = 2
_RUN_RESUME_SCHEMA_VERSION = 1
_RUN_RESUME_FILENAME = ".probabilistic_run.incomplete.json"
_RUNTIME_SOURCE_SUFFIXES = {
    ".c",
    ".dll",
    ".dylib",
    ".h",
    ".py",
    ".pyd",
    ".pyi",
    ".so",
}
_SHAPEFILE_REQUIRED_SIDECARS = (".dbf", ".shx")
_SHAPEFILE_OPTIONAL_SIDECARS = (".cpg", ".prj")
_SPATIAL_DIMENSION_2D = 2
_SPATIAL_DIMENSION_3D = 3


def _surface_point_dimension(name: str, gdf: gpd.GeoDataFrame) -> int:
    """Validate point geometry and return its uniform coordinate dimension."""
    geometry = gdf.geometry
    if geometry.isna().any() or geometry.is_empty.any():
        raise ValueError(
            f"surface {name!r} geometry must contain non-empty points"
        )
    if not geometry.geom_type.eq("Point").all():
        raise ValueError(f"surface {name!r} geometry must contain only points")
    has_z = geometry.has_z.to_numpy(dtype=bool)
    if has_z.any() and not has_z.all():
        raise ValueError(f"surface {name!r} cannot mix 2-D and 3-D points")
    dimension = _SPATIAL_DIMENSION_3D if has_z.all() else _SPATIAL_DIMENSION_2D
    axes = [
        geometry.x.to_numpy(dtype=float),
        geometry.y.to_numpy(dtype=float),
    ]
    if dimension == _SPATIAL_DIMENSION_3D:
        axes.append(geometry.z.to_numpy(dtype=float))
    if not np.isfinite(np.column_stack(axes)).all():
        raise ValueError(f"surface {name!r} coordinates must be finite")
    return dimension


def _validate_surface_value(
    name: str, gdf: gpd.GeoDataFrame, *, value_col: str
) -> None:
    """Require one named surface value to be present, numeric, and finite."""
    if value_col not in gdf.columns:
        raise ValueError(
            f"surface {name!r} is missing requested column {value_col!r}"
        )
    try:
        values = gdf[value_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"surface {name!r} column {value_col!r} must be numeric"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(
            f"surface {name!r} column {value_col!r} must be finite"
        )


def _preflight_probability_surfaces(
    surfaces: Mapping[str, gpd.GeoDataFrame],
    *,
    value_col: str = "probability",
    required_dimension: int | None = None,
    format_name: str | None = None,
) -> None:
    """Validate a complete writer request before filesystem mutation."""
    validate_surface_names(surfaces, context="probability surface")
    dimensions: set[int] = set()
    for name, gdf in surfaces.items():
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(
                f"surface {name!r} must be a GeoDataFrame; "
                f"got {type(gdf).__name__}"
            )
        if len(gdf) == 0:
            continue
        _validate_surface_value(name, gdf, value_col=value_col)
        dimension = _surface_point_dimension(name, gdf)
        dimensions.add(dimension)
        if required_dimension is not None and dimension != required_dimension:
            label = format_name or "requested"
            raise ValueError(
                f"{label} output requires {required_dimension}-D surfaces; "
                f"surface {name!r} is {dimension}-D"
            )
    if len(dimensions) > 1:
        raise ValueError("one output request cannot mix 2-D and 3-D surfaces")


def _preflight_geotiff_surfaces(
    surfaces: Mapping[str, gpd.GeoDataFrame], *, value_col: str
) -> None:
    """Validate GeoTIFF dimensionality and raster support for every surface."""
    _preflight_probability_surfaces(
        surfaces,
        value_col=value_col,
        required_dimension=2,
        format_name="GeoTIFF",
    )
    for gdf in surfaces.values():
        if len(gdf) > 0:
            _gdf_to_raster(gdf, value_col=value_col)


def _preflight_vtk_attributes(
    surfaces: Mapping[str, gpd.GeoDataFrame],
) -> None:
    """Require VTK point attributes to be numeric before any file is written."""
    _preflight_probability_surfaces(
        surfaces,
        required_dimension=3,
        format_name="VTK",
    )
    for name, gdf in surfaces.items():
        if len(gdf) == 0:
            continue
        geometry_name = gdf.geometry.name
        for column in gdf.columns:
            if column == geometry_name:
                continue
            try:
                values = gdf[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"surface {name!r} VTK attribute {column!r} must be numeric"
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(
                    f"surface {name!r} VTK attribute {column!r} must be finite"
                )


def _preflight_output_request(
    surfaces: Mapping[str, gpd.GeoDataFrame],
    *,
    formats: tuple[str, ...],
    include_uncertainty: bool,
) -> None:
    """Preflight every requested output format as one validation transaction."""
    _preflight_probability_surfaces(surfaces)
    if "geotiff" in formats:
        _preflight_geotiff_surfaces(surfaces, value_col="probability")
        if include_uncertainty:
            for value_col in (
                "spatial_u_std",
                "probability_lo",
                "probability_hi",
            ):
                selected = {
                    name: gdf
                    for name, gdf in surfaces.items()
                    if value_col in gdf.columns
                }
                _preflight_geotiff_surfaces(selected, value_col=value_col)
    if "vtk" in formats:
        _preflight_vtk_attributes(surfaces)


# ---------------------------------------------------------------------------
# GeoTIFF rasterisation (2D)
# ---------------------------------------------------------------------------


def _gdf_to_raster(  # noqa: PLR0914
    gdf: gpd.GeoDataFrame, *, value_col: str
) -> tuple[np.ndarray, rasterio.Affine, str | None]:
    """Convert a regular point grid GDF to a 2D raster array + affine transform.

    Requires a complete regular rectangular grid. Irregular support must be
    resampled explicitly upstream so the scientific transformation is named
    and auditable.
    """
    xs = gdf.geometry.x.to_numpy(dtype=float)
    ys = gdf.geometry.y.to_numpy(dtype=float)
    vals = gdf[value_col].to_numpy(dtype=float)
    ux = np.unique(xs)
    uy = np.unique(ys)
    coordinate_pairs = np.column_stack([xs, ys])
    complete = len(ux) * len(uy) == len(gdf) and np.unique(
        coordinate_pairs, axis=0
    ).shape[0] == len(gdf)
    regular_x = len(ux) < _MIN_POINTS_FOR_SPACING_CHECK or np.allclose(
        np.diff(ux), np.diff(ux)[0]
    )
    regular_y = len(uy) < _MIN_POINTS_FOR_SPACING_CHECK or np.allclose(
        np.diff(uy), np.diff(uy)[0]
    )
    if not complete or not regular_x or not regular_y:
        raise ValueError(
            "GeoTIFF output requires a complete rectilinear grid with "
            "regular axis spacing; resample irregular support explicitly upstream"
        )
    nx, ny = len(ux), len(uy)
    x_idx = np.searchsorted(ux, xs)
    y_idx = np.searchsorted(uy, ys)
    grid = np.full((ny, nx), np.nan, dtype=float)
    # GeoTIFF rows are top-down; flip the y index.
    grid[ny - 1 - y_idx, x_idx] = vals
    xsize = float(ux[1] - ux[0]) if len(ux) > 1 else 1.0
    ysize = float(uy[1] - uy[0]) if len(uy) > 1 else 1.0
    transform = from_origin(
        west=float(ux[0]) - 0.5 * xsize,
        north=float(uy[-1]) + 0.5 * ysize,
        xsize=xsize,
        ysize=ysize,
    )

    crs = str(gdf.crs) if gdf.crs is not None else None
    return grid.astype(np.float32), transform, crs


def write_geotiff_outputs(
    surfaces: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    *,
    value_col: str = "probability",
) -> list[Path]:
    """Write one GeoTIFF per surface (2D point grids only).

    Three-dimensional input is rejected; use :func:`write_vtk_outputs` for it.
    """
    _preflight_geotiff_surfaces(surfaces, value_col=value_col)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, gdf in surfaces.items():
        if len(gdf) == 0:
            continue
        arr, transform, crs = _gdf_to_raster(gdf, value_col=value_col)
        out_path = output_dir / f"{name}_{value_col}.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(arr, 1)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Parquet (long form, any dimensionality)
# ---------------------------------------------------------------------------


def write_parquet_outputs(
    surfaces: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
) -> list[Path]:
    """Write one Parquet file per surface with (x, y[, z], probability)."""
    _preflight_probability_surfaces(surfaces)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, gdf in surfaces.items():
        if len(gdf) == 0:
            continue
        out_df = gdf.copy()
        out_df["x"] = out_df.geometry.x
        out_df["y"] = out_df.geometry.y
        if out_df.geometry.has_z.any():
            out_df["z"] = out_df.geometry.z
        out_df = out_df.drop(columns=[out_df.geometry.name])
        out_path = output_dir / f"{name}_probability.parquet"
        out_df.to_parquet(out_path, index=False)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# VTK (3D point clouds)
# ---------------------------------------------------------------------------


def write_vtk_outputs(
    surfaces: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
) -> list[Path]:
    """Write one VTK ``.vtp`` per 3D surface."""
    _preflight_vtk_attributes(surfaces)
    eligible = {name: gdf for name, gdf in surfaces.items() if len(gdf) > 0}
    if not eligible:
        output_dir.mkdir(parents=True, exist_ok=True)
        return []
    try:
        import pyvista as pv  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyVista is required to write requested 3-D VTK outputs"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, gdf in eligible.items():
        coords = np.column_stack(
            [
                gdf.geometry.x.to_numpy(),
                gdf.geometry.y.to_numpy(),
                gdf.geometry.z.to_numpy(),
            ]
        )
        cloud = pv.PolyData(coords)
        geometry_name = gdf.geometry.name
        for col in gdf.columns:
            if col == geometry_name:
                continue
            cloud.point_data[col] = gdf[col].to_numpy(dtype=float)
        out_path = output_dir / f"{name}_probability.vtp"
        cloud.save(out_path)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Unified writer + manifest
# ---------------------------------------------------------------------------


def write_csv_outputs(
    surfaces: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
) -> list[Path]:
    """Write one coordinate-explicit CSV per probability surface."""
    _preflight_probability_surfaces(surfaces)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, gdf in surfaces.items():
        if len(gdf) == 0:
            continue
        out_df = gdf.copy()
        geometry = out_df.geometry
        out_df["x"] = geometry.x
        out_df["y"] = geometry.y
        if geometry.has_z.any():
            out_df["z"] = geometry.z
        out_df = out_df.drop(columns=[geometry.name])
        out_path = output_dir / f"{name}_probability.csv"
        out_df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def write_probability_outputs(
    surfaces: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...] = ("geotiff", "csv"),
    include_uncertainty: bool = True,
) -> list[Path]:
    """Write all configured output formats for a set of surfaces."""
    unknown = sorted(set(formats) - set(ALLOWED_OUTPUT_FORMATS))
    if unknown:
        raise ValueError(
            f"unknown output format(s): {unknown}; "
            f"expected a subset of {list(ALLOWED_OUTPUT_FORMATS)}"
        )
    _preflight_output_request(
        surfaces,
        formats=formats,
        include_uncertainty=include_uncertainty,
    )
    written: list[Path] = []
    if "geotiff" in formats:
        written.extend(
            write_geotiff_outputs(
                surfaces, output_dir, value_col="probability"
            )
        )
        if include_uncertainty:
            for value_col in (
                "spatial_u_std",
                "probability_lo",
                "probability_hi",
            ):
                uncertainty_surfaces = {
                    name: gdf
                    for name, gdf in surfaces.items()
                    if value_col in gdf.columns
                }
                written.extend(
                    write_geotiff_outputs(
                        uncertainty_surfaces,
                        output_dir,
                        value_col=value_col,
                    )
                )
    if "parquet" in formats:
        written.extend(write_parquet_outputs(surfaces, output_dir))
    if "vtk" in formats:
        written.extend(write_vtk_outputs(surfaces, output_dir))
    if "csv" in formats:
        written.extend(write_csv_outputs(surfaces, output_dir))
    return written


# ---------------------------------------------------------------------------
# Paired Bayesian posterior probability draws
# ---------------------------------------------------------------------------


def _posterior_grid_coordinates(
    grid_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, tuple[str, ...], str | None]:
    """Return finite point coordinates and their explicit spatial metadata."""
    geometry = grid_gdf.geometry
    if geometry.isna().any() or geometry.is_empty.any():
        raise ValueError(
            "posterior draw grid contains missing or empty geometry"
        )
    if not geometry.geom_type.eq("Point").all():
        raise ValueError(
            "posterior draw grid geometry must contain only points"
        )
    has_z = geometry.has_z.to_numpy(dtype=bool)
    if has_z.any() and not has_z.all():
        raise ValueError("posterior draw grid cannot mix 2-D and 3-D points")
    coordinate_columns = ("x", "y", "z") if has_z.all() else ("x", "y")
    axes = [geometry.x.to_numpy(dtype=np.float64)]
    axes.append(geometry.y.to_numpy(dtype=np.float64))
    if has_z.all():
        axes.append(geometry.z.to_numpy(dtype=np.float64))
    coordinates = np.column_stack(axes)
    if not np.isfinite(coordinates).all():
        raise ValueError("posterior draw grid coordinates must be finite")
    crs = None if grid_gdf.crs is None else grid_gdf.crs.to_string()
    return coordinates, coordinate_columns, crs


def _validated_probability_array(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate one draw-by-cell probability array without changing precision."""
    array = np.asarray(values)
    if array.ndim != _DRAW_ARRAY_DIMENSIONS:
        raise ValueError(f"{name} must have shape (n_draws, n_cells)")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise ValueError(f"{name} must contain numeric probabilities")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return array


def _combined_draw_estimand(
    component_block: np.ndarray,
    *,
    combination_rule: str,
) -> tuple[np.ndarray, str]:
    """Evaluate the component-probability product within each draw."""
    if combination_rule != "product":
        raise ValueError(
            f"combination_rule must be 'product' (got {combination_rule!r})"
        )
    return (
        np.prod(component_block, axis=2),
        "within_draw_component_product",
    )


def _uncertainty_semantics(
    component_names: tuple[str, ...], state_metadata: Mapping[str, Any]
) -> str:
    """Derive the uncertainty estimand from every component's fitted role."""
    roles = state_metadata.get("component_roles")
    if not isinstance(roles, Mapping) or set(roles) != set(component_names):
        raise ValueError(
            "state_metadata.component_roles must map every component exactly"
        )
    allowed = {
        "joint_posterior",
        "evidence_coefficient_prior_predictive",
        "fixed_prior_predictive",
    }
    unknown = set(roles.values()) - allowed
    if unknown:
        raise ValueError(
            "state_metadata.component_roles contains unsupported roles: "
            + ", ".join(sorted(str(role) for role in unknown))
        )
    role_values = set(roles.values())
    if role_values == {"fixed_prior_predictive"}:
        raise ValueError(
            "posterior draw storage cannot represent replicated fixed prior "
            "probabilities as uncertainty draws"
        )
    if role_values == {"joint_posterior"}:
        return "paired_bayesian_posterior_probability_draws"
    if "joint_posterior" in role_values:
        return "paired_mixed_posterior_prior_predictive_probability_draws"
    return "paired_prior_predictive_probability_draws"


@dataclass(frozen=True)
class PosteriorDrawSummary:
    """Exact summaries reconstructed from persisted posterior draw blocks."""

    component_mean: np.ndarray
    component_interval: np.ndarray
    combined_mean: np.ndarray
    combined_interval: np.ndarray
    index_path: Path


@dataclass(frozen=True)
class PersistedPosteriorDrawState:
    """Hash-verified immutable state reopened from a draw namespace."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    component_names: tuple[str, ...]
    completed_draw_ranges: tuple[tuple[int, int], ...]
    state_fingerprint: str
    complete: bool
    index_path: Path | None


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON through an adjacent temporary file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def _state_array_name(name: str) -> str:
    """Validate one stable state-array identifier."""
    if (
        not isinstance(name, str)
        or not name
        or not name.replace("_", "a").isalnum()
        or name[0].isdigit()
    ):
        raise ValueError(
            "posterior state array names must be non-empty alphanumeric/underscore "
            "identifiers that do not start with a digit"
        )
    return name


def _file_record_path(root: Path, record: Mapping[str, Any]) -> Path:
    """Resolve one indexed payload without leaving its owning namespace."""
    root = Path(root).resolve()
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise TypeError("posterior draw payload path must be a string")
    relative = Path(raw_path)
    candidate = root / relative
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or any(
            (root.joinpath(*relative.parts[:index])).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        )
    ):
        raise ValueError(
            "posterior draw payload must remain inside its posterior namespace"
        )
    path = candidate.resolve()
    if root not in path.parents:
        raise ValueError(
            "posterior draw payload must remain inside its posterior namespace"
        )
    return path


def _verify_file_record(root: Path, record: Mapping[str, Any]) -> Path:
    """Fail closed when one contained payload is absent or corrupted."""
    path = _file_record_path(root, record)
    if not path.is_file():
        raise ValueError(f"posterior draw payload is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"posterior draw payload size mismatch: {path}")
    if _file_sha256(path) != record["sha256"]:
        raise ValueError(f"posterior draw payload hash mismatch: {path}")
    return path


class PosteriorDrawBlockWriter:
    """Incrementally persist one immutable posterior/scenario draw namespace.

    The writer fingerprints the exact posterior forward state before accepting
    probability blocks. Interrupted writes resume only when that fingerprint is
    identical, preventing blocks from separate fits from being mixed. Each
    accepted block is atomically published and hash-verified. Final summaries
    are exact and reconstructed with a disk-backed array, so working memory is
    bounded by the configured draw block and summary cell chunk sizes.
    """

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        grid_gdf: gpd.GeoDataFrame,
        output_dir: Path,
        *,
        component_names: tuple[str, ...],
        n_draws: int,
        block_size: int,
        seed: int,
        combination_rule: str,
        scope: str,
        state_arrays: Mapping[str, np.ndarray],
        state_metadata: Mapping[str, Any],
    ) -> None:
        if not component_names or len(set(component_names)) != len(
            component_names
        ):
            raise ValueError("component_names must be non-empty and unique")
        if any(
            not isinstance(name, str) or not name for name in component_names
        ):
            raise ValueError("component_names must contain non-empty strings")
        if (
            isinstance(n_draws, bool)
            or not isinstance(n_draws, Integral)
            or n_draws < 1
        ):
            raise ValueError("n_draws must be a positive integer")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, Integral)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must be a non-empty string")
        _combined_draw_estimand(
            np.full((1, 1, len(component_names)), 0.5),
            combination_rule=combination_rule,
        )
        if not state_arrays:
            raise ValueError(
                "state_arrays must contain immutable posterior state"
            )

        coordinates, coordinate_columns, crs = _posterior_grid_coordinates(
            grid_gdf
        )
        if len(grid_gdf) < 1:
            raise ValueError("posterior draw grid must not be empty")
        try:
            metadata_json = json.dumps(
                dict(state_metadata), sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "state_metadata must be strict JSON data"
            ) from exc

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir = self.output_dir / "posterior_draws"
        self.work_dir = self.output_dir / ".posterior_draws.incomplete"
        if self.final_dir.exists() and (
            not self.final_dir.is_dir() or self.final_dir.is_symlink()
        ):
            raise ValueError(
                "the reserved posterior_draws output path must be a directory"
            )
        if self.work_dir.exists() and (
            not self.work_dir.is_dir() or self.work_dir.is_symlink()
        ):
            raise ValueError(
                "the reserved .posterior_draws.incomplete path must be a directory"
            )

        self.component_names = tuple(component_names)
        self.n_draws = int(n_draws)
        self.n_cells = len(grid_gdf)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.combination_rule = combination_rule
        self.scope = scope
        self.coordinate_columns = coordinate_columns
        self.crs = crs
        self.state_metadata = json.loads(metadata_json)
        self.uncertainty_semantics = _uncertainty_semantics(
            self.component_names, self.state_metadata
        )
        self._complete = False
        self._block_records: list[dict[str, Any]] = []

        fingerprint = hashlib.sha256()
        fingerprint.update(metadata_json.encode("utf-8"))
        fingerprint.update(np.ascontiguousarray(coordinates).view(np.uint8))
        fingerprint.update(
            json.dumps(
                {
                    "component_names": self.component_names,
                    "n_draws": self.n_draws,
                    "block_size": self.block_size,
                    "seed": self.seed,
                    "combination_rule": self.combination_rule,
                    "scope": self.scope,
                    "coordinate_columns": self.coordinate_columns,
                    "crs": self.crs,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        validated_state: dict[str, np.ndarray] = {}
        for name in sorted(state_arrays):
            safe_name = _state_array_name(name)
            array = np.asarray(state_arrays[name])
            if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError(
                    f"state_arrays[{name!r}] must contain finite real numbers"
                )
            array = np.ascontiguousarray(array)
            fingerprint.update(safe_name.encode("utf-8"))
            fingerprint.update(array.dtype.str.encode("ascii"))
            fingerprint.update(json.dumps(array.shape).encode("ascii"))
            fingerprint.update(array.view(np.uint8))
            validated_state[safe_name] = array
        self.state_fingerprint = fingerprint.hexdigest()
        self._prior_logit_state = validated_state.get("prior_logit")

        if self.final_dir.exists():
            index_path = self.final_dir / "index.json"
            if not index_path.is_file():
                raise ValueError(
                    "existing posterior_draws directory is incomplete"
                )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if (
                index.get("state", {}).get("fingerprint")
                != self.state_fingerprint
            ):
                raise ValueError(
                    "existing posterior state fingerprint differs from this run"
                )
            self._verify_index(self.final_dir, index)
            self._block_records = list(index["blocks"])
            self._complete = True
            return

        progress_path = self.work_dir / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("state_fingerprint") != self.state_fingerprint:
                raise ValueError(
                    "incomplete posterior state fingerprint differs from this run"
                )
            for record in progress.get("blocks", []):
                _verify_file_record(self.work_dir, record)
            _verify_file_record(self.work_dir, progress["coordinates"])
            for record in progress["state_arrays"]:
                _verify_file_record(self.work_dir, record)
            self._coordinate_record = dict(progress["coordinates"])
            self._state_records = list(progress["state_arrays"])
            self._block_records = list(progress.get("blocks", []))
            return

        self.work_dir.mkdir(parents=False, exist_ok=False)
        (self.work_dir / "blocks").mkdir()
        (self.work_dir / "state").mkdir()
        coordinate_path = self.work_dir / "coordinates.npy"
        np.save(coordinate_path, coordinates, allow_pickle=False)
        self._coordinate_record = self._record_file(coordinate_path)
        state_records = []
        for name, array in validated_state.items():
            path = self.work_dir / "state" / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            state_records.append({"name": name, **self._record_file(path)})
        self._state_records = state_records
        self._write_progress()

    @property
    def completed_draw_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return verified draw ranges already present in this namespace."""
        return tuple(
            (int(record["draw_start"]), int(record["draw_stop"]))
            for record in self._block_records
        )

    @property
    def is_complete(self) -> bool:
        """Return whether a matching completed namespace already exists."""
        return self._complete

    def _record_file(self, path: Path) -> dict[str, Any]:
        root = self.final_dir if self._complete else self.work_dir
        return {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    def _write_progress(self) -> None:
        if self._complete:
            return
        _atomic_json_write(
            self.work_dir / "progress.json",
            {
                "schema_version": 2,
                "state_fingerprint": self.state_fingerprint,
                "scope": self.scope,
                "component_names": list(self.component_names),
                "n_draws": self.n_draws,
                "n_cells": self.n_cells,
                "n_components": len(self.component_names),
                "block_size": self.block_size,
                "seed": self.seed,
                "combination_rule": self.combination_rule,
                "coordinate_columns": list(self.coordinate_columns),
                "crs": self.crs,
                "coordinates": self._coordinate_record,
                "state_metadata": self.state_metadata,
                "uncertainty_semantics": self.uncertainty_semantics,
                "state_arrays": self._state_records,
                "blocks": self._block_records,
            },
        )

    def write_block(  # noqa: PLR0914
        self,
        draw_start: int,
        *,
        component_probability: np.ndarray,
        prior_logit: np.ndarray,
        evidence_logit: np.ndarray,
        spatial_logit: np.ndarray,
    ) -> None:
        """Validate and atomically persist one contiguous draw block."""
        if self._complete:
            return
        expected_start = (
            0
            if not self._block_records
            else int(self._block_records[-1]["draw_stop"])
        )
        if draw_start != expected_start:
            raise ValueError(
                f"draw blocks must be contiguous; expected start {expected_start}"
            )
        draw_stop = min(draw_start + self.block_size, self.n_draws)
        expected_shape = (
            draw_stop - draw_start,
            self.n_cells,
            len(self.component_names),
        )
        probability = np.asarray(component_probability, dtype=np.float64)
        evidence = np.asarray(evidence_logit, dtype=np.float64)
        spatial = np.asarray(spatial_logit, dtype=np.float64)
        prior = np.asarray(prior_logit, dtype=np.float64)
        if probability.shape != expected_shape:
            raise ValueError(
                f"component_probability has shape {probability.shape}; expected {expected_shape}"
            )
        if evidence.shape != expected_shape or spatial.shape != expected_shape:
            raise ValueError(
                "evidence_logit and spatial_logit must match component_probability"
            )
        if prior.shape != expected_shape[1:]:
            raise ValueError(
                f"prior_logit has shape {prior.shape}; expected {expected_shape[1:]}"
            )
        if self._prior_logit_state is not None and not np.array_equal(
            prior, self._prior_logit_state
        ):
            raise ValueError(
                "prior_logit differs from persisted prior_logit state"
            )
        for name, array in (
            ("component_probability", probability),
            ("prior_logit", prior),
            ("evidence_logit", evidence),
            ("spatial_logit", spatial),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("component_probability must lie in [0, 1]")
        eta = prior[np.newaxis, :, :] + evidence + spatial
        expected_probability = np.exp(-np.logaddexp(0.0, -eta))
        if not np.allclose(
            probability, expected_probability, rtol=1e-12, atol=1e-15
        ):
            raise ValueError(
                "component_probability does not equal the declared logit decomposition"
            )
        combined, _ = _combined_draw_estimand(
            probability, combination_rule=self.combination_rule
        )
        filename = f"blocks/block_{draw_start:08d}_{draw_stop:08d}.npz"
        final_path = self.work_dir / filename
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{final_path.stem}-", suffix=".npz", dir=final_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary_path,
                schema_version=np.asarray(2, dtype=np.int64),
                draw_start=np.asarray(draw_start, dtype=np.int64),
                draw_stop=np.asarray(draw_stop, dtype=np.int64),
                draw_id=np.arange(draw_start, draw_stop, dtype=np.int64),
                component_names=np.asarray(
                    self.component_names, dtype=np.str_
                ),
                prior_logit=prior,
                evidence_logit=evidence,
                spatial_logit=spatial,
                component_probability=probability,
                combined_probability=combined,
            )
            temporary_path.replace(final_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._block_records.append(
            {
                **self._record_file(final_path),
                "draw_start": draw_start,
                "draw_stop": draw_stop,
                "n_draws": draw_stop - draw_start,
            }
        )
        self._write_progress()

    def _verify_index(self, root: Path, index: Mapping[str, Any]) -> None:
        for record in (
            index["coordinates"],
            *index["state"]["arrays"],
            *index["blocks"],
        ):
            _verify_file_record(root, record)
        ranges = [
            (int(item["draw_start"]), int(item["draw_stop"]))
            for item in index["blocks"]
        ]
        expected = [
            (start, min(start + self.block_size, self.n_draws))
            for start in range(0, self.n_draws, self.block_size)
        ]
        if ranges != expected:
            raise ValueError(
                "posterior draw index does not contain a complete draw partition"
            )

    def _summarize(self, root: Path, ci_level: float) -> PosteriorDrawSummary:
        if not np.isfinite(ci_level) or not 0.0 < ci_level < 1.0:
            raise ValueError("ci_level must be finite and in (0, 1)")
        q = len(self.component_names)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".posterior-summary-", suffix=".dat", dir=self.output_dir
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            draws = np.memmap(
                temporary_path,
                mode="w+",
                dtype=np.float64,
                shape=(self.n_draws, self.n_cells, q + 1),
            )
            for record in self._block_records:
                with np.load(
                    root / record["path"], allow_pickle=False
                ) as payload:
                    start = int(record["draw_start"])
                    stop = int(record["draw_stop"])
                    draws[start:stop, :, :q] = payload["component_probability"]
                    draws[start:stop, :, q] = payload["combined_probability"]
            draws.flush()
            component_mean = np.empty((self.n_cells, q), dtype=np.float64)
            combined_mean = np.empty(self.n_cells, dtype=np.float64)
            component_interval = np.empty(
                (2, self.n_cells, q), dtype=np.float64
            )
            combined_interval = np.empty((2, self.n_cells), dtype=np.float64)
            tail = (1.0 - ci_level) / 2.0
            for start in range(0, self.n_cells, _SUMMARY_CELL_CHUNK_SIZE):
                stop = min(start + _SUMMARY_CELL_CHUNK_SIZE, self.n_cells)
                chunk = np.asarray(draws[:, start:stop, :])
                component_mean[start:stop] = chunk[:, :, :q].mean(axis=0)
                combined_mean[start:stop] = chunk[:, :, q].mean(axis=0)
                interval = np.quantile(chunk, [tail, 1.0 - tail], axis=0)
                component_interval[:, start:stop] = interval[:, :, :q]
                combined_interval[:, start:stop] = interval[:, :, q]
            return PosteriorDrawSummary(
                component_mean=component_mean,
                component_interval=component_interval,
                combined_mean=combined_mean,
                combined_interval=combined_interval,
                index_path=root / "index.json",
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def finalize(self, *, ci_level: float) -> PosteriorDrawSummary:
        """Verify completeness, atomically publish the namespace, and summarize."""
        if self._complete:
            return self._summarize(self.final_dir, ci_level)
        expected_ranges = [
            (start, min(start + self.block_size, self.n_draws))
            for start in range(0, self.n_draws, self.block_size)
        ]
        if list(self.completed_draw_ranges) != expected_ranges:
            raise ValueError(
                "cannot finalize before every posterior draw block is present"
            )
        combination_estimand = _combined_draw_estimand(
            np.full((1, 1, len(self.component_names)), 0.5),
            combination_rule=self.combination_rule,
        )[1]
        index = {
            "schema_version": 2,
            "scope": self.scope,
            "uncertainty_semantics": self.uncertainty_semantics,
            "cross_scenario_pairing": (
                "within_scope_only"
                if self.scope == "baseline"
                else "not_identified"
            ),
            "combination_rule": self.combination_rule,
            "combination_estimand": combination_estimand,
            "decomposition": "prior_logit + evidence_logit + spatial_logit",
            "probability_dtype": "float64",
            "coordinate_dtype": "float64",
            "component_names": list(self.component_names),
            "coordinate_columns": list(self.coordinate_columns),
            "crs": self.crs,
            "n_draws": self.n_draws,
            "n_cells": self.n_cells,
            "n_components": len(self.component_names),
            "seed": self.seed,
            "block_size": self.block_size,
            "ci_level": float(ci_level),
            "coordinates": self._coordinate_record,
            "state": {
                "fingerprint": self.state_fingerprint,
                "metadata": self.state_metadata,
                "arrays": self._state_records,
            },
            "blocks": self._block_records,
        }
        _atomic_json_write(self.work_dir / "index.json", index)
        self.work_dir.replace(self.final_dir)
        self._complete = True
        (self.final_dir / "progress.json").unlink()
        return self._summarize(self.final_dir, ci_level)


def load_posterior_draw_state(  # noqa: PLR0913, PLR0914
    output_dir: Path,
    grid_gdf: gpd.GeoDataFrame,
    *,
    expected_config_hash: str,
    expected_analysis_input_sha256: str,
    expected_implementation_sha256: str,
    expected_scope: str,
) -> PersistedPosteriorDrawState | None:
    """Reopen exact state for restart without refitting the Bayesian model."""
    output_dir = Path(output_dir)
    final_dir = output_dir / "posterior_draws"
    work_dir = output_dir / ".posterior_draws.incomplete"
    if final_dir.is_dir():
        index_path = final_dir / "index.json"
        verify_posterior_draw_bundle(index_path)
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        root = final_dir
        state_metadata = payload["state"]["metadata"]
        state_records = payload["state"]["arrays"]
        state_fingerprint = payload["state"]["fingerprint"]
        complete = True
        returned_index: Path | None = index_path
    elif work_dir.is_dir():
        progress_path = work_dir / "progress.json"
        if not progress_path.is_file():
            raise ValueError(
                "incomplete posterior draw directory lacks progress.json"
            )
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != _INCREMENTAL_DRAW_SCHEMA_VERSION:
            raise ValueError("incomplete posterior draw schema is unsupported")
        root = work_dir
        state_metadata = payload["state_metadata"]
        state_records = payload["state_arrays"]
        state_fingerprint = payload["state_fingerprint"]
        complete = False
        returned_index = None
        for record in (
            payload["coordinates"],
            *state_records,
            *payload["blocks"],
        ):
            _verify_file_record(root, record)
    else:
        return None
    if state_metadata.get("config_hash") != expected_config_hash:
        raise ValueError(
            "persisted posterior config hash differs from this run"
        )
    if (
        state_metadata.get("analysis_input_sha256")
        != expected_analysis_input_sha256
    ):
        raise ValueError(
            "persisted posterior analysis inputs differ from this run"
        )
    if (
        state_metadata.get("implementation_sha256")
        != expected_implementation_sha256
    ):
        raise ValueError(
            "persisted posterior implementation differs from this run"
        )
    if payload.get("scope") != expected_scope:
        raise ValueError("persisted posterior scope differs from this run")
    coordinates, coordinate_columns, crs = _posterior_grid_coordinates(
        grid_gdf
    )
    persisted_coordinates = np.load(
        root / payload["coordinates"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        tuple(payload["coordinate_columns"]) != coordinate_columns
        or payload["crs"] != crs
        or not np.array_equal(persisted_coordinates, coordinates)
    ):
        raise ValueError(
            "persisted posterior coordinates differ from this run"
        )
    arrays = {
        record["name"]: np.load(
            root / record["path"], mmap_mode="r", allow_pickle=False
        )
        for record in state_records
    }
    ranges = tuple(
        (int(record["draw_start"]), int(record["draw_stop"]))
        for record in payload["blocks"]
    )
    return PersistedPosteriorDrawState(
        arrays=arrays,
        metadata=state_metadata,
        component_names=tuple(payload["component_names"]),
        completed_draw_ranges=ranges,
        state_fingerprint=state_fingerprint,
        complete=complete,
        index_path=returned_index,
    )


def verify_posterior_draw_bundle(  # noqa: PLR0912, PLR0914, PLR0915
    index_path: Path,
) -> dict[str, Any]:
    """Independently verify one incremental posterior draw namespace."""
    index_path = Path(index_path)
    if not index_path.is_file():
        raise ValueError(f"posterior draw index does not exist: {index_path}")
    root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != _INCREMENTAL_DRAW_SCHEMA_VERSION:
        raise ValueError(
            "incremental posterior verification requires schema_version 2"
        )
    n_draws = int(index["n_draws"])
    n_cells = int(index["n_cells"])
    n_components = int(index["n_components"])
    component_names = tuple(index["component_names"])
    if (
        len(component_names) != n_components
        or len(set(component_names)) != n_components
    ):
        raise ValueError(
            "posterior draw index has inconsistent component names"
        )
    expected_semantics = _uncertainty_semantics(
        component_names, index["state"]["metadata"]
    )
    if index.get("uncertainty_semantics") != expected_semantics:
        raise ValueError(
            "posterior draw uncertainty semantics are inconsistent with "
            "component inference roles"
        )
    for record in (
        index["coordinates"],
        *index["state"]["arrays"],
        *index["blocks"],
    ):
        _verify_file_record(root, record)
    coordinates = np.load(
        root / index["coordinates"]["path"], mmap_mode="r", allow_pickle=False
    )
    if coordinates.shape != (n_cells, len(index["coordinate_columns"])):
        raise ValueError("posterior coordinate payload shape is inconsistent")
    if not np.isfinite(coordinates).all():
        raise ValueError("posterior coordinates contain nonfinite values")
    for record in index["state"]["arrays"]:
        state_array = np.load(
            root / record["path"], mmap_mode="r", allow_pickle=False
        )
        if (
            state_array.dtype.kind not in "iuf"
            or not np.isfinite(state_array).all()
        ):
            raise ValueError(
                f"posterior state array is invalid: {record['path']}"
            )

    expected_start = 0
    for record in index["blocks"]:
        draw_start = int(record["draw_start"])
        draw_stop = int(record["draw_stop"])
        if (
            draw_start != expected_start
            or not draw_start < draw_stop <= n_draws
        ):
            raise ValueError(
                "posterior draw blocks do not form a contiguous partition"
            )
        with np.load(root / record["path"], allow_pickle=False) as payload:
            expected_shape = (draw_stop - draw_start, n_cells, n_components)
            probability = np.asarray(payload["component_probability"])
            prior = np.asarray(payload["prior_logit"])
            evidence = np.asarray(payload["evidence_logit"])
            spatial = np.asarray(payload["spatial_logit"])
            combined = np.asarray(payload["combined_probability"])
            if (
                probability.shape != expected_shape
                or evidence.shape != expected_shape
                or spatial.shape != expected_shape
                or prior.shape != expected_shape[1:]
                or combined.shape != expected_shape[:2]
            ):
                raise ValueError(
                    "posterior block payload shapes are inconsistent"
                )
            if not np.array_equal(
                payload["draw_id"],
                np.arange(draw_start, draw_stop, dtype=np.int64),
            ) or not np.array_equal(
                payload["component_names"], component_names
            ):
                raise ValueError(
                    "posterior block identifiers are inconsistent"
                )
            arrays = (probability, prior, evidence, spatial, combined)
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError("posterior block contains nonfinite values")
            if np.any((probability < 0.0) | (probability > 1.0)) or np.any(
                (combined < 0.0) | (combined > 1.0)
            ):
                raise ValueError(
                    "posterior block probabilities lie outside [0, 1]"
                )
            eta = prior[np.newaxis, :, :] + evidence + spatial
            reconstructed = np.exp(-np.logaddexp(0.0, -eta))
            if not np.allclose(
                probability, reconstructed, rtol=1e-12, atol=1e-15
            ):
                raise ValueError(
                    "posterior block logit decomposition is inconsistent"
                )
            expected_combined, _ = _combined_draw_estimand(
                probability, combination_rule=index["combination_rule"]
            )
            if not np.allclose(
                combined, expected_combined, rtol=1e-12, atol=1e-15
            ):
                raise ValueError(
                    "posterior block component combination is inconsistent"
                )
        expected_start = draw_stop
    if expected_start != n_draws:
        raise ValueError(
            "posterior draw blocks do not cover every requested draw"
        )
    return {
        "schema_version": 2,
        "scope": index["scope"],
        "n_draws": n_draws,
        "n_cells": n_cells,
        "n_components": n_components,
        "n_blocks": len(index["blocks"]),
        "uncertainty_semantics": expected_semantics,
        "hashes_verified": True,
        "draw_partition_verified": True,
        "probability_bounds_verified": True,
        "decomposition_verified": True,
        "combination_verified": True,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(config: ProbabilisticConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in _RUNTIME_SOURCE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"no runtime source files found beneath {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_tracks_runtime_root(
    git_executable: str, source_root: Path, runtime_root: Path
) -> bool:
    """Return whether the enclosing repository tracks the runtime tree."""
    try:
        relative = runtime_root.resolve().relative_to(source_root)
        result = subprocess.run(  # noqa: S603
            [
                git_executable,
                "-C",
                str(source_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative.as_posix(),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _git_source_provenance(root: Path) -> dict[str, str | bool | None]:
    """Return the live source revision and worktree state when Git is present."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return {"revision": None, "clean": None}
    try:
        top_level = subprocess.run(  # noqa: S603 -- executable is resolved above
            [git_executable, "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"revision": None, "clean": None}
    if top_level.returncode != 0:
        return {"revision": None, "clean": None}
    source_root = Path(top_level.stdout.strip()).resolve()
    if not _git_tracks_runtime_root(git_executable, source_root, root):
        return {"revision": None, "clean": None}
    try:
        revision = subprocess.run(  # noqa: S603
            [git_executable, "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git_executable, "-C", str(source_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"revision": None, "clean": None}
    return {"revision": revision, "clean": not bool(status.strip())}


def _release_version_at_head(root: Path) -> str | None:
    """Return the unique semantic release version tagging ``HEAD``."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [
                git_executable,
                "-C",
                str(root),
                "tag",
                "--points-at",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    versions = {
        match.group(1)
        for tag in result.stdout.splitlines()
        if (match := re.fullmatch(r"v(\d+\.\d+\.\d+)", tag.strip()))
    }
    if len(versions) > 1:
        raise RuntimeError(
            "geoPFA HEAD has multiple semantic release tags; source version "
            "provenance is ambiguous"
        )
    return next(iter(versions), None)


def _require_version_matches_source(
    version: str,
    source: Mapping[str, str | bool | None],
    root: Path,
) -> None:
    """Reject stale generated version metadata for a clean Git checkout."""
    revision = source.get("revision")
    if source.get("clean") is not True or revision is None:
        return
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise RuntimeError("geoPFA Git source revision is malformed")

    release_version = _release_version_at_head(root)
    if release_version is not None:
        if version != release_version:
            raise RuntimeError(
                "geoPFA package version does not match the semantic release "
                "tag on the clean Git source"
            )
        return

    revision_lower = revision.lower()
    version_revision_tokens = (
        token[1:]
        for token in re.split(r"[.+-]", version.lower())
        if re.fullmatch(r"g[0-9a-f]{7,40}", token)
    )
    if not any(
        revision_lower.startswith(token) for token in version_revision_tokens
    ):
        raise RuntimeError(
            "geoPFA package version does not identify clean Git source "
            f"revision {revision}; refresh the installed package from this "
            "checkout before running"
        )


def _latticekrigx_provenance() -> dict[str, Any]:
    spec = importlib.util.find_spec("latticekrigx")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "latticekrigx must be importable to write a probabilistic manifest"
        )
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    return {
        "package": "latticekrigx",
        "version": importlib.metadata.version("latticekrigx"),
        "implementation_sha256": _runtime_tree_hash(root),
        "source": _git_source_provenance(root),
    }


def _probabilistic_implementation_hash() -> str:
    """Hash geoPFA and LatticeKrigX runtime sources used by the workflow."""
    digest = hashlib.sha256()
    geopfa_hash = _runtime_tree_hash(Path(__file__).parents[1])
    latticekrigx_hash = _latticekrigx_provenance()["implementation_sha256"]
    for package, source_hash in (
        ("geopfa", geopfa_hash),
        ("latticekrigx", latticekrigx_hash),
    ):
        digest.update(package.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _shapefile_sidecar(source: Path, suffix: str) -> Path | None:
    """Resolve one Shapefile sidecar, including case-only suffix variants."""
    if not source.parent.is_dir():
        return None
    matches = sorted(
        path
        for path in source.parent.iterdir()
        if path.is_file()
        and path.stem.casefold() == source.stem.casefold()
        and path.suffix.casefold() == suffix
    )
    if len(matches) > 1:
        raise ValueError(
            f"Shapefile input has ambiguous {suffix} sidecars: {source}"
        )
    return matches[0].resolve() if matches else None


def _manifest_input_bundle(name: str, source: str | Path) -> dict[str, Path]:
    """Expand one logical input into every file used by its data format."""
    path = Path(source).resolve()
    if path.suffix.casefold() != ".shp":
        return {name: path}

    bundle = {name: path}
    for suffix in _SHAPEFILE_REQUIRED_SIDECARS:
        sidecar = _shapefile_sidecar(path, suffix)
        bundle[f"{name}{suffix}"] = (
            path.with_suffix(suffix).resolve() if sidecar is None else sidecar
        )
    for suffix in _SHAPEFILE_OPTIONAL_SIDECARS:
        sidecar = _shapefile_sidecar(path, suffix)
        if sidecar is not None:
            bundle[f"{name}{suffix}"] = sidecar
    return bundle


def _manifest_inputs(
    config: ProbabilisticConfig,
    input_artifacts: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    configured: dict[str, str | Path] = {}
    if config.labels.source is not None:
        configured["labels.source"] = config.labels.source
    for component, alpha in sorted(config.alpha.items()):
        if alpha.thermal_raster is not None:
            configured[f"alpha.{component}.thermal_raster"] = (
                alpha.thermal_raster
            )
        if alpha.uncertainty_raster is not None:
            configured[f"alpha.{component}.uncertainty_raster"] = (
                alpha.uncertainty_raster
            )
    if config.site_selection.candidate_source is not None:
        configured["site_selection.candidate_source"] = (
            config.site_selection.candidate_source
        )
    supplied = dict(input_artifacts or {})
    overlap = configured.keys() & supplied.keys()
    if overlap:
        raise ValueError(
            "input_artifacts must not override configured inputs: "
            + ", ".join(sorted(overlap))
        )
    configured.update(supplied)
    expanded: dict[str, Path] = {}
    for name, source in sorted(configured.items()):
        for member_name, member_path in _manifest_input_bundle(
            name, source
        ).items():
            if member_name in expanded:
                raise ValueError(
                    "input artifact names collide after Shapefile bundle "
                    f"expansion: {member_name}"
                )
            expanded[member_name] = member_path
    return expanded


def _manifest_input_records(
    config: ProbabilisticConfig,
    input_artifacts: Mapping[str, str | Path] | None,
) -> list[dict[str, Any]]:
    """Resolve and fingerprint every file consumed by a manifest run."""
    records: list[dict[str, Any]] = []
    for name, path in sorted(
        _manifest_inputs(config, input_artifacts).items()
    ):
        if not path.is_file():
            raise FileNotFoundError(
                "manifest input artifact does not exist or is not a file: "
                f"{path}"
            )
        records.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return records


def _verify_manifest_inputs(
    records: Any, expected: Mapping[str, Path] | None
) -> int:
    if not isinstance(records, list):
        raise TypeError("run manifest inputs must be a list")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("name"), str
        ):
            raise TypeError("run manifest contains an invalid input record")
        name = record["name"]
        if name in names:
            raise ValueError(f"run manifest repeats input name {name!r}")
        names.add(name)
        path = Path(str(record.get("path", ""))).resolve()
        if expected is not None and expected.get(name) != path:
            raise ValueError(
                f"run manifest input {name!r} has an unexpected path"
            )
        if not path.is_file():
            raise ValueError(f"run manifest input is missing: {path}")
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"run manifest input size differs: {path}")
        if record.get("sha256") != _file_sha256(path):
            raise ValueError(f"run manifest input digest differs: {path}")
    if expected is not None and names != set(expected):
        raise ValueError(
            "run manifest input set differs from the effective run inputs"
        )
    return len(names)


def _verify_manifest_outputs(
    output_dir: Path,
    records: Any,
    *,
    allow_resumable_posterior: bool,
    ignored_root_names: frozenset[str] = frozenset(),
) -> int:
    if not isinstance(records, list):
        raise TypeError("run manifest files must be a list")
    recorded_paths: set[Path] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("path"), str
        ):
            raise TypeError("run manifest contains an invalid output record")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "run manifest output paths must remain inside output_dir"
            )
        if (
            allow_resumable_posterior
            and relative.parts[0] == "posterior_draws"
        ):
            continue
        path = (output_dir / relative).resolve()
        if output_dir not in path.parents or path in recorded_paths:
            raise ValueError(
                "run manifest contains an invalid or repeated output path"
            )
        recorded_paths.add(path)
        if not path.is_file():
            raise ValueError(f"run manifest output is missing: {path}")
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"run manifest output size differs: {path}")
        if record.get("sha256") != _file_sha256(path):
            raise ValueError(f"run manifest output digest differs: {path}")
    manifest_path = output_dir / "manifest.json"
    present_paths = {
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not (path.name in ignored_root_names and path.parent == output_dir)
        and not (
            allow_resumable_posterior
            and ".posterior_draws.incomplete" in path.parts
        )
    }
    if recorded_paths != present_paths:
        raise ValueError("run manifest has unlisted or missing output files")
    return len(recorded_paths)


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    """Read one strict JSON object with a user-facing failure context."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must contain a JSON object")
    return payload


def _resume_record_paths(
    root: Path,
    records: Any,
    *,
    context: str,
) -> set[Path]:
    """Resolve the exact file set declared by posterior metadata."""
    if not isinstance(records, list):
        raise TypeError(f"{context} files must be a list")
    root = root.resolve()
    paths: set[Path] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("path"), str
        ):
            raise TypeError(f"{context} contains an invalid file record")
        path = _file_record_path(root, record)
        if not path.is_file():
            raise ValueError(f"{context} file is missing: {path}")
        if path in paths:
            raise ValueError(f"{context} contains a repeated file path")
        paths.add(path)
    return paths


def _verify_incomplete_draw_prefix(payload: Mapping[str, Any]) -> None:
    """Require incomplete draw blocks to be one contiguous prefix."""
    expected_start = 0
    n_draws = int(payload["n_draws"])
    for record in payload.get("blocks", []):
        draw_start = int(record["draw_start"])
        draw_stop = int(record["draw_stop"])
        if (
            draw_start != expected_start
            or not draw_start < draw_stop <= n_draws
        ):
            raise ValueError(
                "incomplete posterior blocks are not a contiguous prefix"
            )
        expected_start = draw_stop


def _posterior_resume_files(
    namespace: Path,
    *,
    complete: bool,
    expected_scope: str,
    expected_config_hash: str,
    expected_implementation_sha256: str,
) -> set[Path]:
    """Verify one resumable posterior namespace and return its files."""
    metadata_path = namespace / ("index.json" if complete else "progress.json")
    if not metadata_path.is_file():
        raise ValueError(
            "resumable posterior namespace lacks "
            f"{metadata_path.name}: {namespace}"
        )
    payload = _read_json_mapping(
        metadata_path, context="resumable posterior metadata"
    )
    if payload.get("schema_version") != _INCREMENTAL_DRAW_SCHEMA_VERSION:
        raise ValueError("resumable posterior schema is unsupported")
    state_metadata = (
        payload.get("state", {}).get("metadata")
        if complete and isinstance(payload.get("state"), dict)
        else payload.get("state_metadata")
    )
    if not isinstance(state_metadata, dict):
        raise TypeError("resumable posterior state metadata is invalid")
    if payload.get("scope") != expected_scope:
        raise ValueError("resumable posterior scope differs from this run")
    if state_metadata.get("config_hash") != expected_config_hash:
        raise ValueError("resumable posterior config differs from this scope")
    if (
        state_metadata.get("implementation_sha256")
        != expected_implementation_sha256
    ):
        raise ValueError(
            "resumable posterior implementation differs from this run"
        )
    state = payload.get("state") if complete else None
    state_records = (
        state.get("arrays", [])
        if isinstance(state, dict)
        else payload.get("state_arrays", [])
    )
    indexed_records = [
        payload.get("coordinates"),
        *state_records,
        *payload.get("blocks", []),
    ]
    if any(not isinstance(record, dict) for record in indexed_records):
        raise TypeError("resumable posterior file records are invalid")
    indexed_paths = _resume_record_paths(
        namespace,
        indexed_records,
        context="resumable posterior",
    )
    if not complete:
        _verify_incomplete_draw_prefix(payload)
    expected_files = indexed_paths | {metadata_path.resolve()}
    present_files = {
        path.resolve() for path in namespace.rglob("*") if path.is_file()
    }
    if present_files != expected_files:
        raise ValueError("resumable posterior namespace has unindexed files")
    if any(path.is_symlink() for path in namespace.rglob("*")):
        raise ValueError("resumable posterior namespace contains a symlink")
    return expected_files


def _run_scope_layout(
    config: ProbabilisticConfig,
) -> tuple[tuple[str, str], ...]:
    """Return the ordered output namespace owned by each run scope."""
    return (
        ("baseline", "."),
        *(
            (f"scenario:{scenario.name}", f"scenarios/{scenario.name}")
            for scenario in config.scenarios
        ),
    )


def _validate_run_resume_marker(  # noqa: PLR0913
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    config: ProbabilisticConfig,
    input_artifacts: Mapping[str, str | Path] | None,
    implementation_sha256: str,
    expected_scopes: list[dict[str, str]],
    validate_namespace: bool,
    verify_input_files: bool = True,
) -> None:
    """Validate the identity and optional files of one interrupted run."""
    if set(payload) != {
        "schema_version",
        "config_hash",
        "implementation_sha256",
        "inputs",
        "scopes",
    }:
        raise ValueError("run resume marker has an unsupported schema")
    if payload.get("schema_version") != _RUN_RESUME_SCHEMA_VERSION:
        raise ValueError("run resume marker schema_version is unsupported")
    if payload.get("config_hash") != _config_hash(config):
        raise ValueError(
            "run resume marker was made by a different effective config"
        )
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "run resume marker was made by a different probabilistic "
            "implementation"
        )
    if verify_input_files:
        _verify_manifest_inputs(
            payload.get("inputs"), _manifest_inputs(config, input_artifacts)
        )
    if payload.get("scopes") != expected_scopes:
        raise ValueError("run resume scope plan differs from this run")
    if not validate_namespace:
        return

    allowed_files = {(output_dir / _RUN_RESUME_FILENAME).resolve()}
    for scope in expected_scopes:
        relative = Path(scope["output_dir"])
        scope_dir = (
            output_dir if relative == Path() else output_dir / relative
        ).resolve()
        complete_dir = scope_dir / "posterior_draws"
        incomplete_dir = scope_dir / ".posterior_draws.incomplete"
        present = [
            path for path in (complete_dir, incomplete_dir) if path.exists()
        ]
        if len(present) > 1 or any(not path.is_dir() for path in present):
            raise ValueError("run resume posterior namespace is ambiguous")
        if present:
            allowed_files.update(
                _posterior_resume_files(
                    present[0],
                    complete=present[0] == complete_dir,
                    expected_scope=scope["name"],
                    expected_config_hash=scope["config_hash"],
                    expected_implementation_sha256=implementation_sha256,
                )
            )

    if any(path.is_symlink() for path in output_dir.rglob("*")):
        raise ValueError("run resume namespace contains a symlink")
    present_files = {
        path.resolve() for path in output_dir.rglob("*") if path.is_file()
    }
    if present_files != allowed_files:
        raise ValueError("run resume namespace contains untracked artifacts")


def verify_manifest(
    output_dir: Path,
    *,
    config: ProbabilisticConfig | None = None,
    input_artifacts: Mapping[str, str | Path] | None = None,
    require_current_implementation: bool = False,
    allow_resumable_posterior: bool = False,
) -> dict[str, int | bool]:
    """Verify a run manifest and every applicable input/output digest.

    ``allow_resumable_posterior`` excludes only the reserved incomplete draw
    transaction. Its state and block digests are verified independently by
    :class:`PosteriorDrawBlockWriter` before computation resumes.
    """
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"run manifest is unreadable: {manifest_path}"
        ) from exc
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported run manifest schema_version")

    config_verified = False
    if config is not None:
        if manifest.get("config_hash") != _config_hash(config):
            raise ValueError(
                "run manifest was made by a different effective config"
            )
        config_verified = True

    implementation_verified = False
    if require_current_implementation:
        if (
            manifest.get("implementation_sha256")
            != _probabilistic_implementation_hash()
        ):
            raise ValueError(
                "run manifest was made by a different probabilistic implementation"
            )
        implementation_verified = True

    expected_inputs = (
        _manifest_inputs(config, input_artifacts)
        if config is not None
        else None
    )
    n_inputs = _verify_manifest_inputs(manifest.get("inputs"), expected_inputs)
    n_files = _verify_manifest_outputs(
        output_dir,
        manifest.get("files"),
        allow_resumable_posterior=allow_resumable_posterior,
    )
    return {
        "schema_version": 1,
        "files_verified": n_files,
        "inputs_verified": n_inputs,
        "config_verified": config_verified,
        "implementation_verified": implementation_verified,
    }


def validate_output_namespace(
    output_dir: Path,
    config: ProbabilisticConfig,
    *,
    input_artifacts: Mapping[str, str | Path] | None = None,
) -> None:
    """Reject output directories that could mix incompatible run artifacts.

    A directory may be fresh or contain only a resumable incremental posterior
    workspace. A completed manifest is immutable, including when it matches
    the effective configuration and implementation. Any completed or invalid
    namespace requires a new output directory (or an explicit user-controlled
    archive/removal step).
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    entries = sorted(output_dir.iterdir())
    if not entries:
        return
    incomplete = output_dir / ".posterior_draws.incomplete"
    if entries == [incomplete] and incomplete.is_dir():
        if not config.outputs.posterior_draw_blocks:
            raise GEOPFAValueError(
                "output_dir contains an incomplete posterior run but posterior "
                "draw blocks are disabled; choose a fresh output_dir"
            )
        return
    if incomplete.exists():
        if not incomplete.is_dir() or not config.outputs.posterior_draw_blocks:
            raise GEOPFAValueError(
                "output_dir contains an invalid incomplete posterior namespace; "
                "choose a fresh output_dir"
            )
        if (output_dir / "posterior_draws").exists():
            raise GEOPFAValueError(
                "output_dir contains both complete and incomplete posterior "
                "namespaces; choose a fresh output_dir"
            )

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GEOPFAValueError(
            "output_dir is not a fresh or resumable run namespace and has no "
            "manifest; choose a fresh output_dir or explicitly archive/remove "
            "the existing artifacts"
        )
    try:
        verify_manifest(
            output_dir,
            config=config,
            input_artifacts=input_artifacts,
            require_current_implementation=True,
            allow_resumable_posterior=incomplete.is_dir(),
        )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if "different effective config" in message:
            message = "output_dir contains artifacts from a different effective config"
        elif "different probabilistic implementation" in message:
            message = (
                "output_dir contains artifacts from a different probabilistic "
                "implementation"
            )
        raise GEOPFAValueError(
            f"{message}; choose a fresh output_dir"
        ) from exc
    raise GEOPFAValueError(
        "output_dir contains a completed manifested run; choose a fresh "
        "output_dir"
    )


def _run_resume_scopes(
    output_dir: Path,
    config: ProbabilisticConfig,
    scope_configs: Mapping[str, ProbabilisticConfig],
) -> list[dict[str, str]]:
    """Bind each streamed posterior scope to its exact effective config."""
    layout = _run_scope_layout(config)
    if tuple(scope_configs) != tuple(name for name, _ in layout):
        raise ValueError("run resume scope configs differ from the run plan")
    scopes: list[dict[str, str]] = []
    for name, relative in layout:
        scope_config = scope_configs[name]
        expected_output_dir = (
            output_dir
            if relative == "."
            else (output_dir / relative).resolve()
        )
        if scope_config.output_dir.resolve() != expected_output_dir:
            raise ValueError(
                f"run resume scope {name!r} has an unexpected output_dir"
            )
        scopes.append(
            {
                "name": name,
                "output_dir": relative,
                "config_hash": _config_hash(scope_config),
            }
        )
    return scopes


def _verify_manifest_before_marker_removal(
    output_dir: Path,
    *,
    config: ProbabilisticConfig,
    input_artifacts: Mapping[str, str | Path] | None,
    implementation_sha256: str,
) -> None:
    """Verify a completed manifest while its owned resume marker remains."""
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json_mapping(manifest_path, context="run manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported run manifest schema_version")
    if manifest.get("config_hash") != _config_hash(config):
        raise ValueError(
            "run manifest was made by a different effective config"
        )
    if manifest.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "run manifest was made by a different probabilistic implementation"
        )
    _verify_manifest_inputs(
        manifest.get("inputs"), _manifest_inputs(config, input_artifacts)
    )
    _verify_manifest_outputs(
        output_dir,
        manifest.get("files"),
        allow_resumable_posterior=False,
        ignored_root_names=frozenset({_RUN_RESUME_FILENAME}),
    )


def _accept_new_manifest(
    output_dir: Path,
    marker: Mapping[str, Any],
    *,
    config: ProbabilisticConfig,
    implementation_sha256: str,
) -> None:
    """Accept the manifest just produced from the marker-bound run."""
    manifest = _read_json_mapping(
        output_dir / "manifest.json", context="run manifest"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("config_hash") != _config_hash(config)
        or manifest.get("implementation_sha256") != implementation_sha256
        or manifest.get("inputs") != marker.get("inputs")
    ):
        raise ValueError(
            "completed manifest differs from its run resume marker"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or any(
        not isinstance(record, dict)
        or record.get("path") == _RUN_RESUME_FILENAME
        for record in files
    ):
        raise ValueError("completed manifest contains an invalid file list")
    if _probabilistic_implementation_hash() != implementation_sha256:
        raise RuntimeError(
            "geoPFA or LatticeKrigX runtime source changed during run"
        )


@dataclass
class _StreamedRunResumeGuard:
    """Own one marker that permits exact streamed GBLK resumption."""

    output_dir: Path
    config: ProbabilisticConfig
    input_artifacts: Mapping[str, str | Path] | None
    implementation_sha256: str
    scopes: list[dict[str, str]]
    managed: bool

    def finish(self, manifest_path: Path) -> None:
        """Remove the marker only after the completed manifest verifies."""
        if not self.managed:
            return
        expected_manifest = self.output_dir / "manifest.json"
        if manifest_path.resolve() != expected_manifest.resolve():
            raise ValueError(
                "run resume guard received an unexpected manifest"
            )
        marker_path = self.output_dir / _RUN_RESUME_FILENAME
        payload = _read_json_mapping(marker_path, context="run resume marker")
        _validate_run_resume_marker(
            self.output_dir,
            payload,
            config=self.config,
            input_artifacts=self.input_artifacts,
            implementation_sha256=self.implementation_sha256,
            expected_scopes=self.scopes,
            validate_namespace=False,
            verify_input_files=False,
        )
        _accept_new_manifest(
            self.output_dir,
            payload,
            config=self.config,
            implementation_sha256=self.implementation_sha256,
        )
        marker_path.unlink()
        self.managed = False


def _begin_streamed_run_resume(
    output_dir: Path,
    config: ProbabilisticConfig,
    *,
    scope_configs: Mapping[str, ProbabilisticConfig],
    input_artifacts: Mapping[str, str | Path] | None,
    expected_implementation_sha256: str,
) -> _StreamedRunResumeGuard:
    """Create or verify the marker for a streamed GBLK run."""
    output_dir = Path(output_dir).resolve()
    implementation_sha256 = _probabilistic_implementation_hash()
    if implementation_sha256 != expected_implementation_sha256:
        raise RuntimeError(
            "geoPFA or LatticeKrigX runtime source changed before run startup"
        )
    scopes = _run_resume_scopes(output_dir, config, scope_configs)
    marker_path = output_dir / _RUN_RESUME_FILENAME
    manifest_path = output_dir / "manifest.json"

    if marker_path.is_file():
        completed_manifest = False
        try:
            payload = _read_json_mapping(
                marker_path, context="run resume marker"
            )
            _validate_run_resume_marker(
                output_dir,
                config=config,
                payload=payload,
                input_artifacts=input_artifacts,
                implementation_sha256=implementation_sha256,
                expected_scopes=scopes,
                validate_namespace=not manifest_path.is_file(),
            )
            if manifest_path.is_file():
                _verify_manifest_before_marker_removal(
                    output_dir,
                    config=config,
                    input_artifacts=input_artifacts,
                    implementation_sha256=implementation_sha256,
                )
                marker_path.unlink()
                managed = False
                completed_manifest = True
            else:
                managed = True
        except (KeyError, TypeError, ValueError) as exc:
            message = str(exc)
            if "different effective config" in message:
                message = (
                    "output_dir contains artifacts from a different "
                    "effective config"
                )
            elif "different probabilistic implementation" in message:
                message = (
                    "output_dir contains artifacts from a different "
                    "probabilistic implementation"
                )
            raise GEOPFAValueError(
                f"{message}; choose a fresh output_dir"
            ) from exc
        if completed_manifest:
            raise GEOPFAValueError(
                "output_dir contains a completed manifested run; choose a "
                "fresh output_dir"
            )
        return _StreamedRunResumeGuard(
            output_dir,
            config,
            input_artifacts,
            implementation_sha256,
            scopes,
            managed,
        )

    validate_output_namespace(
        output_dir, config, input_artifacts=input_artifacts
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _RUN_RESUME_SCHEMA_VERSION,
        "config_hash": _config_hash(config),
        "implementation_sha256": implementation_sha256,
        "inputs": _manifest_input_records(config, input_artifacts),
        "scopes": scopes,
    }
    _atomic_json_write(marker_path, payload)
    _validate_run_resume_marker(
        output_dir,
        payload,
        config=config,
        input_artifacts=input_artifacts,
        implementation_sha256=implementation_sha256,
        expected_scopes=scopes,
        validate_namespace=True,
    )
    return _StreamedRunResumeGuard(
        output_dir,
        config,
        input_artifacts,
        implementation_sha256,
        scopes,
        True,
    )


def write_manifest(
    output_dir: Path,
    *,
    config: ProbabilisticConfig,
    run_id: str | None = None,
    input_artifacts: Mapping[str, str | Path] | None = None,
    expected_implementation_sha256: str | None = None,
) -> Path:
    """Write ``manifest.json`` listing every file in ``output_dir`` with sha256.

    The manifest captures the effective config, producer version, configured
    and caller-supplied input hashes, output hashes, and a run id.
    """
    implementation_sha256 = _probabilistic_implementation_hash()
    if (
        expected_implementation_sha256 is not None
        and implementation_sha256 != expected_implementation_sha256
    ):
        raise RuntimeError(
            "geoPFA or LatticeKrigX runtime source changed during run; "
            "discard the incomplete artifacts and rerun from a stable checkout"
        )
    source_root = Path(__file__).parents[1]
    source = _git_source_provenance(source_root)
    from geopfa import __version__  # noqa: PLC0415

    _require_version_matches_source(__version__, source, source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = [
        {
            "path": str(path.relative_to(output_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and not (
            path.parent == output_dir and path.name == _RUN_RESUME_FILENAME
        )
    ]
    inputs = _manifest_input_records(config, input_artifacts)
    payload = {
        "schema_version": 1,
        "run_id": run_id or datetime.now(UTC).isoformat(),
        "producer": {
            "package": "geoPFA",
            "version": __version__,
            "source": source,
            "dependencies": [_latticekrigx_provenance()],
        },
        "implementation_sha256": implementation_sha256,
        "config_hash": _config_hash(config),
        "config": config.to_dict(),
        "inputs": inputs,
        "files": files,
    }
    if _probabilistic_implementation_hash() != implementation_sha256:
        raise RuntimeError(
            "geoPFA or LatticeKrigX runtime source changed while the run "
            "manifest was being assembled"
        )
    manifest_path = output_dir / "manifest.json"
    _atomic_json_write(manifest_path, payload)
    return manifest_path


__all__ = [
    "PersistedPosteriorDrawState",
    "PosteriorDrawBlockWriter",
    "PosteriorDrawSummary",
    "load_posterior_draw_state",
    "validate_output_namespace",
    "verify_manifest",
    "verify_posterior_draw_bundle",
    "write_csv_outputs",
    "write_geotiff_outputs",
    "write_manifest",
    "write_parquet_outputs",
    "write_probability_outputs",
    "write_vtk_outputs",
]
