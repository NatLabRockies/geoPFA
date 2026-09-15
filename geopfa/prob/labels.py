"""Config-driven labelled-well loading for the probabilistic method.

Wraps :class:`geopfa.io.data_readers.GeospatialDataReaders` so the runner can
load labelled wells from any format the existing geoPFA readers understand
(GeoPackage, shapefile, CSV with 2D or 3D coordinates) via the
``probabilistic.labels`` block of the config.

The output is a :class:`LoadedLabels` container holding the validated
GeoDataFrame plus the originating config. Per-component subsets (filtered to
finite labels for that component) are produced by :func:`component_labels`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geopfa.io.data_readers import GeospatialDataReaders

from .config import LabelsConfig

_VECTOR_EXTS: frozenset[str] = frozenset(
    {".shp", ".gpkg", ".geojson", ".json"}
)
_CSV_EXTS: frozenset[str] = frozenset({".csv", ".txt"})


@dataclass(frozen=True)
class LoadedLabels:
    """The labelled-well GeoDataFrame plus the config it was loaded from."""

    gdf: gpd.GeoDataFrame
    config: LabelsConfig


def _coerce_binary_labels(
    series: pd.Series, *, label_column: str
) -> pd.Series:
    """Return numeric 0/1 labels while preserving genuinely missing values."""

    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    observed = series.notna().to_numpy(dtype=bool)
    nonnumeric = observed & np.isnan(values)
    if np.any(nonnumeric):
        raise ValueError(
            f"label column {label_column!r} contains "
            f"{int(np.sum(nonnumeric))} nonnumeric observed value(s)"
        )
    nonfinite = observed & ~np.isfinite(values)
    if np.any(nonfinite):
        raise ValueError(
            f"label column {label_column!r} contains "
            f"{int(np.sum(nonfinite))} non-finite observed value(s)"
        )
    finite = np.isfinite(values)
    invalid = finite & ~np.isin(values, (0.0, 1.0))
    if np.any(invalid):
        examples = sorted(set(values[invalid].tolist()))[:5]
        raise ValueError(
            f"label column {label_column!r} must contain binary 0/1 values "
            f"or missing values; found {examples}"
        )
    return pd.Series(values, index=series.index, name=series.name, dtype=float)


def _coerce_continuous_labels(
    series: pd.Series, *, label_column: str
) -> pd.Series:
    """Return finite continuous responses while preserving missing values."""
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    observed = series.notna().to_numpy(dtype=bool)
    nonnumeric = observed & np.isnan(values)
    if np.any(nonnumeric):
        raise ValueError(
            f"label column {label_column!r} contains "
            f"{int(np.sum(nonnumeric))} nonnumeric observed value(s)"
        )
    nonfinite = observed & ~np.isfinite(values)
    if np.any(nonfinite):
        raise ValueError(
            f"label column {label_column!r} contains "
            f"{int(np.sum(nonfinite))} non-finite observed value(s)"
        )
    return pd.Series(values, index=series.index, name=series.name, dtype=float)


def _coerce_component_labels(
    series: pd.Series, *, label_column: str, family: str
) -> pd.Series:
    """Validate one response column against its configured likelihood."""
    if family == "bernoulli":
        return _coerce_binary_labels(series, label_column=label_column)
    if family == "gaussian":
        return _coerce_continuous_labels(series, label_column=label_column)
    raise ValueError(f"unsupported observation family {family!r}")


def _validate_columns(gdf: gpd.GeoDataFrame, cfg: LabelsConfig) -> None:
    if cfg.id_col not in gdf.columns:
        raise KeyError(
            f"labelled-well file missing required id column: {cfg.id_col!r}",
        )
    for component, label_col in cfg.label_columns.items():
        if label_col not in gdf.columns:
            raise KeyError(
                f"labelled-well file missing label column {label_col!r} "
                f"for component {component!r}",
            )
        gdf[label_col] = _coerce_component_labels(
            gdf[label_col],
            label_column=label_col,
            family=cfg.observation_model_for(component).family,
        )
        observed = np.isfinite(gdf[label_col].to_numpy(dtype=float))
        for field_name, columns, require_positive in (
            (
                "prior response mean",
                cfg.prior_response_mean_columns,
                False,
            ),
            ("prior response SD", cfg.prior_response_sd_columns, True),
        ):
            if component not in columns:
                continue
            column = columns[component]
            if column not in gdf.columns:
                raise KeyError(
                    f"labelled-well file missing {field_name} column "
                    f"{column!r} for component {component!r}"
                )
            numeric = pd.to_numeric(gdf[column], errors="coerce")
            values = numeric.to_numpy(dtype=float, na_value=np.nan)
            if not np.all(np.isfinite(values[observed])):
                raise ValueError(
                    f"{field_name} column {column!r} must contain finite "
                    "numeric values wherever the component response is observed"
                )
            if require_positive and np.any(values[observed] <= 0.0):
                raise ValueError(
                    f"{field_name} column {column!r} must be positive "
                    "wherever the component response is observed"
                )
            gdf[column] = numeric.astype(float)
        weight_column = cfg.observation_weight_columns.get(component)
        if weight_column is not None:
            if weight_column not in gdf.columns:
                raise KeyError(
                    "labelled-well file missing observation weight column "
                    f"{weight_column!r} for component {component!r}"
                )
            numeric = pd.to_numeric(gdf[weight_column], errors="coerce")
            values = numeric.to_numpy(dtype=float, na_value=np.nan)
            if not np.all(np.isfinite(values[observed])) or np.any(
                values[observed] <= 0.0
            ):
                raise ValueError(
                    f"observation weight column {weight_column!r} must "
                    "contain positive finite numeric values wherever the "
                    "component response is observed"
                )
            gdf[weight_column] = numeric.astype(float)


def _maybe_reproject(
    gdf: gpd.GeoDataFrame, target_crs: str | None
) -> gpd.GeoDataFrame:
    if target_crs is None:
        return gdf
    if gdf.crs is None:
        raise ValueError(
            "cannot reproject labelled wells without a declared source CRS"
        )
    if str(gdf.crs) == str(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def _apply_declared_source_crs(
    gdf: gpd.GeoDataFrame, source_crs: str | None
) -> gpd.GeoDataFrame:
    """Apply a missing declared CRS or reject conflict with embedded metadata."""
    if source_crs is None:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(source_crs)
    if gdf.crs != source_crs:
        raise ValueError(
            "labels.source_crs conflicts with the CRS embedded in the label source"
        )
    return gdf


def _validated_depth(gdf: gpd.GeoDataFrame, depth_col: str) -> np.ndarray:
    """Return a finite, nonnegative, positive-down scientific depth."""
    if depth_col not in gdf.columns:
        raise KeyError(
            f"labelled-well file missing depth column {depth_col!r}"
        )
    numeric = pd.to_numeric(gdf[depth_col], errors="coerce")
    depth = numeric.to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isfinite(depth)):
        raise ValueError(
            f"depth column {depth_col!r} must contain finite numeric values"
        )
    if np.any(depth < 0.0):
        raise ValueError(
            f"depth column {depth_col!r} must use nonnegative positive-down depth"
        )
    gdf[depth_col] = numeric.astype(float)
    return depth


def load_labels(
    cfg: LabelsConfig,
    *,
    target_crs: str | None = None,
) -> LoadedLabels:
    """Load labelled wells from ``cfg.source`` using existing geoPFA readers.

    Parameters
    ----------
    cfg : LabelsConfig
        Label-loading block from the user's PFA configuration.
    target_crs : str or None
        Optional CRS to reproject into. ``None`` keeps the file's CRS.

    Returns
    -------
    LoadedLabels
        Container holding the loaded GeoDataFrame and the config.

    Notes
    -----
    CSV source CRS and coordinate-column semantics come exclusively from
    ``cfg``. ``z_col`` is a Cartesian model coordinate. ``depth_col`` is a
    separate nonnegative, positive-down scientific depth. When only depth is
    declared, model geometry uses ``z = -depth``; when both are declared,
    ``z_col`` controls geometry and ``depth_col`` is retained separately.
    """
    path = Path(cfg.source)
    suffix = path.suffix.lower()
    if suffix in _VECTOR_EXTS:
        if any(
            value is not None for value in (cfg.x_col, cfg.y_col, cfg.z_col)
        ):
            raise ValueError(
                "labels.x_col, y_col, and z_col apply only to CSV sources; "
                "vector-source geometry is authoritative"
            )
        if cfg.layer is not None:
            gdf = gpd.read_file(path, layer=cfg.layer)
        else:
            gdf = GeospatialDataReaders.read_shapefile(path)
        gdf = _apply_declared_source_crs(gdf, cfg.source_crs)
    elif suffix in _CSV_EXTS:
        if cfg.source_crs is None:
            raise ValueError(
                "loading CSV labels requires labels.source_crs (the CRS of the "
                "coordinate columns)",
            )
        if cfg.x_col is None or cfg.y_col is None:
            raise ValueError(
                "loading CSV labels requires labels.x_col and labels.y_col"
            )
        gdf = GeospatialDataReaders.read_csv(
            str(path),
            cfg.source_crs,
            x_col=cfg.x_col,
            y_col=cfg.y_col,
            z_col=cfg.z_col,
        )
    else:
        raise ValueError(
            f"unsupported labels source format: {suffix!r}; expected one of "
            f"{sorted(_VECTOR_EXTS | _CSV_EXTS)}",
        )

    if cfg.depth_col is not None:
        depth = _validated_depth(gdf, cfg.depth_col)
        if cfg.z_col is None and suffix in _CSV_EXTS:
            gdf = gdf.set_geometry(
                gpd.points_from_xy(
                    gdf[cfg.x_col],
                    gdf[cfg.y_col],
                    z=-depth,
                )
            )

    _validate_columns(gdf, cfg)
    gdf = _maybe_reproject(gdf, target_crs)
    return LoadedLabels(gdf=gdf, config=cfg)


def available_components(cfg: LabelsConfig) -> list[str]:
    """Return the component names configured in ``cfg.label_columns``."""
    return list(cfg.label_columns.keys())


def component_labels(loaded: LoadedLabels, component: str) -> gpd.GeoDataFrame:
    """Subset ``loaded.gdf`` to rows with a finite label for ``component``.

    Useful when label coverage differs per component (e.g. wells labelled for
    heat but not barrier).

    Raises
    ------
    KeyError
        If ``component`` is not in ``LabelsConfig.label_columns``.
    """
    if component not in loaded.config.label_columns:
        raise KeyError(
            f"component {component!r} is not configured in "
            f"labels.label_columns: {list(loaded.config.label_columns)}",
        )
    label_col = loaded.config.label_columns[component]
    series = _coerce_component_labels(
        loaded.gdf[label_col],
        label_column=label_col,
        family=loaded.config.observation_model_for(component).family,
    )
    mask = np.isfinite(series.to_numpy(dtype=float))
    subset = loaded.gdf.loc[mask].copy()
    subset[label_col] = series.loc[mask]
    return subset


__all__ = [
    "LoadedLabels",
    "available_components",
    "component_labels",
    "load_labels",
]
