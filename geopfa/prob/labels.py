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
        gdf[label_col] = _coerce_binary_labels(
            gdf[label_col], label_column=label_col
        )


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


def load_labels(  # noqa: PLR0913
    cfg: LabelsConfig,
    *,
    source_crs: str | None = None,
    target_crs: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
) -> LoadedLabels:
    """Load labelled wells from ``cfg.source`` using existing geoPFA readers.

    Parameters
    ----------
    cfg
        :class:`~geopfa.prob.config.LabelsConfig` block from the user's PFA
        config.
    source_crs
        CRS of the CSV coordinate columns (required for CSV sources without
        a CRS column).
    target_crs
        Optional CRS to reproject into. ``None`` keeps the file's CRS.
    x_col, y_col, z_col
        Column names for CSV sources. Defaults: ``"longitude"``,
        ``"latitude"``, no ``z`` column.

    Returns
    -------
    LoadedLabels
        Container holding the loaded GeoDataFrame and the config.
    """
    path = Path(cfg.source)
    suffix = path.suffix.lower()
    if suffix in _VECTOR_EXTS:
        if cfg.layer is not None:
            gdf = gpd.read_file(path, layer=cfg.layer)
        else:
            gdf = GeospatialDataReaders.read_shapefile(path)
    elif suffix in _CSV_EXTS:
        if source_crs is None:
            raise ValueError(
                "loading CSV labels requires source_crs (the CRS of the "
                "coordinate columns)",
            )
        gdf = GeospatialDataReaders.read_csv(
            str(path),
            source_crs,
            x_col=x_col or "longitude",
            y_col=y_col or "latitude",
            z_col=z_col,
        )
    else:
        raise ValueError(
            f"unsupported labels source format: {suffix!r}; expected one of "
            f"{sorted(_VECTOR_EXTS | _CSV_EXTS)}",
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
    series = _coerce_binary_labels(
        loaded.gdf[label_col], label_column=label_col
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
