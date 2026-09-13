"""Multi-region label handling utilities.

These standalone helpers support explicit region-stratified analyses. They do
not activate a second inference backend or silently pool regions. This module
provides helpers to:

* split a :class:`~geopfa.prob.labels.LoadedLabels` into per-region slices,
* report whether each region meets a declared labelled-well threshold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import geopandas as gpd
from .labels import LoadedLabels


@dataclass(frozen=True)
class RegionLabels:
    """Labels for one geographic region.

    Attributes
    ----------
    region_name
        Human-readable region identifier (value from the ``group_by``
        column).
    gdf
        Subset of the full wells GeoDataFrame for this region.
    """

    region_name: str
    gdf: gpd.GeoDataFrame


def split_by_region(
    loaded: LoadedLabels,
    *,
    group_by: str,
) -> list[RegionLabels]:
    """Split ``loaded.gdf`` into per-region slices.

    Parameters
    ----------
    loaded
        Full labelled-wells container.
    group_by
        Column name on ``loaded.gdf`` whose unique values define regions.

    Returns
    -------
    list[RegionLabels]
        One entry per unique region, sorted by region name.

    Raises
    ------
    KeyError
        If ``group_by`` is not a column on the wells GeoDataFrame.
    """
    if group_by not in loaded.gdf.columns:
        raise KeyError(
            f"group_by column {group_by!r} not found on wells GeoDataFrame; "
            f"available columns: {list(loaded.gdf.columns)}",
        )
    regions = sorted(loaded.gdf[group_by].dropna().unique().tolist())
    result: list[RegionLabels] = []
    for reg in regions:
        subset = loaded.gdf[loaded.gdf[group_by] == reg].copy()
        result.append(RegionLabels(region_name=str(reg), gdf=subset))
    return result


def check_region_label_coverage(
    region_labels: list[RegionLabels],
    *,
    label_column: str,
    min_wells: int = 4,
) -> dict[str, bool]:
    """Return a dict mapping region name → whether it has enough labelled wells.

    This helper only reports coverage. It does not select an inference path.
    """
    coverage: dict[str, bool] = {}
    for rl in region_labels:
        valid = rl.gdf[label_column].dropna()
        n_pos = int((valid == 1).sum())
        coverage[rl.region_name] = n_pos >= min_wells
        if not coverage[rl.region_name]:
            warnings.warn(
                f"region {rl.region_name!r} has only {n_pos} positive labels "
                f"for component column {label_column!r} (<{min_wells}); "
                "the requested regional support threshold is not met",
                UserWarning,
                stacklevel=2,
            )
    return coverage


__all__ = [
    "RegionLabels",
    "check_region_label_coverage",
    "split_by_region",
]
