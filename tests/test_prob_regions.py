"""Tests for :mod:`geopfa.prob.regions` — multi-region label splitting."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from geopfa.prob.labels import LoadedLabels
from geopfa.prob.regions import (
    RegionLabels,
    check_region_label_coverage,
    split_by_region,
)


def _make_loaded_labels(n: int = 10, with_region: bool = True) -> LoadedLabels:
    from geopfa.prob.config import LabelsConfig

    rng = np.random.default_rng(42)
    geom = [
        Point(rng.uniform(-120, -110), rng.uniform(35, 45)) for _ in range(n)
    ]
    data: dict = {
        "geometry": geom,
        "heat_label": [1, 0] * (n // 2),
    }
    if with_region:
        data["region"] = ["A"] * (n // 2) + ["B"] * (n // 2)
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    cfg = LabelsConfig(
        source="dummy.gpkg",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    return LoadedLabels(gdf=gdf, config=cfg)


def test_split_by_region_returns_one_entry_per_region() -> None:
    loaded = _make_loaded_labels(n=10)
    result = split_by_region(loaded, group_by="region")
    assert len(result) == 2
    names = {r.region_name for r in result}
    assert names == {"A", "B"}


def test_split_by_region_subsets_are_disjoint_and_complete() -> None:
    loaded = _make_loaded_labels(n=10)
    result = split_by_region(loaded, group_by="region")
    total = sum(len(r.gdf) for r in result)
    assert total == len(loaded.gdf)


def test_split_by_region_result_is_sorted_by_name() -> None:
    loaded = _make_loaded_labels(n=10)
    result = split_by_region(loaded, group_by="region")
    names = [r.region_name for r in result]
    assert names == sorted(names)


def test_split_by_region_missing_column_raises_key_error() -> None:
    loaded = _make_loaded_labels(n=6)
    with pytest.raises(KeyError, match="group_by column"):
        split_by_region(loaded, group_by="nonexistent_col")


def test_check_region_label_coverage_sufficient_returns_true() -> None:
    loaded = _make_loaded_labels(n=10)
    regions = split_by_region(loaded, group_by="region")
    coverage = check_region_label_coverage(
        regions, label_column="heat_label", min_wells=1
    )
    assert all(coverage.values())


def test_check_region_label_coverage_insufficient_warns_and_returns_false() -> (
    None
):
    loaded = _make_loaded_labels(n=10)
    regions = split_by_region(loaded, group_by="region")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coverage = check_region_label_coverage(
            regions, label_column="heat_label", min_wells=100
        )
    assert not any(coverage.values())
    assert len(w) == 2  # one warning per region


def test_region_labels_is_frozen_dataclass() -> None:
    gdf = gpd.GeoDataFrame({"geometry": []})
    rl = RegionLabels(region_name="test", gdf=gdf)
    with pytest.raises((AttributeError, TypeError)):
        rl.region_name = "other"  # type: ignore[misc]
