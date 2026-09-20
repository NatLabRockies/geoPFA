"""Tests for generic probabilistic raster sampling."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geopfa.prob.data import load_processed_pfa, sample_evidence_at_wells


def test_sample_evidence_at_wells_samples_expected_values(
    tmp_path: Path,
) -> None:
    raster_path = tmp_path / "feature.tif"
    values = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype=np.float32)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)
    points = gpd.GeoDataFrame(
        geometry=[Point(0.5, 1.5), Point(1.5, 0.5)],
        crs="EPSG:4326",
    )

    sampled = sample_evidence_at_wells(points, {"feature": raster_path})

    assert sampled["feature"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(sampled["feature"].iloc[1])


def test_sample_evidence_requires_crs_and_at_least_one_raster() -> None:
    point = gpd.GeoDataFrame(geometry=[Point(0.0, 0.0)], crs=None)
    with pytest.raises(ValueError, match="valid CRS"):
        sample_evidence_at_wells(point, {"feature": "unused.tif"})

    point = point.set_crs("EPSG:4326")
    with pytest.raises(ValueError, match="cannot be empty"):
        sample_evidence_at_wells(point, {})


def test_load_processed_pfa_returns_complete_file_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"criteria": {}}')
    layer = tmp_path / "data/geologic/heat/thermal_processed.csv"
    layer.parent.mkdir(parents=True)
    layer.write_text("geometry,value\nPOINT (0 0),1\n")
    parsed = {"criteria": {}}
    monkeypatch.setattr(
        "geopfa.prob.data.safe_json_load", lambda _path: parsed
    )
    monkeypatch.setattr(
        "geopfa.prob.data.GeospatialDataReaders.gather_processed_data",
        lambda root, pfa, crs, validate, strict: {
            "root": root,
            "pfa": pfa,
            "crs": crs,
            "validate": validate,
            "strict": strict,
        },
    )

    pfa, artifacts = load_processed_pfa(
        config_path, tmp_path / "data", crs="EPSG:32611"
    )

    assert pfa["root"] == (tmp_path / "data").resolve()
    assert pfa["pfa"] is parsed
    assert pfa["crs"] == "EPSG:32611"
    assert pfa["validate"] is True and pfa["strict"] is True
    assert artifacts == {
        "processed_config": config_path.resolve(),
        "processed_data:geologic/heat/thermal_processed.csv": layer.resolve(),
    }
