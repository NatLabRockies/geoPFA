"""Tests for the config-driven labels loader."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geopfa.prob.config import LabelsConfig, ObservationModelConfig
from geopfa.prob.labels import (
    LoadedLabels,
    _maybe_reproject,
    available_components,
    component_labels,
    load_labels,
)


def test_reprojection_requires_declared_source_crs() -> None:
    wells = _make_wells_gdf().set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="source CRS"):
        _maybe_reproject(wells, "EPSG:4326")


def _make_wells_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "well_id": [f"W{i:03d}" for i in range(6)],
            "longitude": [
                550_000.0,
                555_000.0,
                560_000.0,
                565_000.0,
                570_000.0,
                575_000.0,
            ],
            "latitude": [
                4_350_000.0,
                4_352_000.0,
                4_354_000.0,
                4_356_000.0,
                4_358_000.0,
                4_360_000.0,
            ],
            "depth_m": [np.nan] * 6,
            "heat_label": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "reservoir_label": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "barrier_label": [np.nan] * 6,
            "label_source": ["proxy"] * 6,
            "label_quality": ["good"] * 6,
            "geometry": [
                Point(550_000.0, 4_350_000.0),
                Point(555_000.0, 4_352_000.0),
                Point(560_000.0, 4_354_000.0),
                Point(565_000.0, 4_356_000.0),
                Point(570_000.0, 4_358_000.0),
                Point(575_000.0, 4_360_000.0),
            ],
        },
        crs="EPSG:32611",
    )


# ---------------------------------------------------------------------------
# GeoPackage loading
# ---------------------------------------------------------------------------


def test_load_labels_from_gpkg(tmp_path: Path) -> None:
    wells = _make_wells_gdf()
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")

    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "heat_label", "reservoir": "reservoir_label"},
        layer="wells",
    )
    loaded = load_labels(cfg, target_crs="EPSG:32611")
    assert isinstance(loaded, LoadedLabels)
    assert isinstance(loaded.gdf, gpd.GeoDataFrame)
    assert len(loaded.gdf) == 6
    assert loaded.config is cfg
    assert "heat_label" in loaded.gdf.columns
    assert "reservoir_label" in loaded.gdf.columns


def test_load_labels_reprojects_to_target_crs(tmp_path: Path) -> None:
    wells = _make_wells_gdf()
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")

    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "heat_label"},
        layer="wells",
    )
    loaded = load_labels(cfg, target_crs="EPSG:4326")
    assert loaded.gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# Shapefile loading
# ---------------------------------------------------------------------------


def test_load_labels_from_shapefile(tmp_path: Path) -> None:
    wells = _make_wells_gdf()
    shp_path = tmp_path / "wells.shp"
    wells.to_file(shp_path)

    cfg = LabelsConfig(
        source=str(shp_path),
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    loaded = load_labels(cfg)
    assert len(loaded.gdf) == 6
    assert "heat_label" in loaded.gdf.columns


# ---------------------------------------------------------------------------
# CSV loading (2D xy)
# ---------------------------------------------------------------------------


def test_load_labels_from_csv_2d(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "well_id": ["A", "B", "C", "D"],
            "longitude": [550_000.0, 560_000.0, 570_000.0, 580_000.0],
            "latitude": [4_350_000.0, 4_355_000.0, 4_360_000.0, 4_365_000.0],
            "heat_label": [1, 0, 1, 0],
        }
    )
    csv_path = tmp_path / "wells.csv"
    df.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="longitude",
        y_col="latitude",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    loaded = load_labels(cfg)
    assert len(loaded.gdf) == 4
    assert isinstance(loaded.gdf, gpd.GeoDataFrame)
    assert loaded.gdf.crs.to_epsg() == 32611


def test_load_labels_csv_with_xy_columns(tmp_path: Path) -> None:
    """CSV with non-default 'x'/'y' column names should still load via config."""
    df = pd.DataFrame(
        {
            "well_id": ["A", "B"],
            "easting": [100.0, 200.0],
            "northing": [400.0, 500.0],
            "heat_label": [1, 0],
        }
    )
    csv_path = tmp_path / "wells.csv"
    df.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="easting",
        y_col="northing",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    loaded = load_labels(cfg)
    assert loaded.gdf.geometry.iloc[0].x == pytest.approx(100.0)
    assert loaded.gdf.geometry.iloc[0].y == pytest.approx(400.0)


def test_csv_loading_requires_source_crs_in_config(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A"],
            "x_m": [100.0],
            "y_m": [400.0],
            "heat_label": [1],
        }
    )
    csv_path = tmp_path / "wells.csv"
    frame.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        x_col="x_m",
        y_col="y_m",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )

    with pytest.raises(ValueError, match="labels.source_crs"):
        load_labels(cfg)


def test_csv_loading_requires_coordinate_columns_in_config(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "wells.csv"
    pd.DataFrame(
        {
            "well_id": ["A"],
            "longitude": [100.0],
            "latitude": [400.0],
            "heat_label": [1],
        }
    ).to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )

    with pytest.raises(ValueError, match="labels.x_col.*labels.y_col"):
        load_labels(cfg)


# ---------------------------------------------------------------------------
# CSV loading (3D xyz)
# ---------------------------------------------------------------------------


def test_load_labels_from_csv_3d(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "well_id": ["A", "B"],
            "longitude": [100.0, 200.0],
            "latitude": [400.0, 500.0],
            "depth_m": [-1000.0, -2000.0],
            "heat_label": [1, 0],
        }
    )
    csv_path = tmp_path / "wells.csv"
    df.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="longitude",
        y_col="latitude",
        z_col="depth_m",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    loaded = load_labels(cfg)
    assert loaded.gdf.geometry.iloc[0].has_z
    assert loaded.gdf.geometry.iloc[0].z == pytest.approx(-1000.0)


def test_csv_depth_column_is_positive_down_and_converted_to_z(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A", "B"],
            "x_m": [100.0, 200.0],
            "y_m": [400.0, 500.0],
            "depth_m": [1_000.0, 2_000.0],
            "temperature_c": [150.0, 225.0],
        }
    )
    csv_path = tmp_path / "wells.csv"
    frame.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="x_m",
        y_col="y_m",
        depth_col="depth_m",
        id_col="well_id",
        label_columns={"heat": "temperature_c"},
        observation_models={
            "heat": ObservationModelConfig(
                family="gaussian", response_scale=50.0
            )
        },
    )

    loaded = load_labels(cfg)

    assert loaded.gdf.geometry.iloc[0].has_z
    assert loaded.gdf.geometry.iloc[0].z == pytest.approx(-1_000.0)
    assert loaded.gdf["depth_m"].tolist() == [1_000.0, 2_000.0]


def test_csv_z_column_controls_geometry_when_depth_is_also_declared(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A", "B"],
            "x_m": [100.0, 200.0],
            "y_m": [400.0, 500.0],
            "elevation_m": [1_500.0, 1_450.0],
            "depth_m": [1_000.0, 2_000.0],
            "heat_label": [1, 0],
        }
    )
    csv_path = tmp_path / "wells.csv"
    frame.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="x_m",
        y_col="y_m",
        z_col="elevation_m",
        depth_col="depth_m",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )

    loaded = load_labels(cfg)

    assert loaded.gdf.geometry.iloc[0].z == pytest.approx(1_500.0)
    assert loaded.gdf["depth_m"].tolist() == [1_000.0, 2_000.0]


def test_csv_depth_column_rejects_negative_depth(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A", "B"],
            "x_m": [100.0, 200.0],
            "y_m": [400.0, 500.0],
            "depth_m": [1_000.0, -2_000.0],
            "heat_label": [1, 0],
        }
    )
    csv_path = tmp_path / "wells.csv"
    frame.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="x_m",
        y_col="y_m",
        depth_col="depth_m",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )

    with pytest.raises(ValueError, match="nonnegative positive-down"):
        load_labels(cfg)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_load_labels_missing_label_column_raises(tmp_path: Path) -> None:
    wells = _make_wells_gdf().drop(columns=["heat_label"])
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")

    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "heat_label"},
        layer="wells",
    )
    with pytest.raises(KeyError, match="heat_label"):
        load_labels(cfg, target_crs="EPSG:32611")


def test_load_labels_missing_id_column_raises(tmp_path: Path) -> None:
    wells = _make_wells_gdf().drop(columns=["well_id"])
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")

    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "heat_label"},
        layer="wells",
    )
    with pytest.raises(KeyError, match="well_id"):
        load_labels(cfg, target_crs="EPSG:32611")


def test_load_labels_unknown_format_raises(tmp_path: Path) -> None:
    cfg = LabelsConfig(
        source=str(tmp_path / "wells.unknown"),
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    with pytest.raises(ValueError, match="format"):
        load_labels(cfg)


def test_load_labels_rejects_fractional_binary_label(tmp_path: Path) -> None:
    wells = _make_wells_gdf()
    wells.loc[2, "heat_label"] = 0.5
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")
    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "heat_label"},
        layer="wells",
    )
    with pytest.raises(ValueError, match="binary 0/1"):
        load_labels(cfg)


def test_load_labels_preserves_continuous_gaussian_response(
    tmp_path: Path,
) -> None:
    wells = _make_wells_gdf()
    wells["temperature_c"] = [145.0, 182.5, np.nan, 230.0, 275.5, 310.0]
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")
    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "temperature_c"},
        observation_models={
            "heat": ObservationModelConfig(
                family="gaussian", response_scale=50.0
            )
        },
        layer="wells",
    )

    loaded = load_labels(cfg)
    subset = component_labels(loaded, "heat")

    np.testing.assert_allclose(
        subset["temperature_c"].to_numpy(),
        [145.0, 182.5, 230.0, 275.5, 310.0],
    )


def test_load_labels_rejects_nonfinite_gaussian_response(
    tmp_path: Path,
) -> None:
    wells = _make_wells_gdf()
    wells["temperature_c"] = [145.0, np.inf, 175.0, 230.0, 275.5, 310.0]
    gpkg_path = tmp_path / "wells.gpkg"
    wells.to_file(gpkg_path, layer="wells", driver="GPKG")
    cfg = LabelsConfig(
        source=str(gpkg_path),
        id_col="well_id",
        label_columns={"heat": "temperature_c"},
        observation_models={
            "heat": ObservationModelConfig(
                family="gaussian", response_scale=50.0
            )
        },
        layer="wells",
    )

    with pytest.raises(ValueError, match="non-finite"):
        load_labels(cfg)


def test_load_labels_rejects_nonnumeric_observed_label(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A", "B", "C"],
            "longitude": [100.0, 200.0, 300.0],
            "latitude": [400.0, 500.0, 600.0],
            "heat_label": ["1", "bad", "0"],
        }
    )
    csv_path = tmp_path / "wells.csv"
    frame.to_csv(csv_path, index=False)
    cfg = LabelsConfig(
        source=str(csv_path),
        source_crs="EPSG:32611",
        x_col="longitude",
        y_col="latitude",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    with pytest.raises(ValueError, match="nonnumeric"):
        load_labels(cfg)


# ---------------------------------------------------------------------------
# Component-level helpers
# ---------------------------------------------------------------------------


def test_available_components_lists_configured_label_components() -> None:
    cfg = LabelsConfig(
        source="x",
        id_col="well_id",
        label_columns={"heat": "heat_label", "reservoir": "reservoir_label"},
    )
    assert available_components(cfg) == ["heat", "reservoir"]


def test_component_labels_drops_nans_and_returns_subset() -> None:
    wells = _make_wells_gdf()
    cfg = LabelsConfig(
        source="x",
        id_col="well_id",
        label_columns={"heat": "heat_label", "barrier": "barrier_label"},
    )
    loaded = LoadedLabels(gdf=wells, config=cfg)

    # heat_label has 6 finite values
    heat_subset = component_labels(loaded, "heat")
    assert len(heat_subset) == 6
    assert "heat_label" in heat_subset.columns

    # barrier_label is all-NaN → empty
    barrier_subset = component_labels(loaded, "barrier")
    assert len(barrier_subset) == 0


def test_component_labels_unknown_component_raises() -> None:
    wells = _make_wells_gdf()
    cfg = LabelsConfig(
        source="x",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    loaded = LoadedLabels(gdf=wells, config=cfg)
    with pytest.raises(KeyError, match="not configured"):
        component_labels(loaded, "reservoir")


def test_component_labels_rejects_invalid_direct_container() -> None:
    wells = _make_wells_gdf()
    wells.loc[0, "heat_label"] = 0.25
    cfg = LabelsConfig(
        source="x",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
    )
    loaded = LoadedLabels(gdf=wells, config=cfg)
    with pytest.raises(ValueError, match="binary 0/1"):
        component_labels(loaded, "heat")
