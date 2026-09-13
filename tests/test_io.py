from types import SimpleNamespace

import geopandas
import pandas as pd
import pytest

from geopfa.io import GeospatialDataWriters


def test_write_shapefile(tmp_path):
    """Minimum test to write a shapefile."""
    filename = tmp_path / "empty.gpkg"

    df = pd.DataFrame(
        {
            "City": ["Golden"],
            "State": ["CO"],
            "Latitude": [39.755],
            "Longitude": [-105.221],
        }
    )
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326",
    )
    GeospatialDataWriters.write_shapefile(
        gdf, filename, target_crs="EPSG:4326"
    )

    assert filename.exists()


def test_write_csv(tmp_path):
    """Minimum test to write a CSV."""
    filename = tmp_path / "empty.gpkg"

    df = pd.DataFrame(
        {
            "City": ["Golden"],
            "State": ["CO"],
            "Latitude": [39.755],
            "Longitude": [-105.221],
        }
    )
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326",
    )
    GeospatialDataWriters.write_csv(gdf, filename, target_crs="EPSG:4326")

    assert filename.exists()


def test_probabilistic_export_defaults_to_input_crs(tmp_path):
    grid = geopandas.GeoDataFrame(
        {
            "probability": [0.25],
            "geometry": geopandas.points_from_xy([1], [2]),
        },
        crs="EPSG:32611",
    )
    result = SimpleNamespace(
        components={"heat": SimpleNamespace(probability=grid)},
        combined=geopandas.GeoDataFrame(),
        calibrated_components={},
    )

    GeospatialDataWriters.export_probabilistic_results(result, tmp_path)

    assert (tmp_path / "heat_probability.csv").exists()


def test_probabilistic_export_rejects_unsupported_format(tmp_path):
    result = SimpleNamespace(
        components={},
        combined=geopandas.GeoDataFrame(),
        calibrated_components={},
    )

    with pytest.raises(ValueError, match="fmt"):
        GeospatialDataWriters.export_probabilistic_results(
            result, tmp_path / "unused", fmt="geojson"
        )

    assert not (tmp_path / "unused").exists()
