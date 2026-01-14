import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from geopfa.extrapolation import prepare_slice


def test_prepare_slice_extracts_xy_from_geometry():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(1, 2), Point(3, 4)],
            "value": [10, 20],
        }
    )

    out = prepare_slice(gdf, value_col="value")

    assert list(out.columns) == ["x", "y", "value"]
    assert np.allclose(out["x"], [1, 3])
    assert np.allclose(out["y"], [2, 4])
    assert np.allclose(out["value"], [10, 20])


def test_prepare_slice_keeps_xy_columns():
    gdf = gpd.GeoDataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
            "geometry": [Point(1, 3), Point(2, 4)],
            "value": [5, 6],
        }
    )

    out = prepare_slice(gdf, value_col="value")
    assert np.allclose(out["x"], [1, 2])
    assert np.allclose(out["y"], [3, 4])


def test_prepare_slice_z_filtering():
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(1, 1), Point(2, 2), Point(3, 3)],
            "z": [10.0, 10.001, 20.0],
            "value": [1, 2, 3],
        }
    )

    out = prepare_slice(gdf, value_col="value", z_value=10.0, z_tol=0.01)
    assert len(out) == 2
    assert np.allclose(out["value"], [1, 2])


def test_prepare_slice_error_no_geometry():
    gdf = gpd.GeoDataFrame({"value": [1, 2]})
    try:
        prepare_slice(gdf, value_col="value")
        assert False, "Expected ValueError"
    except ValueError:
        pass
