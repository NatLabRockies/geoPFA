"""Coverage for generic raster sampling and output-writer edge cases."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from geopfa.prob.data import sample_evidence_at_wells
from geopfa.prob.io import (
    _gdf_to_raster,
    write_geotiff_outputs,
    write_probability_outputs,
    write_vtk_outputs,
)


def test_gdf_to_raster_rejects_irregular_grid() -> None:
    rng = np.random.default_rng(1)
    gdf = gpd.GeoDataFrame(
        {"probability": rng.uniform(0, 1, 50)},
        geometry=gpd.points_from_xy(
            rng.uniform(0, 100, 50), rng.uniform(0, 100, 50)
        ),
        crs="EPSG:32610",
    )
    with pytest.raises(ValueError, match="complete rectilinear"):
        _gdf_to_raster(gdf, value_col="probability")


def test_write_geotiff_outputs_skips_empty_gdf(tmp_path: Path) -> None:
    empty = gpd.GeoDataFrame(columns=["geometry", "probability"])
    written = write_geotiff_outputs({"comp": empty}, tmp_path / "out")
    assert written == []


def test_write_geotiff_outputs_rejects_missing_value_column(
    tmp_path: Path,
) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    gdf = gpd.GeoDataFrame(
        {"other_col": np.arange(16)},
        geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="probability"):
        write_geotiff_outputs(
            {"comp": gdf}, tmp_path / "out", value_col="probability"
        )


def test_sample_evidence_transforms_points_to_exact_raster_crs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasterio.crs import CRS

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([-118.0], [39.0]), crs="EPSG:4326"
    )
    requested: list[object] = []
    original = gpd.GeoDataFrame.to_crs

    def record_to_crs(self, crs=None, epsg=None, inplace=False):
        requested.append(crs if crs is not None else f"EPSG:{epsg}")
        return original(self, crs=crs, epsg=epsg, inplace=inplace)

    class Dataset:
        crs = CRS.from_epsg(26911)
        nodata = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def sample(_coords, indexes=1):
            assert indexes == 1
            yield np.array([1.0])

    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", record_to_crs)
    monkeypatch.setattr(
        "geopfa.prob.data.rasterio.open", lambda _path: Dataset()
    )

    sample_evidence_at_wells(points, {"test": "dummy.tif"})

    assert CRS.from_user_input(requested[-1]).to_epsg() == 26911


def test_write_vtk_outputs_rejects_2d_surfaces(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"probability": [0.5, 0.6]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="VTK output requires 3-D"):
        write_vtk_outputs({"comp": gdf}, output_dir)

    assert not output_dir.exists()


def test_write_vtk_outputs_skips_empty_gdf(tmp_path: Path) -> None:
    empty = gpd.GeoDataFrame(columns=["geometry"])
    assert write_vtk_outputs({"comp": empty}, tmp_path / "out") == []


def _make_prob_gdf(n: int = 16) -> gpd.GeoDataFrame:
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    return gpd.GeoDataFrame(
        {"probability": np.linspace(0.1, 0.9, n)},
        geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()),
        crs="EPSG:4326",
    )


@pytest.mark.parametrize("output_format", ["geotiff", "parquet"])
def test_write_probability_outputs_routes_format(
    tmp_path: Path, output_format: str
) -> None:
    written = write_probability_outputs(
        {"comp": _make_prob_gdf()},
        tmp_path / "out",
        formats=(output_format,),
    )
    expected_suffix = ".tif" if output_format == "geotiff" else ".parquet"
    assert any(path.suffix == expected_suffix for path in written)


def test_write_probability_outputs_controls_uncertainty(
    tmp_path: Path,
) -> None:
    gdf = _make_prob_gdf()
    gdf["spatial_u_std"] = 0.05

    enabled = write_probability_outputs(
        {"comp": gdf},
        tmp_path / "enabled",
        formats=("geotiff",),
        include_uncertainty=True,
    )
    disabled = write_probability_outputs(
        {"comp": gdf},
        tmp_path / "disabled",
        formats=("geotiff",),
        include_uncertainty=False,
    )

    assert any("spatial_u_std" in str(path) for path in enabled)
    assert not any("spatial_u_std" in str(path) for path in disabled)
