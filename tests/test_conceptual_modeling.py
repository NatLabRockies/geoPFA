import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from geopfa.conceptual_modeling import ConceptualModeling
from tests.fixtures.gdf_builders import make_point_z_gdf

# PyVista plotting requires vtkRenderingMatplotlib, which is absent on some
# platforms (e.g. Windows CI). Detect once and skip rendering tests there.
try:
    import pyvista.plotting

    _pv_plotting_available = True
except Exception:
    _pv_plotting_available = False

_needs_pv_plotting = pytest.mark.skipif(
    not _pv_plotting_available,
    reason="vtkRenderingMatplotlib not available on this platform",
)


# ---------------------------------------------------------------------------
# plot_isosurface
# ---------------------------------------------------------------------------


@_needs_pv_plotting
def test_plot_isosurface_smoke(tmp_path):
    gdf = make_point_z_gdf()
    result = ConceptualModeling.plot_isosurface(
        gdf,
        col="value",
        units="val",
        title="smoke",
        contour_level=0.5,
        off_screen=True,
        screenshot_path=str(tmp_path / "iso.png"),
    )
    assert result is not None
    assert {"grid", "iso", "plotter"} <= result.keys()


def test_plot_isosurface_empty_gdf_warns():
    gdf = gpd.GeoDataFrame({"geometry": []})
    with pytest.warns(UserWarning, match="empty"):
        result = ConceptualModeling.plot_isosurface(
            gdf, col="value", units="val", title="smoke", contour_level=0.5
        )
    assert result is None


def test_plot_isosurface_missing_col_raises():
    gdf = make_point_z_gdf()
    with pytest.raises(ValueError, match="not found"):
        ConceptualModeling.plot_isosurface(
            gdf,
            col="nonexistent",
            units="val",
            title="smoke",
            contour_level=0.5,
        )


def test_plot_isosurface_2d_geom_raises():
    gdf = gpd.GeoDataFrame(
        {"value": [0.4, 0.6], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    with pytest.raises(ValueError, match="Z coordinates"):
        ConceptualModeling.plot_isosurface(
            gdf, col="value", units="val", title="smoke", contour_level=0.5
        )


# ---------------------------------------------------------------------------
# plot_conceptual_model
# ---------------------------------------------------------------------------


@_needs_pv_plotting
def test_plot_conceptual_model_single_col_smoke(tmp_path):
    gdf = make_point_z_gdf()
    result = ConceptualModeling.plot_conceptual_model(
        gdf,
        cols="value",
        units="val",
        title="smoke",
        contour_levels=0.5,
        off_screen=True,
        screenshot_path=str(tmp_path / "cm_single.png"),
    )
    assert result is not None
    assert {
        "grid",
        "grid_clipped",
        "iso_components",
        "plotter",
    } <= result.keys()


@_needs_pv_plotting
def test_plot_conceptual_model_multi_col_smoke(tmp_path):
    gdf = make_point_z_gdf()
    gdf["value2"] = np.random.default_rng(1).random(len(gdf))
    result = ConceptualModeling.plot_conceptual_model(
        gdf,
        cols=["value", "value2"],
        units="val",
        title="smoke",
        contour_levels=0.5,
        off_screen=True,
        screenshot_path=str(tmp_path / "cm_multi.png"),
    )
    assert result is not None
    assert set(result["iso_components"].keys()) <= {"value", "value2"}


def test_plot_conceptual_model_empty_gdf_warns():
    gdf = gpd.GeoDataFrame({"geometry": []})
    with pytest.warns(UserWarning, match="empty"):
        result = ConceptualModeling.plot_conceptual_model(
            gdf, cols="value", units="val", title="smoke", contour_levels=0.5
        )
    assert result is None


def test_plot_conceptual_model_missing_col_raises():
    gdf = make_point_z_gdf()
    with pytest.raises(ValueError, match="not found"):
        ConceptualModeling.plot_conceptual_model(
            gdf,
            cols="nonexistent",
            units="val",
            title="smoke",
            contour_levels=0.5,
        )


def test_plot_conceptual_model_2d_geom_raises():
    gdf = gpd.GeoDataFrame(
        {"value": [0.4, 0.6], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    with pytest.raises(ValueError, match="Z coordinates"):
        ConceptualModeling.plot_conceptual_model(
            gdf, cols="value", units="val", title="smoke", contour_levels=0.5
        )
