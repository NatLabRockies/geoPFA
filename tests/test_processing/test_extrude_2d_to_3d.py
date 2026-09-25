import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from geopfa.processing import Processing


def _build_pfa(lines, crs=26910):
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    return {
        "criteria": {
            "geologic": {
                "components": {
                    "reservoir": {"layers": {"faults": {"data": gdf}}}
                }
            }
        }
    }


def _bottom_xy_offset(gdf3):
    """Return the (dx, dy) shift between a solid's top and bottom trace."""
    coords = list(gdf3.geometry.iloc[0].exterior.coords)
    n = (len(coords) - 1) // 2
    top, bottom = coords[:n], coords[n : 2 * n]
    top_first_xy = np.array(top[0][:2])
    bottom_last_xy = np.array(bottom[-1][:2])
    return bottom_last_xy - top_first_xy


def test_extrude_2d_to_3d_requires_strike_for_dip_to_apply():
    """dip alone (no strike) is documented to yield a vertical extrusion."""
    line = LineString([(0, 0), (100, 0)])
    pfa = _build_pfa([line])
    extent = (-500, -500, -1000, 500, 500, 0)

    pfa = Processing.extrude_2d_to_3d(
        pfa,
        criteria="geologic",
        component="reservoir",
        layer="faults",
        extent=extent,
        nz=1,
        dip=None,
        strike=None,
    )
    gdf3 = pfa["criteria"]["geologic"]["components"]["reservoir"]["layers"][
        "faults"
    ]["data"]
    offset = _bottom_xy_offset(gdf3)
    assert np.allclose(offset, [0, 0])


def test_extrude_2d_to_3d_auto_strike_dips_each_line_independently():
    """With strike=None and dip given, differently oriented traces should
    each dip along their own bearing rather than a shared direction."""
    ew_line = LineString([(0, 0), (100, 0)])
    ns_line = LineString([(0, 0), (0, 100)])
    pfa = _build_pfa([ew_line, ns_line])
    extent = (-500, -500, -1000, 500, 500, 0)

    pfa = Processing.extrude_2d_to_3d(
        pfa,
        criteria="geologic",
        component="reservoir",
        layer="faults",
        extent=extent,
        nz=1,
        dip=65,
        strike=None,
    )
    gdf3 = pfa["criteria"]["geologic"]["components"]["reservoir"]["layers"][
        "faults"
    ]["data"]

    ew_offset = _bottom_xy_offset(gdf3.iloc[[0]])
    ns_offset = _bottom_xy_offset(gdf3.iloc[[1]])

    assert not np.allclose(ew_offset, [0, 0])
    assert not np.allclose(ns_offset, [0, 0])
    assert not np.allclose(ew_offset, ns_offset)
