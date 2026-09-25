from itertools import starmap

import geopandas as gpd
from shapely.geometry import Point

from geopfa.processing import Processing


def _build_gdf(points, crs=26910):
    """Build a single-fault GeoDataFrame from a list of (x, y, z) points."""
    return gpd.GeoDataFrame(
        {"Fault_Number": [1] * len(points)},
        geometry=list(starmap(Point, points)),
        crs=crs,
    )


def test_create_fault_surfaces_keeps_concave_vertical_trace():
    """A concave vertical fault trace should be built as a strip of
    quads following the trace, rather than being flattened to a convex
    hull, which would drop the concave (2, 0) vertex."""
    trace_xy = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
    points = [(x, y, z) for z in (0, -100) for x, y in trace_xy]
    gdf = _build_gdf(points)

    surfaces = Processing.create_fault_surfaces_from_points(
        gdf, "Fault_Number"
    )
    geom = surfaces.iloc[0].geometry

    assert geom.geom_type == "MultiPolygon"
    assert len(geom.geoms) == len(trace_xy) - 1
    xy_coords = {
        (c[0], c[1]) for quad in geom.geoms for c in quad.exterior.coords
    }
    assert (2, 0) in xy_coords


def test_create_fault_surfaces_handles_dipping_top_bottom_pair():
    """Ring-fault-style data (few points, top/bottom XY differ) should
    produce a valid, non-degenerate strip of quads."""
    top = [(0, 0, 100), (10, 0, 100), (20, 0, 100)]
    bottom = [(0, 5, -100), (10, 5, -100), (20, 5, -100)]
    gdf = _build_gdf(top + bottom)

    surfaces = Processing.create_fault_surfaces_from_points(
        gdf, "Fault_Number"
    )
    geom = surfaces.iloc[0].geometry

    assert geom.geom_type == "MultiPolygon"
    assert len(geom.geoms) == len(top) - 1
    # Consecutive quads share a full edge, which the strict OGC
    # MultiPolygon validity rules disallow (boundaries may only touch at
    # finitely many points), so `geom.is_valid` is expected to be False
    # here even though each quad is itself well-formed.
    assert all(quad.is_valid for quad in geom.geoms)
    assert geom.area > 0


def test_create_fault_surfaces_falls_back_for_multi_segment_trace():
    """Multiple disjoint sub-traces sharing one fault number should not
    be stitched into a single ribbon; the convex hull fallback is used
    instead."""
    top = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (1000, 0, 0),
        (1001, 0, 0),
        (1002, 0, 0),
    ]
    bottom = [(x, 5, -100) for x, _, _ in top]
    gdf = _build_gdf(top + bottom)

    surfaces = Processing.create_fault_surfaces_from_points(
        gdf, "Fault_Number"
    )
    geom = surfaces.iloc[0].geometry

    assert geom.geom_type == "Polygon"
    assert geom.is_valid
    assert len(geom.exterior.coords) - 1 < len(top) + len(bottom)


def test_slice_geometry_at_z_handles_quad_strip_multipolygon():
    """A MultiPolygon quad-strip fault surface should be sliceable at a
    z-level, returning the union of footprints from the quads that span
    that elevation."""
    top = [(0, 0, 100), (10, 0, 100), (20, 0, 100)]
    bottom = [(0, 5, -100), (10, 5, -100), (20, 5, -100)]
    gdf = _build_gdf(top + bottom)

    surfaces = Processing.create_fault_surfaces_from_points(
        gdf, "Fault_Number"
    )
    geom = surfaces.iloc[0].geometry

    fp = Processing.slice_geometry_at_z(geom, 0)
    assert fp is not None
    assert not fp.is_empty

    assert Processing.slice_geometry_at_z(geom, 1000) is None
