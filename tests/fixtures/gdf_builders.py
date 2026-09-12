import geopandas as gpd
from shapely.geometry import Point
import numpy as np


def gdf_from_xy_value(xs, ys, Z):
    """
    Helper that converts x,y,Z to a GeoDataFrame with column 'value'.
    Z can contain NaNs.
    """

    pts = []
    vals = []
    ny, nx = Z.shape

    for i in range(ny):
        for j in range(nx):
            pts.append(Point(xs[j], ys[i]))
            vals.append(Z[i, j])

    return gpd.GeoDataFrame({"geometry": pts, "value": vals})


def extract_xy_from_gdf(gdf):
    """Return arrays x, y extracted from geometry."""
    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    return x, y


def make_point_z_gdf(nx=5, ny=5, nz=3, seed=0):
    """Return a small regular Point-Z GeoDataFrame for 3D plotter smoke tests.

    Parameters
    ----------
    nx, ny, nz : int
        Number of points along each spatial axis.
    seed : int
        Random seed for the 'value' column.

    Returns
    -------
    geopandas.GeoDataFrame
        Point-Z GeoDataFrame with a single 'value' column in [0, 1].
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 100.0, nx)
    ys = np.linspace(0.0, 100.0, ny)
    zs = np.linspace(-500.0, -100.0, nz)
    pts = [
        Point(float(x), float(y), float(z)) for z in zs for y in ys for x in xs
    ]
    return gpd.GeoDataFrame({"value": rng.random(len(pts)), "geometry": pts})
