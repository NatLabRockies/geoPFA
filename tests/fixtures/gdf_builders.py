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

