import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from .campbell2d import campbell2d, DEFAULT_THETA, NEG_THETA, POS_THETA

def generate_campbell2d_grid(
    nx=20,
    ny=20,
    theta=DEFAULT_THETA,
    noise=0.0,
    missing_pattern="none",
):
    """
    Generates:
        - GeoDataFrame with x,y,value
        - X_grid, Y_grid (meshgrids)
        - Z_true (full Campbell2D)
        - Z_obs (with missing values)
        - nan_mask
    """

    xs = np.linspace(-3, 3, nx)
    ys = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(xs, ys)

    Z_true = campbell2d(X, Y, theta)

    if noise > 0:
        Z_true = Z_true + noise * np.random.randn(*Z_true.shape)

    nan_mask = np.zeros_like(Z_true, dtype=bool)

    if missing_pattern == "center_block":
        cx0, cx1 = nx // 3, 2 * nx // 3
        cy0, cy1 = ny // 3, 2 * ny // 3
        nan_mask[cy0:cy1, cx0:cx1] = True

    elif missing_pattern == "border_missing":
        nan_mask[0:3, :] = True
        nan_mask[-3:, :] = True
        nan_mask[:, 0:3] = True
        nan_mask[:, -3:] = True

    elif missing_pattern == "random_20pct":
        nan_mask = np.random.rand(*Z_true.shape) < 0.2

    # Apply mask
    Z_obs = Z_true.copy()
    Z_obs[nan_mask] = np.nan

    # Build GeoDataFrame
    pts = []
    vals = []
    for i in range(ny):
        for j in range(nx):
            pts.append(Point(float(X[i, j]), float(Y[i, j])))
            vals.append(Z_obs[i, j])

    gdf = gpd.GeoDataFrame({"geometry": pts, "value": vals})

    return gdf, X, Y, Z_true, Z_obs, nan_mask


def get_theta_choices():
    return {
        "default": DEFAULT_THETA,
        "negative": NEG_THETA,
        "positive": POS_THETA,
    }

