"""Methods to transform data from evidence layers."""

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point
from itertools import starmap
import warnings


def normalize_gdf(gdf, col, norm_to=1):
    """Normalize a GeoDataFrame using min-max scaling

    Normalize the values in a specified column of a GeoDataFrame using
    min-max scaling, such that the minimum value becomes 0 and the
    maximum value becomes norm_to.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing the column to normalize.
    col : str
        The name of the column in the GeoDataFrame to normalize.
    norm_to : int or float, optional
        The value to which the maximum column value should be scaled
        (default is 1).

    Returns
    -------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame with the normalized column.

    ..NOTE:: This function modifies the input GeoDataFrame in place,
            thus even if the output is assigned to a new variable,
            the original input GeoDataFrame will be modified.
    """
    min_val = gdf[col].min()
    max_val = gdf[col].max()

    # avoid division by zero if all values in the column are the same
    if min_val == max_val:
        gdf[col] = norm_to  # all values are the same, set them to norm_to
    else:
        # Perform min-max normalization
        gdf[col] = (gdf[col] - min_val) / (max_val - min_val) * norm_to

    return gdf


def normalize_array(rasterized_array, method):
    """Normalize a 2D or 3D NumPy array.

    Parameters
    ----------
    rasterized_array : np.ndarray
        Input NumPy array to be normalized.
    method : str
        Method to use to normalize rasterized_array. Can be one of
        ['minmax','mad']

    Returns
    -------
    normalized_array : np.ndarray
        Normalized 2D or 3D NumPy array.
    """
    if method == "minmax":
        min_val = np.nanmin(rasterized_array)
        max_val = np.nanmax(rasterized_array)

        if np.isclose(max_val, min_val):
            # degenerate case: constant or effectively constant array
            normalized_array = np.zeros_like(rasterized_array, dtype=float)
        else:
            # normalize the array to the range [0, 1]
            normalized_array = (rasterized_array - min_val) / (
                max_val - min_val
            )

    elif method == "mad":
        median = np.nanmedian(rasterized_array)
        mad = np.nanmedian(np.abs(rasterized_array - median))

        if mad == 0 or np.isnan(mad):
            warnings.warn(
                "MAD normalization is ill-defined for this layer (MAD = 0). "
                "Falling back to minmax normalization.",
                stacklevel=2,
            )
            # reuse the minmax branch
            return normalize_array(rasterized_array, method="minmax")

        num = rasterized_array - median
        den = 1.482 * mad
        normalized_array = num / den

    else:
        raise ValueError("Invalid method. Please use 'minmax' or 'mad'.")

    return normalized_array


def detect_geom_dimension(gdf: gpd.GeoDataFrame) -> int:
    """Detect whether a GeoDataFrame geometry is 2D or 3D.

    Raises a ValueError if gdf is empty or has mixed dimensionality.
    """
    if gdf is None or len(gdf) == 0:
        raise ValueError(
            "Cannot detect geometry dimension from an empty GeoDataFrame."
        )

    dims = set()
    for geom in gdf.geometry:
        if geom is None:
            continue
        # shapely Point has .has_z; for other geometry types, fall back on coord length
        has_z = getattr(geom, "has_z", False)
        if has_z:
            dims.add(3)
        else:
            dims.add(2)
        if len(dims) > 1:
            break

    if len(dims) == 0:
        raise ValueError(
            "Could not determine geometry dimension; all geometries are None."
        )

    if len(dims) > 1:
        raise ValueError(
            "Mixed 2D and 3D geometries found within a single GeoDataFrame. "
            "All geometries for a given layer must be consistently 2D or 3D."
        )

    return dims.pop()


class VoterVetoTransformation:
    """Unified transformation for voter-veto that supports both 2D and 3D."""

    @staticmethod
    def transform(array, method):
        """Transform to relative favorability values

        Function to transform rasterized array to map data values to
        relative favorability values. Includes several types of
        transformation methods

        Parameters
        ----------
        array : np.ndarray
            Input 2D or 3D rasterized np.array to transform
        method : str
            Method to transform data to relative favorability. Can be one
            of ['inverse', 'negate', 'ln', 'None', 'hill', 'valley']

        Returns
        -------
        transformed_array : np.ndarray
            Array with data values transformed to relative favorability
            values
        """
        if method.lower() == "inverse":
            transformed_array = 1 / array
        elif method.lower() == "negate":
            transformed_array = -array
        elif method.lower() == "ln":
            transformed_array = np.log(array)
        elif method.lower() == "none":
            transformed_array = array
        elif method.lower() in {"hill", "valley"}:
            median = np.nanmedian(array)
            mad = np.nanmedian(np.abs(array - median))
            if mad == 0:
                mad = 1e-6  # prevent division by zero
            squared_dist = (array - median) ** 2
            gaussian = np.exp(-squared_dist / (2 * mad**2))
            transformed_array = (
                gaussian if method.lower() == "hill" else 1 - gaussian
            )
        else:
            raise ValueError(
                "Transformation method ", method, " not yet implemented."
            )

        return transformed_array

    @staticmethod
    def rasterize_model_2d(gdf, col):
        """2D rasterization: point GeoDataFrame -> 2D numpy array (vectorized)."""
        if len(gdf) == 0:
            raise ValueError("GeoDataFrame 'gdf' is empty.")

        xs = gdf.geometry.x.to_numpy()
        ys = gdf.geometry.y.to_numpy()
        vals = gdf[col].to_numpy()

        unique_x = np.sort(np.unique(xs))
        unique_y = np.sort(np.unique(ys))

        num_cols = len(unique_x)
        num_rows = len(unique_y)

        raster = np.full((num_rows, num_cols), np.nan, dtype=np.float32)

        # invert Y once
        min_y = unique_y.min()
        max_y = unique_y.max()
        inverted_y = min_y + (max_y - ys)

        # map coordinates -> indices
        xi = np.searchsorted(unique_x, xs)
        yi = np.searchsorted(unique_y, inverted_y)

        raster[yi, xi] = vals

        return raster

    @staticmethod
    def derasterize_model_2d(rasterized_model, gdf_geom):
        """2D derasterization: 2D array -> GeoDataFrame with 2D points (vectorized)."""
        if len(gdf_geom) == 0:
            raise ValueError("GeoDataFrame 'gdf_geom' is empty.")

        unique_x = np.sort(gdf_geom.geometry.x.unique())
        unique_y = np.sort(gdf_geom.geometry.y.unique())
        crs = gdf_geom.crs

        raster = np.flipud(rasterized_model)

        xs, ys = np.meshgrid(unique_x, unique_y)
        geoms = [Point(x, y) for x, y in zip(xs.ravel(), ys.ravel())]  # noqa: FURB140

        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        gdf["favorability"] = raster.ravel()
        return gdf

    @staticmethod
    def rasterize_model_3d(gdf, col):
        """3D rasterization: point-Z GeoDataFrame -> 3D numpy array (vectorized)."""
        if len(gdf) == 0:
            raise ValueError("GeoDataFrame 'gdf' is empty.")

        xs = gdf.geometry.x.to_numpy()
        ys = gdf.geometry.y.to_numpy()
        zs = np.array(
            [p.z if getattr(p, "has_z", False) else 0.0 for p in gdf.geometry]
        )
        vals = gdf[col].to_numpy()

        unique_x = np.sort(np.unique(xs))
        unique_y = np.sort(np.unique(ys))
        unique_z = np.sort(np.unique(zs))

        nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)

        raster = np.full((nz, ny, nx), np.nan, dtype=np.float32)

        xi = np.searchsorted(unique_x, xs)
        yi = np.searchsorted(unique_y, ys)
        zi = np.searchsorted(unique_z, zs)

        raster[zi, yi, xi] = vals
        return raster

    @staticmethod
    def derasterize_model_3d(rasterized_model, gdf_geom):
        """3D derasterization: 3D array -> GeoDataFrame with 3D points (vectorized)."""
        if len(gdf_geom) == 0:
            raise ValueError("GeoDataFrame 'gdf_geom' is empty.")

        unique_x = np.sort(gdf_geom.geometry.x.unique())
        unique_y = np.sort(gdf_geom.geometry.y.unique())
        unique_z = np.sort(
            gdf_geom.geometry.apply(
                lambda p: p.z if getattr(p, "has_z", False) else 0
            ).unique()
        )
        crs = gdf_geom.crs

        zs, ys, xs = np.meshgrid(unique_z, unique_y, unique_x, indexing="ij")
        geoms = [  # noqa: FURB140
            Point(x, y, z)
            for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel())
        ]

        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        gdf["favorability"] = rasterized_model.ravel()
        return gdf
