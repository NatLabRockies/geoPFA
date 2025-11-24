"""Methods to transform data from evidence layers."""

import geopandas as gpd
import numpy as np


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
    # Find the min and max of the column
    min_val = gdf[col].min()
    max_val = gdf[col].max()

    # Avoid division by zero if all values in the column are the same
    if min_val == max_val:
        gdf[col] = norm_to  # All values are the same, set them to norm_to
    else:
        # Perform min-max normalization
        gdf[col] = (gdf[col] - min_val) / (max_val - min_val) * norm_to

    return gdf


def normalize_array(rasterized_array, method):
    """Normalize a 2D NumPy array.

    Parameters
    ----------
    rasterized_array : np.ndarray
        Input 2D NumPy array to be normalized.
    method : str
        Method to use to normalize rasterized_array. Can be one of
        ['minmax','mad']

    Returns
    -------
    normalized_array : np.ndarray
        Normalized 2D NumPy array.
    """
    if method == "minmax":
        # Find the minimum and maximum values in the array
        min_val = np.nanmin(rasterized_array)
        max_val = np.nanmax(rasterized_array)

        # Normalize the array to the range [0, 1]
        normalized_array = (rasterized_array - min_val) / (max_val - min_val)
        print("Normalized a layer using " + method)

    elif method == "mad":
        num = rasterized_array - np.nanmedian(rasterized_array)
        den = 1.482 * np.nanmedian(np.abs(num))
        normalized_array = num / den

        print("Normalized a layer using " + method + " >:(")

    else:
        raise ValueError("Invalid method. Please use 'minmax' or 'mad'.")

    return normalized_array


def transform(array, method):
    """Transform to relative favorability values

    Function to transform rasterized array to map data values to
    relative favorability values. Includes several types of
    transformation methods

    Parameters
    ----------
    array : np.ndarray
        Input 2D rasterized np.array to transform
    method : str
        Method to transform data to relative favorability. Can be one
        of ['inverse', 'negate', 'ln', 'None', 'hill', 'valley']

    Returns
    -------
    transformed_array : np.ndarray
        Array with data values transformed to relative favorability
        values
    """
    if (method == "inverse") | (method == "Inverse"):
        transformed_array = 1 / array
    elif (method == "negate") | (method == "Negate"):
        transformed_array = -array
    elif (method == "ln") | (method == "Ln"):
        transformed_array = np.log(array)
    elif (method == "none") | (method == "None"):
        transformed_array = array
    elif method in {"hill", "valley"}:
        median = np.nanmedian(array)
        mad = np.nanmedian(np.abs(array - median))
        if mad == 0:
            mad = 1e-6  # prevent division by zero
        squared_dist = (array - median) ** 2
        gaussian = np.exp(-squared_dist / (2 * mad**2))
        transformed_array = gaussian if method == "hill" else 1 - gaussian
    else:
        raise ValueError(
            "Transformation method ", method, " not yet implemented."
        )
    print("Transformed a layer using " + method)

    return transformed_array
