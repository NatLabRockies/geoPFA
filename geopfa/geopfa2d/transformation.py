"""
Set of methods to transform data from evidence layers into evidence layers.
"""

import geopandas as gpd
import numpy as np
import shapely

from geopfa.transformation import normalize_gdf as _normalize_gdf
from geopfa.transformation import normalize_array as _normalize_array
from geopfa.transformation import transform as _transform


class VoterVetoTransformation:
    """Class of functions for use in transforming data layers into evidence layers
    i.e., data values to 'favorability' values.
    """

    @staticmethod
    def normalize_gdf(gdf, col, norm_to=1):
        return _normalize_gdf(gdf, col, norm_to)

    @staticmethod
    def normalize_array(rasterized_array,method):
        """Normalize a 2D NumPy array.
        """
        return _normalize_array(rasterized_array, method)

    @staticmethod
    def transform(array, method):
        """Function to transform rasterized array to map data values to relative favorability values.
        Includes several types of transformation methods
        """
        return _transform(array, method)

    @staticmethod
    def rasterize_model(gdf, col):
        """Function to go from a geodataframe to a rasterized 2D numpy array representation for
        use in linear algebra functions. Maintains the resolution of the geodataframe

        Parameters
        ----------
        gdf : geodataframe
            GeoDataFrame with point geometry containing model to rasterize
        col : str
            Name of column in gdf where data values are stored

        Returns
        -------
        rasterized_model : np.ndarray
            Numpy array containing 2D rasterized version of gdf
        """
        if len(gdf) == 0:
            raise ValueError("GeoDataFrame 'gdf' is empty.")

        # Get the unique x and y coordinates from the GeoDataFrame
        unique_x = np.sort(gdf.geometry.x.unique())
        unique_y = np.sort(gdf.geometry.y.unique())

        # Debugging: Print unique coordinates
        # print("Unique X Coordinates:", unique_x)
        # print("Unique Y Coordinates:", unique_y)

        # Check if unique_y is empty
        if len(unique_y) == 0:
            raise ValueError("No y-coordinates found in 'gdf'.")

        # Determine the number of unique x and y coordinates
        num_cols = len(unique_x)
        num_rows = len(unique_y)

        # Create an empty 2D NumPy array representing the rasterized model
        rasterized_model = np.zeros((num_rows, num_cols), dtype=np.float32)  # Use float32 to support non-integer values

        # Invert the y-coordinates
        min_y = gdf.geometry.y.min()
        max_y = gdf.geometry.y.max()
        gdf['inverted_y'] = min_y + (max_y - gdf.geometry.y)

        # Debugging: Print inverted_y values
        # print("Inverted Y Values:", gdf['inverted_y'].unique())

        # Tolerance for floating point comparisons
        tolerance = 1e-6

        # Iterate over each point in the GeoDataFrame and rasterize onto the model
        for _, row in gdf.iterrows():
            # Extract the associated value or column at the point
            value = row[col]

            # Find the index of the point's coordinates in the unique x and y arrays
            col_idx = np.where(np.abs(unique_x - row.geometry.x) < tolerance)[0]
            row_idx = np.where(np.abs(unique_y - row.inverted_y) < tolerance)[0]

            # Debugging: Print coordinate indices
            # print(f"Point ({row.geometry.x}, {row.geometry.y}) -> Col Index: {col_idx}, Row Index: {row_idx}")

            if len(col_idx) == 0 or len(row_idx) == 0:
                print(f"Warning: Point ({row.geometry.x}, {row.geometry.y}) not found in unique coordinates")
                continue

            col_idx = col_idx[0]
            row_idx = row_idx[0]

            # Update the pixel value with the associated value
            rasterized_model[row_idx, col_idx] = value

        return rasterized_model

    @staticmethod
    def derasterize_model(rasterized_model, gdf_geom):
        """Function to go from a rasterized 2D numpy array representation back to a geodataframe.
        Retains geometry of the original geodataframe that was rasterized.

        Parameters
        ----------
        rasterized_model : np.ndarray
            Numpy array containing 2D rasterized version of gdf
        gdf_geom : Pandas GeoSeries
            GeoSeries representing geomtry from original GeoDataFrame to use to transform
            rasterized array back into a GeoDataFrame.

        Returns
        -------
        gdf : geodataframe
            GeoDataFrame with point geometry containing model to rasterize
        """
        if len(gdf_geom) == 0:
            raise ValueError("GeoDataFrame 'gdf_geom' is empty.")

        # Get the unique x and y coordinates from the GeoDataFrame
        unique_x = gdf_geom.geometry.x.unique()
        unique_y = gdf_geom.geometry.y.unique()
        crs = gdf_geom.crs

        # Determine the number of unique x and y coordinates
        num_cols = len(unique_x)
        num_rows = len(unique_y)

        # Create an empty list to store Point geometries
        geometries = []
        rasterized_model = np.flipud(rasterized_model)

        # Iterate over each row and column in the rasterized model
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                # Calculate the x and y coordinates of the raster cell
                x_coord = unique_x[col_idx]
                y_coord = unique_y[row_idx]

                # Create a Point geometry using the x and y coordinates
                point = shapely.geometry.Point(x_coord, y_coord)

                # Append the Point geometry to the list
                geometries.append(point)

        # Create a GeoDataFrame from the geometries and rasterized values
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        # Assign the values from the rasterized model to the 'col' column of the GeoDataFrame
        gdf['favorability'] = rasterized_model.flatten()
        return gdf
