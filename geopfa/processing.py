"""
Processing utilities for preparing geothermal favorability layers.

This module provides classes and functions to transform raw geospatial data
gridded datasets suitable for consistent spatial analysis. Operations largely
act on the `pfa` dictionary (built with the data_readers module) and include
data cleaning, interpolation, and other various preprocessing functions for
PFA layer combination.
"""

import time
import warnings

import geopandas as gpd
import pandas as pd
import scipy
import numpy as np
import math
import shapely
from shapely.geometry.polygon import Polygon, LineString, Point
from shapely.ops import unary_union
from pykrige.ok3d import OrdinaryKriging3D
from scipy.interpolate import griddata
from osgeo import osr
from itertools import starmap

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from geopfa.transformation import transform
from geopfa.extrapolation import backfill_gdf, drop_z_from_geometry

try:
    # Shapely 2.0 vectorized accessors (fast path)
    from shapely import get_z

    _HAS_GET_Z = True
except Exception:
    _HAS_GET_Z = False


class Cleaners:
    """Class of functions for use in processing data into models"""

    @staticmethod
    def set_crs(pfa, target_crs=3857):
        """Function to project all data layers to the same desired CRS. Used to get all
        data layers on the same CRS.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames to convert CRS.
        target_crs : int
            Nubmber associated with the desired CRS of resulting interpolation. Defaults
            to 3857, which is WGS84.

        Returns
        -------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames projected onto target_crs.
        """
        for criteria in pfa["criteria"]:
            for component in pfa["criteria"][criteria]["components"]:
                for layer in pfa["criteria"][criteria]["components"][
                    component
                ]["layers"]:
                    pfa["criteria"][criteria]["components"][component][
                        "layers"
                    ][layer]["data"] = pfa["criteria"][criteria]["components"][
                        component
                    ]["layers"][layer]["data"].to_crs(target_crs)
        return pfa

    @staticmethod
    def clean_unnamed_columns(pfa):
        """
        Loops through the `pfa` dictionary, finds GeoDataFrames, and drops columns that start with 'Unnamed:'.

        Parameters
        ----------
        pfa : dict
            Nested dictionary structure containing criteria, components, and layers.

        Returns
        -------
        None
            The function modifies the GeoDataFrames in-place within the `pfa` dictionary.
        """
        for criteria in pfa["criteria"]:
            print(criteria)
            for component in pfa["criteria"][criteria]["components"]:
                print("\t" + component)
                for layer in pfa["criteria"][criteria]["components"][
                    component
                ]["layers"]:
                    print("\t\t" + layer)

                    # Access the GeoDataFrame
                    gdf = pfa["criteria"][criteria]["components"][component][
                        "layers"
                    ][layer]["model"]

                    # Check if it is a GeoDataFrame and contains 'Unnamed:' columns
                    if isinstance(gdf, gpd.GeoDataFrame):
                        unnamed_columns = [
                            col
                            for col in gdf.columns
                            if col.startswith("Unnamed:")
                        ]

                        if unnamed_columns:
                            print(
                                f"\t\t\tDropping columns: {', '.join(unnamed_columns)}"
                            )
                            gdf = gdf.drop(columns=unnamed_columns)
                        else:
                            print("\t\t\tNo 'Unnamed:' columns found.")
        return pfa

    @staticmethod
    def convert_z_measurements(gdf, z_meas, target_z_meas):
        """
        Converts depth or elevation measurements from one reference system to another using GDAL Python Bindings.

        Parameters
        ----------
            gdf (GeoDataFrame): GeoDataFrame containing point geometry and Z values in the geometry.
            z_meas (str): Current measurement reference (e.g., 'm-msl', 'epsg:####', or 'ft-msl').
            target_z_meas (str): Target measurement reference (e.g., 'm-msl', 'epsg:####', or 'ft-msl').

        Returns
        -------
            GeoDataFrame: A GeoDataFrame with updated geometry where the Z value is converted to the target reference.
        """
        METERS_TO_FEET = 3.28084
        FEET_TO_METERS = 1 / METERS_TO_FEET

        # Set up source and target spatial references
        source_srs = osr.SpatialReference()
        if z_meas.startswith("epsg:"):
            source_srs.ImportFromEPSG(int(z_meas.split(":")[1]))
            print("\t\t successful import")

        target_srs = osr.SpatialReference()
        if target_z_meas.startswith("epsg:"):
            target_srs.ImportFromEPSG(int(target_z_meas.split(":")[1]))
            print("\t\t successful import")

        # Coordinate transformation
        transform = osr.CoordinateTransformation(source_srs, target_srs)

        # Function to update Z values based on input and target references
        def convert_z(geom):
            current_z = geom.z
            new_z = current_z

            if z_meas == "m-msl" and target_z_meas == "ft-msl":
                new_z = current_z * METERS_TO_FEET
            elif z_meas == "ft-msl" and target_z_meas == "m-msl":
                new_z = current_z * FEET_TO_METERS
            elif z_meas.startswith("epsg:") and target_z_meas.startswith(
                "epsg:"
            ):
                print("\t\t ", "transforming ", z_meas, " to ", target_z_meas)
                _x, _y, z = transform.TransformPoint(geom.x, geom.y, current_z)
                new_z = z  # Updated Z from the transformation

            return shapely.geometry.Point(geom.x, geom.y, new_z)

        # Apply Z conversion
        gdf["geometry"] = gdf.geometry.apply(convert_z)

        return gdf

    @staticmethod
    def filter(data, quantile=0.9):
        """Filter out data values above a specified quantile by setting them to that quantile.

        Parameters
        ----------
        data : pandas.Series
            Series of data values to filter
        quantile : int
            Number representing the quantile, which when exceeded, produces an outlier.

        Returns
        -------
        data : pandas.Series
            Filtered version of the input data, with values above specified quantile set to
            that quantile.
        """
        q = data.quantile(quantile)
        data.loc[data > q] = q
        return data

    @staticmethod
    def filter_series(series, quantile=0.9):
        """Filter out data values above a specified quantile by setting them to that quantile.

        Parameters
        ----------
        series : pandas.Series
            Series of data values to filter.
        quantile : float
            Number representing the quantile, which when exceeded, produces an outlier.

        Returns
        -------
        series : pandas.Series
            Filtered version of the input data, with values above the specified quantile set to
            that quantile.
        """
        series = series.copy()
        q = series.quantile(quantile)
        series.loc[series > q] = q
        return series

    @staticmethod
    def filter_geodataframe(gdf, column, quantile=0.9):
        """Apply the filter function to specified columns in a GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing the data to filter.
        column : list of str
            Column name to apply the filter to.
        quantile : float
            Number representing the quantile, which when exceeded, produces an outlier.

        Returns
        -------
        gdf_filtered : geopandas.GeoDataFrame
            GeoDataFrame with the specified columns filtered.
        """
        if column in gdf.columns:
            gdf[column] = Cleaners.filter_series(gdf[column], quantile)
        else:
            print(
                f"Column '{column}' could not be filtered because it is not in the dataframe."
            )
        return gdf

    @staticmethod
    def get_extent(gdf: gpd.GeoDataFrame, dim="auto"):
        """
        Compute the spatial extent (bounding box) of a GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing Point, LineString, Polygon,
            Multi*, or GeometryCollection geometries.
        dim : {"auto", 2, 3}, optional
            Controls dimensionality of returned extent.
            - "auto" (default): returns 2D extent if all geometries
            are 2D. Returns 3D extent if ANY geometries contain Z
            values; warns if mixed 2D/3D.
            - 2: always return 2D extent [xmin, ymin, xmax, ymax],
            even if geometries contain Z values.
            - 3: force 3D extent [xmin, ymin, zmin, xmax, ymax, zmax].
            Raises ValueError if no Z values are present.

        Returns
        -------
        extent : list
            2D: [xmin, ymin, xmax, ymax]
            3D: [xmin, ymin, zmin, xmax, ymax, zmax]

        Raises
        ------
        ValueError
            If GeoDataFrame is empty.
            If dim=3 but no Z values exist.
            If dim is invalid.

        Warns
        -----
        UserWarning
            If mixed 2D and 3D geometries are present and dim="auto".
            Z bounds will be computed using geometries with Z only.

        Notes
        -----
        - XY bounds are computed using GeoDataFrame.total_bounds.
        - Z bounds are computed by iterating over all coordinates.
        - Multi* and GeometryCollection geometries are handled recursively.
        """

        if gdf.empty or gdf.geometry.is_empty.all():
            raise ValueError("Cannot compute extent of empty GeoDataFrame.")

        if dim not in ("auto", 2, 3):
            raise ValueError("dim must be one of {'auto', 2, 3}.")

        xmin, ymin, xmax, ymax = gdf.total_bounds

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                "Invalid extent: xmax/xmin or ymax/ymin ordering is incorrect."
            )

        # Always return 2D if explicitly requested
        if dim == 2:
            return [xmin, ymin, xmax, ymax]

        # Detect Z presence
        has_z_flags = gdf.geometry.apply(
            lambda geom: getattr(geom, "has_z", False)
        )

        any_z = has_z_flags.any()
        all_z = has_z_flags.all()

        if not any_z:
            if dim == 3:
                raise ValueError(
                    "dim=3 requested but geometries do not contain Z values."
                )
            return [xmin, ymin, xmax, ymax]

        if dim == "auto" and not all_z:
            warnings.warn(
                "Mixed 2D and 3D geometries detected. "
                "Z extent will be computed using geometries with Z values only.",
                UserWarning,
                stacklevel=2,
            )

        def extract_z(geom):
            if geom.is_empty:
                return []

            zs = []

            def recurse(g):
                if hasattr(g, "geoms"):
                    for part in g.geoms:
                        recurse(part)
                else:
                    try:
                        coords = list(g.coords)
                    except Exception:
                        try:
                            coords = list(g.exterior.coords)
                            for ring in g.interiors:
                                coords += list(ring.coords)
                        except Exception:
                            return
                    for c in coords:
                        if len(c) == 3:
                            zs.append(c[2])

            recurse(geom)
            return zs

        z_values = []
        for geom in gdf.geometry:
            z_values.extend(extract_z(geom))

        if not z_values:
            if dim == 3:
                raise ValueError(
                    "dim=3 requested but no valid Z coordinates found."
                )
            return [xmin, ymin, xmax, ymax]

        zmin = min(z_values)
        zmax = max(z_values)

        if zmax <= zmin:
            raise ValueError("Invalid extent: zmax must be greater than zmin.")

        return [xmin, ymin, zmin, xmax, ymax, zmax]

    @staticmethod
    def set_extent(gdf: gpd.GeoDataFrame, extent):
        """
        Clip a GeoDataFrame to a 2D or 3D extent.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing geometries to clip.
        extent : list or tuple
            - Length 4: [xmin, ymin, xmax, ymax]
            - Length 6: [xmin, ymin, zmin, xmax, ymax, zmax]

        Returns
        -------
        geopandas.GeoDataFrame
            Clipped GeoDataFrame.

        Raises
        ------
        ValueError
            If extent length is not 4 or 6.

        Warns
        -----
        UserWarning
            If a 3D extent is provided but geometries do not
            contain Z values.

        Notes
        -----
        - 2D clipping uses GeoPandas .clip() with a shapely box.
        - For 3D extents:

          1. Geometries are clipped in XY.
          2. Z filtering is applied based on bounding-box overlap.

        - Geometries are NOT sliced in Z. They are either
          retained or discarded based on Z intersection.
        """

        if len(extent) not in (4, 6):
            raise ValueError("Extent must be length 4 (2D) or 6 (3D).")

        if len(extent) == 4:
            xmin, ymin, xmax, ymax = extent
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = extent

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                "Invalid extent: xmax/xmin or ymax/ymin ordering is incorrect."
            )

        if len(extent) == 6 and zmax <= zmin:
            raise ValueError("Invalid extent: zmax must be greater than zmin.")

        bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)
        gdf_xy = gdf.clip(bbox)

        if len(extent) == 4:
            return gdf_xy

        has_z_flags = gdf_xy.geometry.apply(
            lambda geom: getattr(geom, "has_z", False) if geom else False
        )

        if not has_z_flags.any():
            warnings.warn(
                "3D extent provided but geometries do not contain Z values. "
                "Only XY clipping was applied.",
                UserWarning,
                stacklevel=2,
            )
            return gdf_xy

        def z_overlap(geom):
            if not getattr(geom, "has_z", False):
                return False

            zs = []

            def recurse(g):
                if hasattr(g, "geoms"):
                    for part in g.geoms:
                        recurse(part)
                else:
                    try:
                        coords = list(g.coords)
                    except Exception:
                        try:
                            coords = list(g.exterior.coords)
                            for ring in g.interiors:
                                coords += list(ring.coords)
                        except Exception:
                            return
                    for c in coords:
                        if len(c) == 3:
                            zs.append(c[2])

            recurse(geom)

            if not zs:
                return False

            return not (max(zs) < zmin or min(zs) > zmax)

        mask = gdf_xy.geometry.apply(z_overlap)
        return gdf_xy[mask].copy()


class Exclusions:
    """Class of functions to handle exclusion areas in a PFA area"""

    @staticmethod
    def mask_exclusion_areas(
        gdf_points, gdf_exclusion_areas, value_col="value", set_to=0
    ):
        """
        Mask points within exclusion areas by setting their values to zero.

        Parameters
        ----------
        gdf_points : geopandas.GeoDataFrame
            GeoDataFrame containing point geometries and values.
        gdf_exclusion_areas : geopandas.GeoDataFrame
            GeoDataFrame containing polygon geometries representing exclusion areas.
        value_col : str, optional
            The column name in gdf_points that contains the values to be masked, by default 'value'.

        Returns
        -------
        GeoDataFrame
            Updated GeoDataFrame with points within exclusion areas masked to zero.
        """
        # Ensure both GeoDataFrames have the same CRS
        if gdf_points.crs != gdf_exclusion_areas.crs:
            gdf_exclusion_areas = gdf_exclusion_areas.to_crs(gdf_points.crs)

        # Perform a spatial join to find points within exclusion areas
        joined = gpd.sjoin(
            gdf_points, gdf_exclusion_areas, how="left", predicate="within"
        )

        # Mask points within exclusion areas
        gdf_points.loc[~joined.index_right.isna(), value_col] = set_to

        return gdf_points

    @staticmethod
    def add_exclusions(pfa, pr_label="pr"):
        """
        Masks exclusion areas by setting probability or favorability values to a specified value (e.g., zero)
        within those areas in the provided point data.

        This function iterates through the exclusion components in the provided `pfa` object and updates
        the probability/favorability (`pr`) values by applying exclusion masks. The exclusion areas are
        defined by geometries stored in the `pfa['exclusions']` dictionary, and the function sets the
        `pr_excl` attribute in `pfa` to store the modified probability values.

        Parameters
        ----------
        pfa : dict
            A dictionary containing the exclusion components and point data, including:
            - 'exclusions': Contains the exclusion areas and the value to which `pr` should be set within
            these areas.
            - pr_label : str, optional
            The label of the probability/favorability column (default is 'pr').

        pr_label : str, optional
            The label of the probability or favorability data in the `pfa` dictionary to be updated.
            Defaults to 'pr'.

        Returns
        -------
        dict
            The updated `pfa` dictionary, where `pfa['pr_excl']` contains the probability/favorability
            values after exclusion masks have been applied.

        Notes
        ------
        - The exclusion areas are applied sequentially, with each subsequent exclusion potentially
          modifying the previously excluded points.
        - The exclusion areas are stored in shapefiles within the ``pfa['exclusions']`` structure,
          and each area is associated with a ``set_to`` value indicating what the probability/favorability
          should be set to inside the exclusion area.
        """

        c = 0
        for exclusion_component in pfa["exclusions"]["components"]:
            for layer in pfa["exclusions"]["components"][exclusion_component][
                "layers"
            ]:
                set_to = pfa["exclusions"]["components"][exclusion_component][
                    "set_to"
                ]

                # Transition approach to be cleaned once model usage is consolidated.
                shp = pfa["exclusions"]["components"][exclusion_component][
                    "layers"
                ][layer]
                if "model" in shp:
                    shp = shp["model"]
                else:
                    raise ValueError(
                        f"Exclusion layer {layer} does not contain 'model' key."
                    )

                value_col = "favorability"
                gdf_points = pfa[pr_label].copy() if c == 0 else pfa["pr_excl"]

                pfa["pr_excl"] = Exclusions.mask_exclusion_areas(
                    gdf_points=gdf_points,
                    gdf_exclusion_areas=shp,
                    value_col=value_col,
                    set_to=set_to,
                )
                c += 1
        return pfa

    @staticmethod
    def buffer_distance(
        gdf_points, gdf_exclusion_areas, buffer_distance, value_col="value"
    ):
        """
        Mask points within exclusion areas (defined by buffers around points) by setting their values to zero.

        Parameters
        ----------
        gdf_points : geopandas.GeoDataFrame
            GeoDataFrame containing point geometries and values.
        gdf_exclusion_areas : geopandas.GeoDataFrame
            GeoDataFrame containing point geometries representing exclusion areas.
        buffer_distance : float
            The distance to buffer around exclusion points.
        value_col : str, optional
            The column name in gdf_points that contains the values to be masked, by default 'value'.

        Returns
        -------
        GeoDataFrame
            Updated GeoDataFrame with points within exclusion areas masked to zero.
        """
        # Ensure both GeoDataFrames have the same CRS
        if gdf_points.crs != gdf_exclusion_areas.crs:
            gdf_exclusion_areas = gdf_exclusion_areas.to_crs(gdf_points.crs)

        # Create buffers around exclusion points
        gdf_exclusion_buffers = gdf_exclusion_areas.copy()
        gdf_exclusion_buffers["geometry"] = gdf_exclusion_areas.buffer(
            buffer_distance
        )

        # Perform a spatial join to find points within exclusion buffers
        joined = gpd.sjoin(
            gdf_points, gdf_exclusion_buffers, how="left", predicate="within"
        )

        # Ensure no duplicate indices in the joined DataFrame
        joined = joined[~joined.index.duplicated(keep="first")]

        # Create a boolean index with the correct length
        is_within = joined.index_right.notna().reindex(
            gdf_points.index, fill_value=False
        )

        # Mask points within exclusion buffers
        gdf_points.loc[is_within, value_col] = 0

        return gdf_points


class Processing:
    """Class of functions for use in processing data into models"""

    # ------ 2D-specific functions (from _Processing2D) ------

    @staticmethod
    def interpolate_points_2d(
        pfa,
        criteria,
        component,
        layer,
        nx,
        ny,
        extent=None,
        interp_method="linear",
    ):
        """Funtion to interpolate, or go from points to a 2D image.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        nx : int
            Number of points in the x-direction
        ny : int
            Number of points in the y-direction
        extent : list
            List of length 4 containing the extent (i.e., bounding box) of the gdf,
            in this order: [x_min, y_min, x_max, y_max]
        interp_method : str
            Method to use for interpolation. Can be one of: 'nearest' (KNN), 'linear',
            or 'cubic'.
        power: int
            Determines how the distance between data points affects the interpolation result.
            Specifically, it defines the rate at which the influence of a data point decreases
            as the distance from the interpolation location increases.
            Adjust power parameter accordingly to change how strongly the distance influences the
            interpolation (default is 2).


        Returns
        -------
        pfa : dict
            Updated pfa config which includes interpolation
        """
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        data_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data_col"]

        if gdf.geometry.type.iloc[0] == "Polygon":
            print(
                "Notice: interpolate_points() function recieved GeoDataFrame with geometry type 'Polygon.' Converting geometry to 'Point' geometry using centroids."
            )
            gdf.geometry = gdf.geometry.centroid

        # Extract coordinates and values from the GeoDataFrame
        x = gdf.geometry.x
        y = gdf.geometry.y
        values = gdf[data_col]

        # Define the grid for interpolation
        if extent is None:
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
        else:
            x_min, y_min, x_max, y_max = extent
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        xv, yv = np.meshgrid(x_grid, y_grid)

        # TODO: Properly implement IDW. The commented out code below does not work
        # Choose interpolation method
        # if interp_method == 'idw':
        #     # IDW interpolation inline
        #     grid = np.zeros_like(xv)
        #     distances = np.sqrt((x[:, None, None] - xv[None, :, :])**2 + (y[:, None, None] - yv[None, :, :])**2)
        #     weights = 1 / np.power(distances, power)
        #     weighted_values = weights * values[:, None, None]
        #     grid = np.sum(weighted_values, axis=0) / np.sum(weights, axis=0)
        # pygem IDW option
        # if interp_method == 'idw':
        # mesh_points = np.array([x.values, y.values, values.values])
        # idw = IDW(power)
        # idw.read_parameters('tests/test_datasets/parameters_idw_cube.prm')
        # new_mesh_points = idw(mesh_points.T)
        # grid = idw(xv.flatten(), yv.flatten()).reshape(nx, ny)
        # else:
        # Otherwise, default to scipy interpolation

        # Clean out any rows with NaNs before interpolation
        valid = ~(x.isna() | y.isna() | values.isna())
        x, y, values = x[valid], y[valid], values[valid]

        grid = scipy.interpolate.griddata(
            (x, y), values, (xv, yv), method=interp_method
        )

        # Create a new GeoDataFrame with the interpolated values
        interpolated_gdf = gpd.GeoDataFrame(
            {"value_interpolated": grid.flatten()},
            geometry=gpd.points_from_xy(xv.flatten(), yv.flatten()),
            crs=gdf.crs,
        )
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = interpolated_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "value_interpolated"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]
        return pfa

    @staticmethod
    def mark_buffer_areas(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        buffer_distance,
        polygon_value,
        buffer_value,
        background_value,
    ):
        """
        Marks grid points within a spatial extent based on their location relative to polygons and buffer areas,
        using vectorized operations with GeoPandas for performance optimization.

        This function creates a grid of points within a given spatial extent and classifies each point as either:
        - Inside a polygon (assigned `polygon_value`)
        - Inside a buffer around a polygon but outside the polygon itself (assigned `buffer_value`)
        - Outside both the polygon and its buffer (assigned `background_value`)

        The classifications are performed using vectorized operations for efficiency, and the results are stored
        in the `pfa` dictionary under the specified `criteria`, `component`, and `layer`.

        Parameters
        ----------
        pfa : dict
            A dictionary containing spatial data, including:
            - 'criteria': Holds various components and layers, where polygon and buffer data are stored.
        criteria : str
            The key for the specific criterion in the `pfa` dictionary under which the polygon data is stored.
        component : str
            The key for the specific component in the `pfa['criteria']` dictionary where the data layer is located.
        layer : str
            The key for the specific layer within the component where the polygon geometries are stored.
        extent : tuple of float
            The spatial extent in which the grid of points will be created, defined as (x_min, y_min, x_max, y_max).
        nx : int
            The number of points to generate along the x-axis within the extent.
        ny : int
            The number of points to generate along the y-axis within the extent.
        buffer_distance : float
            The distance to create buffer zones around the polygons.
        polygon_value : float
            The classification value to assign to points inside a polygon.
        buffer_value : float
            The classification value to assign to points inside a buffer but outside the polygon.
        background_value : float
            The classification value to assign to points outside both the polygon and buffer areas.

        Returns
        -------
        dict
            The updated `pfa` dictionary, where the specified layer's grid points are classified based on their
            spatial relationship to the polygons and buffers. The classification is stored in the `model` attribute
            of the layer, with the 'classification' column representing the assigned values.

        Notes
        -----
        - The function generates a grid of points within the provided extent using `numpy` and classifies the
          points based on spatial relationships to the polygons and buffers.
        - The polygon geometries and their buffers are extracted from the `pfa` dictionary and processed with
          vectorized GeoPandas operations for performance optimization.
        - The results are stored back in the `pfa` dictionary, with the classifications as part of the layer's model data.
        """

        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Create a grid of points within the spatial extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        xv, yv = np.meshgrid(x_points, y_points)
        points = np.c_[xv.ravel(), yv.ravel()]

        # Create a GeoDataFrame from the generated points
        gdf_points = gpd.GeoDataFrame(
            geometry=[shapely.geometry.Point(p) for p in points],
            crs=gdf_polygons.crs,
        )

        # Create buffer areas around polygons
        buffers = gdf_polygons.buffer(buffer_distance)

        # Vectorized operations to classify points
        gdf_points["inside_polygon"] = gdf_points.geometry.apply(
            lambda point: gdf_polygons.contains(point).any()
        )
        gdf_points["inside_buffer"] = gdf_points.geometry.apply(
            lambda point: buffers.contains(point).any()
        )

        # Assign classifications based on the spatial relationship
        gdf_points["classification"] = np.where(
            gdf_points["inside_polygon"],
            polygon_value,
            np.where(
                gdf_points["inside_buffer"], buffer_value, background_value
            ),
        )

        # Drop the temporary columns
        gdf_points = gdf_points.drop(
            columns=["inside_polygon", "inside_buffer"]
        )

        # Update the pfa dictionary with the new model
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_points
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "classification"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = f"classification ({polygon_value}=polygon, {buffer_value}=buffer, {background_value}=outside)"

        return pfa

    @staticmethod
    def polygons_to_points(pfa, criteria, component, layer, extent, nx, ny):
        """Calculate aggregated polygon values (sum or average) within a specified grid.

        Parameters
        ----------
        pfa : dict
            Configuration dictionary specifying relationships between criteria, components,
            and data layers.
        criteria : str
            Criteria associated with Polygon data.
        component : str
            Component associated with Polygon data.
        layer : str
            Layer associated with Polygon data.
        extent : list
            List of length 4 containing the extent [x_min, y_min, x_max, y_max].
        nx : int
            Number of grid cells in the x direction.
        ny : int
            Number of grid cells in the y direction.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes the aggregated polygon values as a point model.
        """
        # Extract GeoDataFrame containing polygons
        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]
        col = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data_col"]

        # Define the extent
        x_min, y_min, x_max, y_max = extent

        # Calculate cell size from nx and ny
        cell_size_x = (x_max - x_min) / nx
        cell_size_y = (y_max - y_min) / ny

        # Create a grid over the specified extent
        grid_cells = []
        for i in range(nx):
            for j in range(ny):
                x_start = x_min + i * cell_size_x
                y_start = y_min + j * cell_size_y
                grid_cell = shapely.geometry.box(
                    x_start,
                    y_start,
                    x_start + cell_size_x,
                    y_start + cell_size_y,
                )
                grid_cells.append(grid_cell)

        # Create GeoDataFrame from grid cells
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf_polygons.crs)

        # Initialize columns to store the aggregated values
        grid_gdf["sum"] = 0.0
        grid_gdf["count"] = 0

        # Use spatial indexing to speed up the intersection checks
        sindex = gdf_polygons.sindex

        # Iterate over each grid cell to calculate the aggregated values
        for grid_idx, grid_cell in grid_gdf.iterrows():
            possible_matches_index = list(
                sindex.intersection(grid_cell.geometry.bounds)
            )
            possible_matches = gdf_polygons.iloc[possible_matches_index]

            for _, polygon in possible_matches.iterrows():
                poly = polygon.geometry
                value = polygon[col]
                if poly.intersects(grid_cell.geometry):
                    intersection = poly.intersection(grid_cell.geometry)
                    intersection_area = intersection.area
                    grid_gdf.at[grid_idx, "sum"] += value * intersection_area
                    grid_gdf.at[grid_idx, "count"] += intersection_area

        # Calculate the center point of each grid cell
        points = []
        for _, row in grid_gdf.iterrows():
            centroid = row.geometry.centroid
            if row["count"] > 0:
                average_value = row["sum"] / row["count"]
            else:
                average_value = 0
            points.append((centroid, average_value))

        # Create a GeoDataFrame with point representation
        point_geometries = [
            shapely.geometry.Point(xy[0].x, xy[0].y) for xy in points
        ]
        values = [xy[1] for xy in points]

        point_gdf = gpd.GeoDataFrame(
            {"geometry": point_geometries, "value": values},
            crs=gdf_polygons.crs,
        )

        # Update the pfa dictionary with the new point representation model
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = point_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "value"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "aggregated value"

        return pfa

    @staticmethod
    def generate_grid_points(extent, nx, ny, crs):
        """
        Generate grid points (centroids) for a regular grid within a given extent.

        Args:
            extent (tuple): A tuple of (xmin, ymin, xmax, ymax) defining the bounding box.
            nx (int): Number of grid cells in the x direction.
            ny (int): Number of grid cells in the y direction.
            crs: Coordinate reference system for the GeoDataFrame.

        Returns:
            GeoDataFrame containing centroids of the grid cells.
        """
        xmin, ymin, xmax, ymax = extent

        # Generate the x and y coordinates for the grid
        x_coords = np.linspace(xmin, xmax, nx)
        y_coords = np.linspace(ymin, ymax, ny)

        # Create centroids by calculating the middle of each grid cell
        centroids = []
        for x in x_coords:
            for y in y_coords:
                centroids.append(shapely.geometry.Point(x, y))

        # Create a GeoDataFrame with the centroids
        gdf_points = gpd.GeoDataFrame(geometry=centroids, crs=crs)

        return gdf_points

    @staticmethod
    def distance_from_lines(pfa, criteria, component, layer, extent, nx, ny):
        """
        Function to calculate the minimum distance from each point in a grid
        to any LineString in the specified GeoDataFrame. The result is stored
        in `pfa` as a new GeoDataFrame with a 'distance' column.

        Parameters
        ----------
        pfa : dict
            Data structure specifying criteria, components, and layers (with
            associated GeoDataFrames). Must include a 'data' GeoDataFrame
            containing LineString geometries.
        criteria : str
            Criteria name.
        component : str
            Component name.
        layer : str
            Layer name (which holds the GeoDataFrame to measure distance from).
        extent : list
            [x_min, y_min, x_max, y_max] bounding box for the grid points.
        nx : int
            Number of points along the x-direction.
        ny : int
            Number of points along the y-direction.

        Returns
        -------
        pfa : dict
            Updated dictionary that now includes the distance model under:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['model']
        """
        gdf_linestrings = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Create a grid of points within the spatial extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        points = [
            shapely.geometry.Point(x, y) for x in x_points for y in y_points
        ]

        # Create GeoDataFrame of the grid points
        gdf_points = gpd.GeoDataFrame(geometry=points, crs=gdf_linestrings.crs)

        # Combine all LineString geometries into a single (multi-)geometry
        lines_union = gdf_linestrings.geometry.unary_union

        # Calculate distance from each point to the union of lines
        distances = [pt.distance(lines_union) for pt in gdf_points.geometry]

        # Create a new GeoDataFrame with distances
        gdf_distances = gdf_points.copy()
        gdf_distances["distance"] = distances

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_distances
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def calculate_intersections(gdf_lines):
        """
        Calculate intersection points between 2D line geometries in a GeoDataFrame.

        Intersections between LineString and MultiLineString geometries are detected.
        If the intersection geometry is:
        - Point / MultiPoint: all points are included
        - LineString / MultiLineString: the centroid of each overlapping segment is used
        - GeometryCollection: points and line-derived centroids are extracted

        Parameters
        ----------
        gdf_lines : geopandas.GeoDataFrame
            GeoDataFrame containing 2D line geometries in the 'geometry' column.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of intersection points, using the same CRS as the input.
        """
        intersections = []
        lines = gdf_lines.geometry.tolist()

        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1 :]:
                if not line1.intersects(line2):
                    continue

                intersection = line1.intersection(line2)

                if isinstance(intersection, shapely.geometry.Point):
                    intersections.append(intersection)
                elif isinstance(intersection, shapely.geometry.MultiPoint):
                    intersections.extend(list(intersection.geoms))
                elif isinstance(intersection, shapely.geometry.LineString):
                    intersections.append(intersection.centroid)
                elif isinstance(
                    intersection, shapely.geometry.MultiLineString
                ):
                    intersections.extend(
                        geom.centroid for geom in intersection.geoms
                    )
                elif isinstance(
                    intersection, shapely.geometry.GeometryCollection
                ):
                    for geom in intersection.geoms:
                        if isinstance(geom, shapely.geometry.Point):
                            intersections.append(geom)
                        elif isinstance(
                            geom,
                            (
                                shapely.geometry.LineString,
                                shapely.geometry.MultiLineString,
                            ),
                        ):
                            intersections.append(geom.centroid)

        return gpd.GeoDataFrame(geometry=intersections, crs=gdf_lines.crs)

    @staticmethod
    def vectorized_distance_calculation(
        gdf_points, tree, intersection_tree=None
    ):
        """Calculates the nearest line and intersection distances for each point in a GeoDataFrame using vectorized operations
        and spatial indexing with STRtree.

        This function computes the shortest distance from each point in the `gdf_points` GeoDataFrame to the nearest line
        in the `tree` (a spatial index of line geometries). Optionally, it can also compute the nearest distance to intersections
        (if an `intersection_tree` is provided).

        Parameters
        ----------
        gdf_points : GeoDataFrame
            A GeoDataFrame containing point geometries for which distances will be calculated.
        tree : STRtree
            A spatial index (STRtree) containing line geometries. This allows for efficient querying of the nearest line
            for each point.
        intersection_tree : STRtree, optional
            An optional spatial index (STRtree) containing intersection geometries. If provided, the function calculates
            the nearest distance to intersections as well. If not provided, the intersection distances are set to infinity.

        Returns
        -------
        tuple
            A tuple containing two pandas Series:

            - nearest_line_distances: The nearest distance from each point to the nearest line.
            - nearest_intersection_distances: The nearest distance from each point to the nearest intersection (or infinity
              if no intersection tree is provided).

        Notes
        -----
        - The function uses an inner helper `get_nearest_line_distance` to query the spatial index (`tree`) and calculate the
          distance between a point and its nearest line.
        - If no line is found for a point, the distance is set to infinity (`float('inf')`).
        - When an `intersection_tree` is provided, it computes the minimum distance between a point and the intersection geometries.
        - The function returns `float('inf')` for intersection distances if no intersections are found or if the `intersection_tree`
          is not provided.

        """

        # Function to get the nearest line distance for each point
        def get_nearest_line_distance(point):
            # Query the STRtree for the nearest lines
            nearest_lines = tree.query(point)
            if nearest_lines:  # If lines are found
                # Ensure nearest_lines are geometries, unpacking from STRtree structure if needed
                nearest_line_geom = (
                    nearest_lines[0].geometry
                    if hasattr(nearest_lines[0], "geometry")
                    else nearest_lines[0]
                )
                return point.distance(
                    nearest_line_geom
                )  # Calculate distance between point and line
            return float("inf")  # If no lines are found, return infinity

        # Calculate the nearest line distance for each point in the GeoDataFrame
        nearest_line_distances = gdf_points.geometry.apply(
            get_nearest_line_distance
        )

        # Handle intersections similarly (if available)
        if intersection_tree:
            nearest_intersection_distances = gdf_points.geometry.apply(
                lambda point: min(
                    [
                        point.distance(geom)
                        for geom in intersection_tree.query(point)
                    ],
                    default=float("inf"),
                )
            )
        else:
            nearest_intersection_distances = pd.Series(
                [float("inf")] * len(gdf_points)
            )

        return nearest_line_distances, nearest_intersection_distances

    @staticmethod
    def distance_from_lines_with_intersections(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        weight_line=1.0,
        weight_intersection=0.5,
    ):
        """
        TODO: Get this to work!!!
        Calculate the weighted distances from points to the nearest lines and intersections,
        and update the provided PFA (Potential Field Analysis) dictionary with the computed distance model.

        This function creates a grid of points over a specified spatial extent, computes the distance
        from each point to the nearest line and nearest line intersection, and then combines the distances
        using specified weights. The result is stored in the 'pfa' dictionary under the specified criteria,
        component, and layer.

        Parameters
        ----------
        pfa : dict
            The PFA (Potential Field Analysis) dictionary that contains geospatial data for various criteria,
            components, and layers.
        criteria : str
            The criteria key within the PFA dictionary to access the desired data.
        component : str
            The component key within the criteria to access the desired layer data.
        layer : str
            The layer key within the component to access the lines for distance calculation.
        extent : tuple
            A tuple (x_min, y_min, x_max, y_max) specifying the spatial extent for the grid of points.
        nx : int
            Number of points along the x-axis for the grid.
        ny : int
            Number of points along the y-axis for the grid.
        weight_line : float, optional (default=0.7)
            The weight assigned to the distance from the nearest line.
        weight_intersection : float, optional (default=0.3)
            The weight assigned to the distance from the nearest line intersection.

        Returns
        -------
        pfa : dict
            The updated PFA dictionary with a new distance model stored in the specified layer.
        """
        # Extract lines and intersections from the PFA
        gdf_lines = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]  # Lines data
        gdf_intersections = Processing.calculate_intersections(gdf_lines)

        # Define the CRS (Coordinate Reference System) to match the lines
        crs = gdf_lines.crs
        # Generate the grid of centroids within the given extent
        gdf_points = Processing.generate_grid_points(extent, nx, ny, crs)

        # Build spatial index using STRtree for lines and intersections
        tree = shapely.strtree.STRtree(gdf_lines.geometry)  # STRtree for lines
        intersection_tree = (
            shapely.strtree.STRtree(gdf_intersections.geometry)
            if gdf_intersections is not None
            else None
        )  # STRtree for intersections

        # Perform vectorized distance calculations for both lines and intersections
        line_distances, intersection_distances = (
            Processing.vectorized_distance_calculation(
                gdf_points, tree, intersection_tree
            )
        )

        # Combine the distances with weights (line and intersection distances)
        combined_distances = (
            weight_line * line_distances
            + weight_intersection * intersection_distances
        )

        # Assign combined distances to the points
        gdf_points["distance"] = combined_distances

        # Update the PFA dictionary with the calculated distances
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_points
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (weighted by line and intersection proximity)"

        return pfa

    @staticmethod
    def extrapolate_2d(
        pfa,
        criteria,
        component,
        layer,
        dataset="model",
        *,
        data_col="value_interpolated",
        training_size=0.2,
        verbose=False,
    ):
        """Function to extrapolate 2D fields to max extent grid using Gaussian Process Regression.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        dataset : str
            Layer data source for extrapolation.
        data_col : str
            Column in `dataset` to use for extrapolation input observations.
        training_size : float
            Percent of randomly select input observations to train on.
        verbose : bool
            Display training progress, assessment metrics, and final plots.
        Returns
        -------
        pfa : dict
            Updated pfa config which includes extrapolated layer.
        """

        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ][dataset]

        # drop z value if present
        gdf = drop_z_from_geometry(gdf, geom_col="geometry")

        test_size = 1 - training_size

        extrapolated_gdf = backfill_gdf(
            gdf,
            value_col=data_col,
            z_value=None,
            verbose=verbose,
            test_size=test_size,
        )

        # Update the PFA dictionary with extrapolation results
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = extrapolated_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "value_extrapolated"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]

        return pfa

    @staticmethod
    def distance_from_polygons(
        pfa, criteria, component, layer, extent, nx, ny
    ):
        """
        Calculate the true distance from a grid of points to a set of polygons,
        using Shapely's .distance() method on the union of all polygons.
        A point inside any polygon will get a distance of 0.

        Parameters
        ----------
        pfa : dict
            Configuration dictionary specifying relationships between criteria, components,
            and data layers.
        criteria : str
            Criteria associated with Polygon data to calculate distances from.
        component : str
            Component associated with Polygon data to calculate distances from.
        layer : str
            Layer associated with Polygon data to calculate distances from.
        extent : list
            [x_min, y_min, x_max, y_max] bounding box for the grid.
        nx : int
            Number of points in the x-direction.
        ny : int
            Number of points in the y-direction.

        Returns
        -------
        pfa : dict
            Updated pfa config, with a new GeoDataFrame in
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['distance_model']
            that contains the grid points plus a 'distance' column.
        """
        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Create a single geometry (union) from all polygons
        polygons_union = gdf_polygons.geometry.unary_union

        # Create a grid of points within the specified extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        points = [
            shapely.geometry.Point(x, y) for x in x_points for y in y_points
        ]

        # Convert that list of points to a GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf_polygons.crs)

        # Compute distances and store in a new GeoDataFrame
        distances = [pt.distance(polygons_union) for pt in grid_gdf.geometry]
        distance_gdf = grid_gdf.copy()
        distance_gdf["distance"] = distances

        # Save the result in the pfa dictionary (under 'distance_model')
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "distance_model"
        ] = distance_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def distance_from_points(pfa, criteria, component, layer, extent, nx, ny):
        """Function to calculate distance from Point objects.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames, particularly the gdf with Point
            geometry to calculate distances from.
        criteria : str
            Criteria associated with Point data to calculate distances from.
        component : str
            Component associated with Point data to calculate distances from.
        layer : str
            Layer associated with Point data to calculate distances from.
        extent : list
            List of length 4 containing the extent (i.e., bounding box) to use to produce the
            distance model. Can be produced using get_extent() function below. Should be in
            this order: [x_min, y_min, x_max, y_max]
        nx : int
            Number of points in the x-direction
        ny : int
            Number of points in the y-direction

        Returns
        -------
        pfa : dict
            Updated pfa config which includes interpolation
        """
        gdf_points = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Ensure we have only simple Point geometries
        points = []
        for geom in gdf_points.geometry:
            if geom.geom_type == "Point":
                points.append(geom)
            elif geom.geom_type == "MultiPoint":
                points.extend(geom.geoms)
            else:
                raise ValueError(
                    f"Unsupported geometry type: {geom.geom_type}"
                )

        if not points:
            raise ValueError(
                "No valid Point geometries found in the GeoDataFrame"
            )

        # Convert points to a numpy array of coordinates, ignoring the z-dimension if it exists
        points_coords = np.array([(point.x, point.y) for point in points])

        # Create a grid of points within the spatial extent
        x_min, y_min, x_max, y_max = extent
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        grid_points = [
            shapely.geometry.Point(x, y) for x in x_points for y in y_points
        ]

        # Create a GeoDataFrame from the generated points
        gdf_grid_points = gpd.GeoDataFrame(
            geometry=grid_points, crs=gdf_points.crs
        )

        # Convert grid points to a numpy array of coordinates
        grid_coords = np.array(
            [(point.x, point.y) for point in gdf_grid_points.geometry]
        )

        # Ensure both coordinate arrays have the same number of dimensions
        assert points_coords.shape[1] == 2, (
            "Points coordinates should have two columns (x, y)"
        )
        assert grid_coords.shape[1] == 2, (
            "Grid coordinates should have two columns (x, y)"
        )

        # Calculate distances between the grid points and the specified points
        distances = scipy.spatial.distance.cdist(grid_coords, points_coords)

        # Store minimum distances in a GeoDataFrame
        gdf_distances = gdf_grid_points.copy()
        gdf_distances["distance"] = distances.min(axis=1)

        # Update pfa dictionary with the distance model
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_distances
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def weighted_distance_from_points(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        alpha=1000.0,
        weight_points=True,
        weight_min=1.0,
        weight_max=2.0,
    ):
        """
        Compute a "weighted distance score" from a set of points,
        where each point has a data value in `data_col`. That data
        value is normalized/transformed to produce a point weight (0-1),
        then each cell's score is sum_i [ w_i * exp(-dist_i / alpha ) ].

        Parameters
        ----------
        pfa : dict
            The PFA dictionary, which must contain:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']
            ...a GeoDataFrame of points, each with a relevant 'data_col'.
        criteria : str
            E.g. 'geologic'
        component : str
            E.g. 'fluid'
        layer : str
            E.g. 'hot_springs'
        extent : list of float
            [x_min, y_min, x_max, y_max] bounding box for the grid
        nx, ny : int
            Number of points along x and y in the final grid
        alpha : float, default=1000
            Decay constant for distance (in meters).
        weight_points : bool, default=True
            Whether to apply data-based point weighting or not.
        weight_min : float, default=1.0
            Minimum weight for a transformed point.
        weight_max : float, default=2.0
            Maximum weight for a transformed point.

        Returns
        -------
        pfa : dict
            The updated dictionary, storing a new GeoDataFrame with column
            "weighted_point_score" in:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['model'].
        """

        def normalize_point_weights(values, out_min=1.0, out_max=5.0):
            # Scale a 1D array of values to a user-defined positive range [out_min..out_max].
            vals = values.copy()
            valid_mask = ~np.isnan(vals)
            valid_data = vals[valid_mask]
            if valid_data.size == 0:
                scaled = np.full_like(vals, out_min)
                return scaled
            vmin = valid_data.min()
            vmax = valid_data.max()
            rng = vmax - vmin
            if rng == 0:
                midpoint = (out_min + out_max) / 2.0
                scaled = np.full_like(vals, midpoint)
                return scaled

            # Normal minmax to [0..1] then scale to [out_min..out_max]
            mm = (valid_data - vmin) / rng
            scaled_valid = mm * (out_max - out_min) + out_min

            scaled = np.full_like(vals, np.nan)
            scaled[valid_mask] = scaled_valid

            return scaled

        layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        data_col = layer_dict.get("data_col", None)
        gdf_points = layer_dict["data"].copy()

        if data_col is None:
            print("No data_col => defaulting all weights=1.0")
            gdf_points["temp_val"] = 1.0
            data_col = "temp_val"

        # Collect geometry + data
        point_list, data_list = [], []
        for geom, val in zip(gdf_points.geometry, gdf_points[data_col]):
            if geom.geom_type == "Point":
                point_list.append(geom)
                data_list.append(val)
            elif geom.geom_type == "MultiPoint":
                for subg in geom.geoms:
                    point_list.append(subg)
                    data_list.append(val)
            else:
                raise ValueError(f"Unsupported geometry: {geom.geom_type}")

        if not point_list:
            print("No valid points found.")
            return pfa

        # Convert data_list to 1D NumPy
        data_array_1d = np.array(data_list, dtype=float)

        # Transform data values into weight
        arr_2d = data_array_1d.reshape(-1, 1)
        arr_2d = np.nan_to_num(arr_2d, nan=0)  # handle NaNs
        arr_1d = arr_2d.ravel()

        # scale them to [out_min..out_max]
        if not weight_points:
            print("User not weighting points. Defaulting all weights=1.0")
            out_min = out_max = 1.0  # all points get equal weight
        else:
            out_min = weight_min
            out_max = weight_max

        # Scale once with the chosen range
        weights_1d = normalize_point_weights(
            arr_1d,
            out_min=out_min,
            out_max=out_max,
        )

        # Create grid, compute distances
        x_min, y_min, x_max, y_max = extent
        x_coords = np.linspace(x_min, x_max, nx)
        y_coords = np.linspace(y_min, y_max, ny)
        grid_points = [
            shapely.geometry.Point(x, y) for x in x_coords for y in y_coords
        ]

        gdf_grid = gpd.GeoDataFrame(geometry=grid_points, crs=gdf_points.crs)
        point_coords = np.array([(p.x, p.y) for p in point_list])
        grid_coords = np.array([(pt.x, pt.y) for pt in gdf_grid.geometry])

        dist_matrix = scipy.spatial.distance.cdist(
            grid_coords, point_coords
        )  # shape=(nx*ny, n_points)

        # Weighted sum => sum_i [ w_i * exp(-dist_ij / alpha) ]
        decays = np.exp(-dist_matrix / alpha)
        weighted = (
            decays * weights_1d
        )  # broadcasting => shape=(n_grid, n_points)
        score_1d = np.sum(weighted, axis=1)

        gdf_grid["weighted_point_score"] = score_1d
        layer_dict["model"] = gdf_grid
        layer_dict["model_data_col"] = "weighted_point_score"
        layer_dict["model_units"] = "score (0-∞)"

        return pfa

    @staticmethod
    def point_density(
        pfa, criteria, component, layer, extent, cell_size, nx, ny
    ):
        """Calculate point density within a specified extent and return as a GeoDataFrame.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames, particularly the gdf with Point
            geometry to calculate distances from.
        criteria : str
            Criteria associated with Point data to calculate distances from.
        component : str
            Component associated with Point data to calculate distances from.
        layer : str
            Layer associated with Point data to calculate distances from.
        extent : list
            List of length 4 containing the extent (i.e., bounding box) to use to produce the
            distance model. Can be produced using get_extent() function below. Should be in
            this order: [x_min, y_min, x_max, y_max]
        cell_size : float
            Size of each cell in the grid used for density calculation.
            Example Cell Sizes for EPSG:3857
                High Detail: 50 meters
                Moderate Detail: 100 meters
                Lower Detail: 500 meters - 1 kilometer
        nx : int
            Number of cells in the x direction for the output grid.
        ny : int
            Number of cells in the y direction for the output grid.

        Returns
        -------
        density_gdf : GeoDataFrame
            GeoDataFrame populated with point density within the specified extent, with
            point geometry.
        """

        # Extract GeoDataFrame from pfa dictionary
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]

        # Define the extent
        xmin, ymin, xmax, ymax = extent

        # Create grid for density calculation using cell_size
        x_cells = int((xmax - xmin) / cell_size)
        y_cells = int((ymax - ymin) / cell_size)

        density_grid = np.zeros((y_cells, x_cells))

        for geom in gdf.geometry:
            # Check if geometry is Point or MultiPoint
            if isinstance(geom, shapely.geometry.Point):
                points = [geom]
            elif isinstance(geom, shapely.geometry.MultiPoint):
                points = geom.geoms
            else:
                points = []

            # Iterate through each point in the geometry
            for point in points:
                # Ignore the z component by only considering the x and y coordinates
                if (xmin <= point.x <= xmax) and (ymin <= point.y <= ymax):
                    x_index = int((point.x - xmin) / cell_size)
                    y_index = int((point.y - ymin) / cell_size)
                    if (x_index > x_cells - 1) | (y_index > y_cells - 1):
                        continue
                    density_grid[y_index, x_index] += 1

        # Calculate the size of each cell in the high-resolution grid
        x_step = (xmax - xmin) / nx
        y_step = (ymax - ymin) / ny

        # Initialize high-resolution density grid
        high_res_density = np.zeros((ny, nx))

        # Fill high-resolution density grid with aggregated values from density_grid
        for i in range(x_cells):
            for j in range(y_cells):
                value = density_grid[j, i]
                x_start = int(i * cell_size / x_step)
                y_start = int(j * cell_size / y_step)
                x_end = int((i + 1) * cell_size / x_step)
                y_end = int((j + 1) * cell_size / y_step)
                high_res_density[y_start:y_end, x_start:x_end] = value

        # Create points for the high-resolution grid
        points = [
            shapely.geometry.Point(
                xmin + (i + 0.5) * x_step, ymin + (j + 0.5) * y_step
            )
            for j in range(ny)
            for i in range(nx)
        ]

        # Flatten the high-resolution density grid to match the points
        densities = high_res_density.flatten()

        # Ensure the lengths match
        assert len(points) == len(densities), (
            f"Points length {len(points)} and densities length {len(densities)} do not match."
        )

        # Create the GeoDataFrame
        density_gdf = gpd.GeoDataFrame(
            {"geometry": points, "density": densities}
        )
        density_gdf = density_gdf.set_crs(gdf.crs)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = density_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "density"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "density per" + str(cell_size) + " m^2"

        return pfa

    @staticmethod
    def polygon_density(
        pfa, criteria, component, layer, extent, cell_size, nx, ny
    ):
        """Calculate polygon density within a specified extent and return as a GeoDataFrame.

        Parameters
        ----------
        pfa : dict
            Configuration specifying criteria, components, and data layers' relationship.
        criteria : str
            Criteria associated with Polygon data to calculate density from.
        component : str
            Component associated with Polygon data to calculate density from.
        layer : str
            Layer associated with Polygon data to calculate density from.
        extent : list
            List of length 4 containing the extent [x_min, y_min, x_max, y_max].
        cell_size : float
            Size of each cell in the grid used for density calculation.
        nx : int
            Number of cells in the x direction for the output grid.
        ny : int
            Number of cells in the y direction for the output grid.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes polygon density as a GeoDataFrame.
        """
        # Extract GeoDataFrame containing polygons
        gdf_polygons = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Define the extent
        xmin, ymin, xmax, ymax = extent

        # Create grid for density calculation using cell_size
        x_cells = int((xmax - xmin) / cell_size)
        y_cells = int((ymax - ymin) / cell_size)

        density_grid = np.zeros((y_cells, x_cells))

        for geom in gdf_polygons.geometry:
            # Check if geometry is Polygon or MultiPolygon
            if isinstance(geom, shapely.geometry.Polygon):
                polygons = [geom]
            elif isinstance(geom, shapely.geometry.MultiPolygon):
                polygons = geom.geoms
            else:
                polygons = []

            # Iterate through each polygon in the geometry
            for polygon in polygons:
                # Get bounding box of polygon
                bbox = polygon.bounds
                xmin_poly, ymin_poly, xmax_poly, ymax_poly = bbox

                # Calculate intersecting grid cells
                x_start = max(0, int((xmin_poly - xmin) / cell_size))
                x_end = min(x_cells, int((xmax_poly - xmin) / cell_size)) + 1
                y_start = max(0, int((ymin_poly - ymin) / cell_size))
                y_end = min(y_cells, int((ymax_poly - ymin) / cell_size)) + 1

                # Increment density count for intersecting grid cells
                density_grid[y_start:y_end, x_start:x_end] += 1

        # Calculate the size of each cell in the high-resolution grid
        x_step = (xmax - xmin) / nx
        y_step = (ymax - ymin) / ny

        # Initialize high-resolution density grid
        high_res_density = np.zeros((ny, nx))

        # Fill high-resolution density grid with aggregated values from density_grid
        for i in range(x_cells):
            for j in range(y_cells):
                value = density_grid[j, i]
                x_start = int(i * cell_size / x_step)
                y_start = int(j * cell_size / y_step)
                x_end = int((i + 1) * cell_size / x_step)
                y_end = int((j + 1) * cell_size / y_step)
                high_res_density[y_start:y_end, x_start:x_end] = value

        # Create points for the high-resolution grid
        points = [
            shapely.geometry.Point(
                xmin + (i + 0.5) * x_step, ymin + (j + 0.5) * y_step
            )
            for j in range(ny)
            for i in range(nx)
        ]

        # Flatten the high-resolution density grid to match the points
        densities = high_res_density.flatten()

        # Ensure the lengths match
        assert len(points) == len(densities), (
            f"Points length {len(points)} and densities length {len(densities)} do not match."
        )

        # Create the GeoDataFrame
        density_gdf = gpd.GeoDataFrame(
            {"geometry": points, "density": densities}
        )
        density_gdf = density_gdf.set_crs(gdf_polygons.crs)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = density_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "density"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "density per " + str(cell_size) + " m^2"

        return pfa

    @staticmethod
    def extract_fault_traces_slice(
        pfa, criteria, component, layer, min_z=None, tolerance=1e6
    ):
        """
        Extracts a 2D representation of faults by taking a slice at the bottom of the model.

        Parameters
        ----------
        gdf : GeoDataFrame
            A GeoDataFrame containing 3D fault geometries (Point geometries).
        fault_id_col : str
            The column name representing fault IDs to group points into separate faults.
        bottom_z_threshold : float
            A Z-value threshold used to select the bottom slice of the model. Points with
            Z-values within this threshold from the minimum Z will be included.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with LineString geometries representing the bottom traces of each fault.
        """
        # Extract GeoDataFrame containing 3D data
        gdf_3d = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        fault_id_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["id_col"]

        # Ensure the geometry column contains 3D Points
        if not all(gdf_3d.geometry.geom_type == "Point"):
            raise ValueError("All geometries must be Point geometries.")
        if not all(gdf_3d.geometry.apply(lambda geom: geom.has_z)):
            raise ValueError("All geometries must be 3D (with Z-coordinates).")

        if min_z is None:
            # Find the minimum Z value in the dataset
            min_z = gdf_3d.geometry.apply(lambda geom: geom.z).min()
            print(min_z)

        # Filter points at exactly min_z (or within a small tolerance)
        bottom_points = gdf_3d[
            gdf_3d.geometry.apply(
                lambda geom: abs(geom.z - min_z) <= tolerance
            )
        ]

        # Extract fault traces (LineStrings) for each unique fault ID
        fault_traces = []
        for fault_id, group in bottom_points.groupby(fault_id_col):
            # Ensure there are enough points to create a LineString
            if len(group) > 1:
                # Sort points by X and Y for proper LineString creation
                group = group.sort_values(
                    by=["geometry"],
                    key=lambda col: col.apply(lambda geom: (geom.x, geom.y)),
                )
                trace = shapely.geometry.LineString(
                    group.geometry.apply(
                        lambda geom: (geom.x, geom.y, geom.z)
                    ).tolist()
                )
                fault_traces.append({"fault_id": fault_id, "geometry": trace})

        if not fault_traces:
            raise ValueError(
                "No valid fault traces found at the minimum Z value."
            )

        # Create a GeoDataFrame from the fault traces
        fault_traces_gdf = gpd.GeoDataFrame(
            fault_traces, geometry="geometry", crs=gdf_3d.crs
        )

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "data"
        ] = fault_traces_gdf
        return pfa

    @staticmethod
    def process_faults(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        alpha_fault=1000.0,
        alpha_intersection=500.0,
        weight_fault=0.7,
        weight_intersection=0.3,
        use_intersections=True,
    ):
        """
        Process faults by:
        1) Computing distance from faults (via distance_from_lines).
        2) (Optional) Finding fault intersections, computing distance to those intersections.
        3) Applying exponential decays to fault distance and (optionally) intersection distance.
        4) Combining via a weighted sum to produce a final favorability.

        Parameters
        ----------
        pfa : dict
            The PFA dictionary containing fault data in:
            pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']
            Must have valid line geometry for distance_from_lines to work.
        criteria : str
            E.g. 'geologic'
        component : str
            E.g. 'fluid'
        layer : str
            E.g. 'intersecting_faults'
        extent : list
            [x_min, y_min, x_max, y_max] bounding box
        nx, ny : int
            Grid dimensions in x and y directions
        alpha_fault : float
            Decay constant for fault distance
        alpha_intersection : float
            Decay constant for intersection distance
        weight_fault : float
            Weight for the fault distance favorability
        weight_intersection : float
            Weight for the intersection distance favorability
        use_intersections : bool
            If False, ignore intersection distances (treat them as zero).
            If True, do the intersection calculation + decay.

        Returns
        -------
        pfa : dict
            Updated dictionary with a new 'model' containing 'favorability'
            under pfa['criteria'][criteria]['components'][component]['layers'][layer].
        """
        # Compute distance from faults
        pfa = Processing.distance_from_lines(
            pfa, criteria, component, layer, extent, nx, ny
        )
        fault_layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        fault_model_gdf = fault_layer_dict.get("model", None)
        dist_col = fault_layer_dict.get("model_data_col", None)

        if fault_model_gdf is None or dist_col != "distance":
            print(
                "ERROR: distance_from_lines did not produce a valid 'model' with 'distance' column."
            )
            return pfa

        fault_dist_1d = fault_model_gdf["distance"].to_numpy()
        try:
            fault_dist_2d = fault_dist_1d.reshape((ny, nx))
        except ValueError as e:
            print("ERROR reshaping fault distance array:", e)
            return pfa

        # find fault intersections
        gdf_lines = fault_layer_dict["data"]
        if not use_intersections or gdf_lines.empty:
            # skip intersection logic
            intersect_score_2d = np.zeros((ny, nx))
            if not use_intersections:
                print(
                    "use_intersections=False => intersection favorability is zero."
                )
            else:
                print(
                    "WARNING: No line geometry found, skipping intersections."
                )
        else:
            gdf_intersections = Processing.calculate_intersections(gdf_lines)
            if gdf_intersections.empty:
                print(
                    "No fault intersections found => intersection favorability is zero."
                )
                intersect_favor_2d = np.zeros((ny, nx))
            else:
                intersections_union = gdf_intersections.geometry.union_all()
                grid_points = Processing.generate_grid_points(
                    extent, nx, ny, crs=gdf_lines.crs
                )
                dist_intersect_1d = np.array(
                    [
                        pt.distance(intersections_union)
                        for pt in grid_points.geometry
                    ]
                )
                try:
                    dist_intersect_2d = dist_intersect_1d.reshape((ny, nx))
                except ValueError as e:
                    print("ERROR reshaping intersection distances:", e)
                    dist_intersect_2d = np.zeros((ny, nx))
                # Exponential decay for intersection distance
                intersect_score_2d = np.exp(
                    -dist_intersect_2d / alpha_intersection
                )

        # Exponential decay for fault distances
        fault_score_2d = np.exp(-fault_dist_2d / alpha_fault)

        # Combine with weights in a weighted sum
        composite_2d = (
            weight_fault * fault_score_2d
            + weight_intersection * intersect_score_2d
        )

        composite_1d = composite_2d.flatten()
        fault_model_gdf = fault_model_gdf.copy()
        fault_model_gdf["fault score"] = composite_1d

        fault_layer_dict["model"] = fault_model_gdf
        fault_layer_dict["model_data_col"] = "fault score"
        fault_layer_dict["model_units"] = "fault score (0-1)"

        return pfa

    # ------ 3D-specific functions (from _Processing3D) ------

    @staticmethod
    def interpolate_points_3d(
        pfa,
        criteria,
        component,
        layer,
        nx,
        ny,
        nz,
        extent=None,
        method="linear",
    ):
        """
        Function to interpolate, or go from points to a 3D image (voxel grid).

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        nx : int
            Number of points in the x-direction.
        ny : int
            Number of points in the y-direction.
        nz : int
            Number of points in the z-direction.
        extent : list or tuple, optional
            List of length 6 containing the 3D extent (i.e., bounding box) of the grid,
            in this order: [x_min, y_min, z_min, x_max, y_max, z_max].
            If None, the extent is inferred from the input data.
        method : str
            Method to use for interpolation. Can be one of:
            'nearest', 'linear', or 'cubic'.
            Passed directly to scipy.interpolate.griddata.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes the interpolated 3D model
            stored in the specified layer under:

            - layer["model"]
            - layer["model_data_col"] = "value_interpolated"

        Notes
        -----
        - Uses scipy.interpolate.griddata to perform interpolation.
        - Builds a full 3D meshgrid in memory before evaluation.
        - Suitable for small to moderate grid sizes.
        - For very large grids (e.g., > ~1e6 voxels), consider using
          fast_interpolate_points_3d() for improved performance
          and reduced memory usage.
        """

        total_pts = nx * ny * nz

        if total_pts > 1_000_000:
            warnings.warn(
                "Large 3D grid detected (>1e6 cells). "
                "If computation is prohibitively slow, consider "
                "fast_interpolate_points_3d() for improved "
                "performance and lower memory usage.",
                RuntimeWarning,
                stacklevel=2,
            )

        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        data_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data_col"]

        # Convert polygons → centroids if needed
        if gdf.geometry.type.iloc[0] == "Polygon":
            t1 = time.time()
            gdf.geometry = gdf.geometry.centroid

        # Extract coordinates + values
        t2 = time.time()
        x = gdf.geometry.x
        y = gdf.geometry.y
        z = gdf.geometry.apply(lambda geom: geom.z)
        values = gdf[data_col]

        # Determine grid extents
        t3 = time.time()
        if extent is None:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = extent

        # Build grid
        t4 = time.time()
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)
        xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        # Interpolation
        t5 = time.time()
        points = np.vstack((x.to_numpy(), y.to_numpy(), z.to_numpy())).T
        grid_values = griddata(
            points, values.values, (xv, yv, zv), method=method
        )

        # Construct GeoDataFrame
        t6 = time.time()
        interpolated_points = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
        interpolated_gdf = gpd.GeoDataFrame(
            {"value_interpolated": grid_values.ravel()},
            geometry=gpd.points_from_xy(
                interpolated_points[:, 0],
                interpolated_points[:, 1],
                z=interpolated_points[:, 2],
            ),
            crs=gdf.crs,
        )

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = interpolated_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "value_interpolated"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]

        return pfa

    @staticmethod
    def fast_interpolate_points_3d(
        pfa,
        criteria,
        component,
        layer,
        nx,
        ny,
        nz,
        extent=None,
        method="linear",  # "linear" or "nearest"
        build_gdf=True,  # set False to skip heavy GeoDataFrame creation
        chunk_points=2_000_000,  # max # of grid points per chunk to evaluate
        use_representative_point=True,  # faster/safer than centroid for polygons
        dtype=np.float32,  # memory saver vs float64
    ):
        """
        Function to interpolate, or go from points to a 3D image (voxel grid),
        using a memory-efficient and scalable approach.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        nx : int
            Number of points in the x-direction.
        ny : int
            Number of points in the y-direction.
        nz : int
            Number of points in the z-direction.
        extent : list or tuple, optional
            List of length 6 containing the 3D extent (i.e., bounding box) of the grid,
            in this order: [x_min, y_min, z_min, x_max, y_max, z_max].
            If None, the extent is inferred from the input data.
        method : str
            Method to use for interpolation. Can be one of:
            'nearest' or 'linear'.
            - 'nearest' uses scipy.spatial.cKDTree (fast KNN lookup).
            - 'linear' uses scipy.interpolate.LinearNDInterpolator.
        build_gdf : bool, optional
            If True (default), builds a GeoDataFrame of 3D Points for the output.
            If False, stores compact NumPy arrays instead to reduce memory usage.
        chunk_points : int, optional
            Maximum number of grid points evaluated per chunk.
            Controls memory usage during interpolation.
        use_representative_point : bool, optional
            If True, converts Polygon geometries to representative_point()
            before interpolation (faster and safer than centroid for complex shapes).
        dtype : numpy dtype, optional
            Data type used for coordinate and value arrays (default: float32).
            Lower precision reduces memory usage.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes the interpolated 3D model
            stored in the specified layer. Depending on build_gdf:

            - If True: layer["model"] is a GeoDataFrame of 3D Points.
            - If False: layer["model"] is a dictionary containing
              coordinate arrays and a 3D values array.

        Notes
        -----
        - Designed for large 3D grids where full meshgrid construction
          would be memory-intensive.
        - Performs interpolation in chunks to reduce peak memory usage.
        - Typically much faster and more scalable than interpolate_points_3d()
          for large grids.
        - Does not support cubic interpolation.
        """
        t0 = time.time()

        node = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]
        gdf = node["data"]
        data_col = node["data_col"]

        # If polygons: convert to points once
        if gdf.geometry.type.iloc[0] == "Polygon":
            print(
                "Converting polygons to points "
                f"via {'representative_point()' if use_representative_point else 'centroid'}"
            )
            if use_representative_point:
                gdf = gdf.set_geometry(gdf.geometry.representative_point())
            else:
                gdf = gdf.set_geometry(gdf.geometry.centroid)

        # Extract coordinates (vectorized for x,y; z via shapely.get_z if available)
        x = gdf.geometry.x.to_numpy(dtype=dtype, copy=False)
        y = gdf.geometry.y.to_numpy(dtype=dtype, copy=False)

        # --- z extraction (vectorized when Shapely 2 is present) ---
        geoms = gdf.geometry.to_numpy()  # a GeometryArray

        if _HAS_GET_Z:
            # Shapely 2.x: just pass the array; no ".data"
            try:
                z = get_z(geoms).astype(dtype, copy=False)
            except Exception:
                # If any geometry lacks Z, fall back gracefully
                z = np.fromiter(
                    (getattr(geom, "z", np.nan) for geom in geoms),
                    count=len(geoms),
                    dtype=dtype,
                )
        else:
            # Shapely 1.x fallback (still vector-ish, avoids .apply)
            z = np.fromiter(
                (getattr(geom, "z", np.nan) for geom in geoms),
                count=len(geoms),
                dtype=dtype,
            )

        values = gdf[data_col].to_numpy(dtype=dtype, copy=False)

        # Define grid
        if extent is None:
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            z_min, z_max = float(z.min()), float(z.max())
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = map(float, extent)

        x_grid = np.linspace(x_min, x_max, nx, dtype=dtype)
        y_grid = np.linspace(y_min, y_max, ny, dtype=dtype)
        z_grid = np.linspace(z_min, z_max, nz, dtype=dtype)

        print(f"\t\tGrid resolution: {nx} x {ny} x {nz}")

        # Construct grid coordinate arrays lazily to avoid a huge full mesh in memory
        # We'll evaluate in chunks of flattened (x,y,z) points.
        total_pts = nx * ny * nz

        # Build interpolator / index once
        t1 = time.time()
        if method == "nearest":
            tree = cKDTree(np.column_stack((x, y, z)))
            interp_obj = tree  # alias
        elif method == "linear":
            interp_obj = LinearNDInterpolator(
                np.column_stack((x, y, z)), values, fill_value=np.nan
            )
        else:
            raise ValueError(
                "method must be 'linear' or 'nearest' for this optimized version"
            )
        print(f"\t\tInterpolator built in {time.time() - t1:.3f}s")

        # Preallocate output
        out = np.empty(total_pts, dtype=dtype)

        # Helper to evaluate a chunk of linear indices
        def eval_chunk(flat_idx_slice):
            # Map flat indices -> (ix, iy, iz)
            inds = np.arange(
                flat_idx_slice.start, flat_idx_slice.stop, dtype=np.int64
            )
            iz = inds % nz
            iy = (inds // nz) % ny
            ix = inds // (ny * nz)

            X = x_grid[ix]
            Y = y_grid[iy]
            Z = z_grid[iz]
            pts = np.column_stack((X, Y, Z))

            if method == "nearest":
                dists, locs = interp_obj.query(pts, k=1, workers=-1)
                vals = values[locs].astype(dtype, copy=False)
                return vals
            vals = interp_obj(pts)
            # interp_obj returns float64 by default; cast down
            return vals.astype(dtype, copy=False)

        # Chunked evaluation to keep peak memory in check
        t2 = time.time()
        if chunk_points is None or chunk_points >= total_pts:
            out[:] = eval_chunk(slice(0, total_pts))
        else:
            start = 0
            while start < total_pts:
                end = min(start + int(chunk_points), total_pts)
                out[start:end] = eval_chunk(slice(start, end))
                start = end
        print(f"\t\tInterpolation evaluated in {time.time() - t2:.3f}s")

        # Optionally build GeoDataFrame (slow for very large grids)
        if build_gdf:
            t3 = time.time()
            # Create coordinates for geometry creation (still heavy; consider skipping for huge grids)
            xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
            interpolated_points = np.column_stack(
                (xv.ravel(), yv.ravel(), zv.ravel())
            )
            interpolated_gdf = gpd.GeoDataFrame(
                {"value_interpolated": out},
                geometry=gpd.points_from_xy(
                    interpolated_points[:, 0],
                    interpolated_points[:, 1],
                    z=interpolated_points[:, 2],
                ),
                crs=gdf.crs,
            )
            print(f"\t\tGeoDataFrame built in {time.time() - t3:.3f}s")
            model_obj = interpolated_gdf
            model_col = "value_interpolated"
        else:
            # Store compact arrays—let downstream code build a GDF only if truly needed
            model_obj = {
                "x": x_grid,
                "y": y_grid,
                "z": z_grid,
                "values": out.reshape(nx, ny, nz),
                "crs": gdf.crs,
            }
            model_col = None  # not applicable

        # Update PFA
        node["model"] = model_obj
        node["model_data_col"] = (
            model_col if model_col is not None else "values"
        )
        node["model_units"] = node["units"]

        print(f"\t\tTotal time: {time.time() - t0:.3f}s")
        return pfa

    @staticmethod
    def kriging_3d(
        pfa,
        criteria,
        component,
        layer,
        nx,
        ny,
        nz,
        extent=None,
        variogram_model="linear",
    ):
        """Function to interpolate 3D points to a 3D grid using PyKrige.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        nx, ny, nz : int
            Number of grid points in the x, y, and z directions.
        extent : list
            List of length 6 containing the 3D extent of the grid,
            in this order: [x_min, y_min, z_min, x_max, y_max, z_max].
        variogram_model : str
            Variogram model to use for kriging. Default is 'linear'.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes interpolation results.
        """
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        data_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data_col"]

        if gdf.geometry.type.iloc[0] == "Polygon":
            print(
                "Notice: interpolate_points_3d() received GeoDataFrame with geometry type 'Polygon.' Converting geometry to 'Point' geometry using centroids."
            )
            gdf.geometry = gdf.geometry.centroid

        # Extract coordinates and values from the GeoDataFrame
        x = gdf.geometry.x
        y = gdf.geometry.y
        z = gdf.geometry.apply(
            lambda geom: geom.z
        )  # Extract z directly from the geometry
        values = gdf[data_col]

        # Define the 3D grid for interpolation
        if extent is None:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = extent

        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)
        xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

        # Perform 3D Kriging interpolation
        kriging = OrdinaryKriging3D(
            x.values,
            y.values,
            z.values,
            values.values,
            variogram_model=variogram_model,
        )
        grid, _ = kriging.execute("grid", x_grid, y_grid, z_grid)

        # Create a new GeoDataFrame with the interpolated values
        interpolated_points = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
        interpolated_gdf = gpd.GeoDataFrame(
            {"value_interpolated": grid.ravel()},
            geometry=gpd.points_from_xy(
                interpolated_points[:, 0],
                interpolated_points[:, 1],
                z=interpolated_points[:, 2],
            ),
            crs=gdf.crs,
        )

        # Update the PFA dictionary with interpolation results
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = interpolated_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "value_interpolated"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]
        return pfa

    @staticmethod
    def distance_from_lines_3d(
        pfa, criteria, component, layer, extent, nx, ny, nz
    ):
        """Function to calculate distance from LineString objects in 3D.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames, particularly the gdf with LineString
            geometry to calculate distances from.
        criteria : str
            Criteria associated with LineString data to calculate distances from.
        component : str
            Component associated with LineString data to calculate distances from.
        layer : str
            Layer associated with LineString data to calculate distances from.
        extent : list
            List of length 6 containing the 3D extent (i.e., bounding box) to use to produce the
            distance model. Order: [x_min, y_min, z_min, x_max, y_max, z_max].
        nx, ny, nz : int
            Number of points in the x, y, and z directions.

        Returns
        -------
        pfa : dict
            Updated pfa config which includes distances from LineString objects.
        """
        gdf_linestrings = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["data"]

        # Define extent
        x_min, y_min, z_min, x_max, y_max, z_max = extent

        # Generate 3D grid of points
        x_points = np.linspace(x_min, x_max, nx)
        y_points = np.linspace(y_min, y_max, ny)
        z_points = np.linspace(z_min, z_max, nz)
        grid_points = [
            shapely.geometry.Point(x, y, z)
            for x in x_points
            for y in y_points
            for z in z_points
        ]

        # Create a GeoDataFrame for grid points
        gdf_points = gpd.GeoDataFrame(
            geometry=grid_points, crs=gdf_linestrings.crs
        )

        # Convert LineString geometries to 3D if not already 3D
        def ensure_3d(geom):
            if geom.has_z:
                return geom
            return shapely.geometry.LineString(
                [(x, y, 0) for x, y in geom.coords]
            )

        gdf_linestrings["geometry"] = gdf_linestrings.geometry.apply(ensure_3d)

        # Calculate 3D distances
        distances = []
        for point in gdf_points.geometry:
            min_distance = float("inf")
            for line in gdf_linestrings.geometry:
                # Compute distance between the point and the LineString in 3D
                distance = line.distance(point)
                min_distance = min(min_distance, distance)
            distances.append(min_distance)

        # Add distances to the points GeoDataFrame
        gdf_points["distance"] = distances

        # Update the PFA dictionary with the distance results
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = gdf_points
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "distance"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = "distance (m)"

        return pfa

    @staticmethod
    def distance_from_3d_solids(
        pfa,
        *,
        criteria: str,
        component: str,
        layer: str,
        extent: tuple,
        nx: int,
        ny: int,
        nz: int,
    ):
        """
        Compute a 3D distance field to the 3D Polygon geometries whose vertices have
        z-coordinates.

        For each horizontal slice:
        1. Extract 2D footprints from the 3D solids at the given z-level.
        2. Union the footprints and calculate 2D planar distances on a grid.
        3. For slices without geometry, propagate distances from the nearest valid layer
        while accounting for vertical gaps.

        The result is stored as a 3D GeoDataFrame of grid points with a 'distance' attribute,
        saved to `layer['model']` in the PFA dictionary.

        Parameters
        ----------
        pfa : dict
            PFA dictionary.
        criteria : str
            Name of the criteria level in the PFA hierarchy.
        component : str
            Name of the component within the criteria.
        layer : str
            Layer name containing the 3D solid geometries.
        extent : tuple
            Bounding box [xmin, ymin, zmin, xmax, ymax, zmax] in same CRS units.
        nx : int
            Number of grid cells in the x-direction.
        ny : int
            Number of grid cells in the y-direction.
        nz : int
            Number of grid layers in the z-direction.

        Returns
        -------
        pfa : dict
            Updated PFA dictionary with distance model.
        """
        layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        gdf3 = layer_dict["data"]

        xmin, ymin, zmin, xmax, ymax, zmax = extent
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        zs = np.linspace(zmin, zmax, nz)
        dz = zs[1] - zs[0] if nz > 1 else 0
        ztol = dz * 0.5 + 1e-9

        # initialize distance volume
        D = np.full((nz, ny, nx), np.inf)
        valid = []

        for iz, zv in enumerate(zs):
            fps = []
            for geom3d in gdf3.geometry:
                fp = Processing.slice_geometry_at_z(geom3d, zv, z_tol=ztol)
                if fp is None or fp.is_empty:
                    continue
                fps.append(fp)

            if not fps:
                continue

            uni = unary_union(fps)
            pts2 = [shapely.geometry.Point(x, y) for y in ys for x in xs]
            dist2 = np.array([pt.distance(uni) for pt in pts2]).reshape(ny, nx)
            D[iz] = dist2
            valid.append(iz)

        # vertical propagation
        if valid:
            for iz in range(nz):
                if iz not in valid:
                    nearest = min(valid, key=lambda v: abs(v - iz))
                    D[iz] = D[nearest] + abs(zs[nearest] - zs[iz])

        # flatten to GeoDataFrame
        all_pts, all_d = [], []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    all_pts.append(
                        shapely.geometry.Point(xs[ix], ys[iy], zs[iz])
                    )
                    all_d.append(D[iz, iy, ix])

        gdf_out = gpd.GeoDataFrame(
            {"distance": all_d}, geometry=all_pts, crs=gdf3.crs
        )

        layer_dict["model"] = gdf_out
        layer_dict["model_data_col"] = "distance"
        layer_dict["model_units"] = "m"
        return pfa

    @staticmethod
    def weighted_distance_from_points_3d(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nx,
        ny,
        nz,
        alpha=1000.0,
        weight_points=True,
        weight_min=1.0,
        weight_max=2.0,
        mode="max",
        k_nearest=None,
    ):
        """
        Compute a weighted influence field from 3D points over a 3D grid.

        For each voxel centroid, calculates an exponential decay function
        of distance to all 3D points, optionally weighted by transformed
        attribute values. Influence scores across points are aggregated
        using the specified mode.

        The result is stored as a 3D GeoDataFrame of voxel centroids with
        'weighted_point_score', saved to `layer['model']` in the PFA dictionary.

        Aggregation Modes
        -----------------
        - "mean" :
            Averages the influence from all points at each voxel.
            Useful for sparse datasets or when you want an overall
            picture of distributed influence.

        - "max" :
            Selects only the single strongest influence at each voxel.
            Helps highlight individual points outside clusters, avoiding
            dilution by nearby high-density areas.

        - "knearest" :
            Sums the influence from the k nearest points to each voxel.
            Provides a balance between focusing on dominant single points
            and capturing the effect of clusters. Setting k=1 behaves
            like "max"; higher values blend influence from multiple neighbors.

        Parameters
        ----------
        pfa : dict
            PFA dictionary.
        criteria : str
            Name of the criteria level in the PFA hierarchy.
        component : str
            Name of the component within the criteria.
        layer : str
            Name of the layer containing point geometries whose influence is computed.
            Attribute values for weighting are read from `layer_dict['data']` using
            the column specified in `layer_dict['data_col']` if it exists.
        extent : tuple
            Bounding box [xmin, ymin, zmin, xmax, ymax, zmax] in same CRS units.
        nx : int
            Number of grid cells in the x-direction.
        ny : int
            Number of grid cells in the y-direction.
        nz : int
            Number of grid layers in the z-direction.
        alpha : float, optional
            Decay length scale in the same units as the CRS (e.g. meters). Default is 1000.
            A larger value gives slower decay, making the influence spread farther.
        weight_points : bool, optional
            Whether to apply weighting based on attribute values. Default is True.
        weight_min : float, optional
            Minimum weight after normalization. Default is 1.0.
        weight_max : float, optional
            Maximum weight after normalization. Default is 2.0.
        mode : str, optional
            Aggregation mode: one of 'mean', 'max', or 'knearest'. Default is 'max'.
        k_nearest : int, optional
            Number of nearest neighbors to consider if mode='knearest'.

        Returns
        -------
        pfa : dict
            Updated PFA dictionary with weighted influence model.
        """

        # helper
        def normalize(vals, lo=1.0, hi=5.0):
            a = vals.astype(float)
            mask = ~np.isnan(a)
            if not mask.any():
                return np.full_like(a, lo)
            vmin, vmax = a[mask].min(), a[mask].max()
            if vmin == vmax:
                return np.full_like(a, (lo + hi) / 2.0)
            a[mask] = (a[mask] - vmin) / (vmax - vmin) * (hi - lo) + lo
            return a

        layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        gdf_pts = layer_dict["data"]
        data_col = layer_dict.get("data_col")

        # collect points and values
        pts, vals = [], []
        for geom, val in zip(
            gdf_pts.geometry,
            gdf_pts[data_col] if data_col else np.ones(len(gdf_pts)),
        ):
            if geom.geom_type == "Point":
                pts.append(geom)
                vals.append(val if data_col else 1.0)
            elif geom.geom_type == "MultiPoint":
                for sub in geom.geoms:
                    pts.append(sub)
                    vals.append(val if data_col else 1.0)

        if not pts:
            print("No valid points - pfa unchanged.")
            return pfa

        arr_vals = np.nan_to_num(np.asarray(vals, dtype=float))
        weights = (
            normalize(arr_vals, weight_min, weight_max)
            if weight_points
            else np.ones_like(arr_vals)
        )

        # build grid
        xmin, ymin, zmin, xmax, ymax, zmax = extent
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        zs = np.linspace(zmin, zmax, nz)
        grid_xyz = np.array(
            [(x, y, z) for z in zs for y in ys for x in xs]
        )  # (n_grid, 3)

        # distances & decays
        pt_xyz = np.array([(p.x, p.y, p.z) for p in pts])
        dmat = scipy.spatial.distance.cdist(
            grid_xyz, pt_xyz
        )  # (n_grid, n_pts)
        decays = np.exp(-dmat / alpha)

        # aggregation modes
        if mode == "mean":
            score = np.mean(decays * weights, axis=1)
        elif mode == "max":
            score = np.max(decays * weights, axis=1)
        elif mode == "knearest":
            if k_nearest is None:
                raise ValueError("k_nearest must be set for mode='knearest'")
            idx = np.argpartition(dmat, k_nearest, axis=1)[:, :k_nearest]
            rows = np.repeat(
                np.arange(dmat.shape[0])[:, None], k_nearest, axis=1
            )
            dk = dmat[rows, idx]
            wk = weights[idx]
            score = np.sum(np.exp(-dk / alpha) * wk, axis=1)
        else:
            raise ValueError("mode must be one of ['mean','max','knearest']")

        # store results
        gdf_grid = gpd.GeoDataFrame(
            geometry=list(starmap(shapely.geometry.Point, grid_xyz)),
            data={"weighted_point_score": score},
            crs=gdf_pts.crs,
        )
        layer_dict["model"] = gdf_grid
        layer_dict["model_data_col"] = "weighted_point_score"
        layer_dict["model_units"] = "score (0-inf)"

        return pfa

    @staticmethod
    def point_density_3d_grid(
        pfa,
        criteria,
        component,
        layer,
        extent,
        cell_size_x,
        cell_size_y,
        cell_size_z,
    ):
        """Compute point density on a 3D voxel grid.

        Counts the number of points in each cell of a grid defined by
        cell_size_x, cell_size_y, and cell_size_z within the given extent.
        Returns a GeoDataFrame of voxel centroids with density values.
        """
        # Extract GeoDataFrame from pfa dictionary
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]

        # Define the extent
        xmin, ymin, zmin, xmax, ymax, zmax = extent

        # Create grid for density calculation
        x_cells = int((xmax - xmin) / cell_size_x)
        y_cells = int((ymax - ymin) / cell_size_y)
        z_cells = int((zmax - zmin) / cell_size_z)

        density_grid = np.zeros((z_cells, y_cells, x_cells))

        # Count points in each 3D cell
        for geom in gdf.geometry:
            if isinstance(geom, shapely.geometry.Point):
                points = [geom]
            elif isinstance(geom, shapely.geometry.MultiPoint):
                points = geom.geoms
            else:
                points = []

            for point in points:
                # Check if the point is within the extent
                if (
                    (xmin <= point.x <= xmax)
                    and (ymin <= point.y <= ymax)
                    and (zmin <= point.z <= zmax)
                ):
                    # Calculate indices
                    x_index = int((point.x - xmin) / cell_size_x)
                    y_index = int((point.y - ymin) / cell_size_y)
                    z_index = int((point.z - zmin) / cell_size_z)

                    # Ensure indices are within bounds
                    if (
                        0 <= x_index < x_cells
                        and 0 <= y_index < y_cells
                        and 0 <= z_index < z_cells
                    ):
                        density_grid[z_index, y_index, x_index] += 1

        # Create a uniform grid-based GeoDataFrame
        points = []
        densities = []

        for z_idx in range(z_cells):
            for y_idx in range(y_cells):
                for x_idx in range(x_cells):
                    # Calculate centroid of each grid cell
                    x_coord = xmin + (x_idx + 0.5) * cell_size_x
                    y_coord = ymin + (y_idx + 0.5) * cell_size_y
                    z_coord = zmin + (z_idx + 0.5) * cell_size_z

                    # Append geometry and density
                    points.append(
                        shapely.geometry.Point(x_coord, y_coord, z_coord)
                    )
                    densities.append(density_grid[z_idx, y_idx, x_idx])

        # Create the GeoDataFrame
        density_gdf = gpd.GeoDataFrame(
            {"geometry": points, "density": densities}
        )
        density_gdf = density_gdf.set_crs(gdf.crs)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = density_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "density"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = f"density per {cell_size_x}x{cell_size_y}x{cell_size_z} m^3"

        return pfa

    @staticmethod
    def point_density_3d_projected(
        pfa,
        criteria,
        component,
        layer,
        extent,
        cell_size_x,
        cell_size_y,
        cell_size_z,
        nx,
        ny,
        nz,
    ):
        """Project coarse 3D point density onto a finer grid.

        First counts points on a coarse voxel grid (cell_size_*),
        then assigns those densities to a higher-resolution block
        model defined by nx, ny, nz. Each fine voxel inherits the
        density of its containing coarse cell.
        """
        # Extract GeoDataFrame from pfa dictionary
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]

        # Define the extent
        xmin, ymin, zmin, xmax, ymax, zmax = extent

        # Calculate lower-resolution grid dimensions
        x_cells = int((xmax - xmin) / cell_size_x)
        y_cells = int((ymax - ymin) / cell_size_y)
        z_cells = int((zmax - zmin) / cell_size_z)

        # Initialize lower-resolution density grid
        density_grid = np.zeros((z_cells, y_cells, x_cells))

        # Count points in each lower-resolution cell
        for geom in gdf.geometry:
            if isinstance(geom, shapely.geometry.Point):
                points = [geom]
            elif isinstance(geom, shapely.geometry.MultiPoint):
                points = geom.geoms
            else:
                points = []

            for point in points:
                # Check if the point is within the extent
                if (
                    (xmin <= point.x <= xmax)
                    and (ymin <= point.y <= ymax)
                    and (zmin <= point.z <= zmax)
                ):
                    # Calculate indices for the lower-resolution grid
                    x_index = int((point.x - xmin) / cell_size_x)
                    y_index = int((point.y - ymin) / cell_size_y)
                    z_index = int((point.z - zmin) / cell_size_z)

                    # Ensure indices are within bounds
                    if (
                        0 <= x_index < x_cells
                        and 0 <= y_index < y_cells
                        and 0 <= z_index < z_cells
                    ):
                        density_grid[z_index, y_index, x_index] += 1

        # Initialize higher-resolution grid
        higher_resolution_densities = np.zeros((nz, ny, nx))
        higher_resolution_points = []

        # Calculate higher-resolution cell sizes
        cell_size_x_hr = (xmax - xmin) / nx
        cell_size_y_hr = (ymax - ymin) / ny
        cell_size_z_hr = (zmax - zmin) / nz

        # Assign densities from lower-resolution grid to higher-resolution grid
        for z_idx_hr in range(nz):
            for y_idx_hr in range(ny):
                for x_idx_hr in range(nx):
                    # Determine the centroid of the higher-resolution cell
                    x_coord = xmin + (x_idx_hr + 0.5) * cell_size_x_hr
                    y_coord = ymin + (y_idx_hr + 0.5) * cell_size_y_hr
                    z_coord = zmin + (z_idx_hr + 0.5) * cell_size_z_hr

                    # Determine which lower-resolution cell this higher-resolution cell falls into
                    x_idx_lr = int((x_coord - xmin) / cell_size_x)
                    y_idx_lr = int((y_coord - ymin) / cell_size_y)
                    z_idx_lr = int((z_coord - zmin) / cell_size_z)

                    # Assign density from lower-resolution cell if within bounds
                    if (
                        0 <= x_idx_lr < x_cells
                        and 0 <= y_idx_lr < y_cells
                        and 0 <= z_idx_lr < z_cells
                    ):
                        density = density_grid[z_idx_lr, y_idx_lr, x_idx_lr]
                    else:
                        density = 0  # Assign zero if outside bounds

                    higher_resolution_densities[
                        z_idx_hr, y_idx_hr, x_idx_hr
                    ] = density
                    higher_resolution_points.append(
                        shapely.geometry.Point(x_coord, y_coord, z_coord)
                    )

        # Create the GeoDataFrame with the higher-resolution grid
        density_gdf = gpd.GeoDataFrame(
            {
                "geometry": higher_resolution_points,
                "density": higher_resolution_densities.flatten(),
            }
        )
        density_gdf = density_gdf.set_crs(gdf.crs)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model"
        ] = density_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_data_col"
        ] = "density"
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "model_units"
        ] = f"density per {cell_size_x}x{cell_size_y}x{cell_size_z} m^3"

        return pfa

    # ------ General processing functions (not strictly 2D or 3D) ------

    @staticmethod
    def extrude_2d_to_3d(
        pfa,
        criteria,
        component,
        layer,
        extent,
        nz,
        strike=None,
        dip=None,
        target_z_meas=None,
    ):
        """
        Extrude 2D/3D geometries into 3D solids using a 3D extent and optional dip.

        Parameters
        ----------
        pfa : dict
            Project/frame archive structure.
        criteria, component, layer : str
            Keys into pfa["criteria"][criteria]["components"][component]["layers"][layer].
        extent : list or tuple
            [xmin, ymin, zmin, xmax, ymax, zmax].
            The smallest of (zmin, zmax) is treated as the bottom,
            the largest as the top of the extrusion. x/y are used to clip geometry.
        nz : int
            Number of vertical layers (stored as metadata).
        strike : float, optional
            Global strike azimuth in degrees, clockwise from North.
            If None or used with dip=None, extrusion is vertical.
        dip : float, optional
            Global dip angle in degrees from horizontal (0 to 90).
            If None or dip ~ 90°, extrusion is vertical.
        target_z_meas : any, optional
            Stored in layer_dict["z_meas"] for downstream use.
        """

        # Unpack extent (note: your convention is [xmin, ymin, zmin, xmax, ymax, zmax])
        xmin, ymin, z0, xmax, ymax, z1 = extent

        # Enforce consistent vertical ordering
        z_min = min(z0, z1)
        z_max = max(z0, z1)

        # Build XY clipping box
        xy_box = shapely.geometry.box(xmin, ymin, xmax, ymax)

        layer_dict = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]
        gdf2 = layer_dict["data"]

        # Backup original data
        layer_dict["old_data"] = gdf2.copy()
        layer_dict["z_meas"] = target_z_meas

        def _dip_offset(z_top, z_bot, strike_deg, dip_deg):
            """
            Compute (dx, dy) offset for the bottom trace along dip direction over
            the vertical span from z_top to z_bot.
            """
            if strike_deg is None or dip_deg is None:
                return 0.0, 0.0

            # Treat near-vertical as vertical extrusion
            if dip_deg >= 89.9:
                return 0.0, 0.0

            dz = z_top - z_bot  # positive if z_top > z_bot
            if dz == 0:
                return 0.0, 0.0

            dip_rad = math.radians(dip_deg)
            # tan(dip) = vertical / horizontal  =>  horizontal = vertical / tan(dip)
            horiz = abs(dz) / math.tan(dip_rad)

            strike_rad = math.radians(strike_deg)
            dip_az = strike_rad + math.pi / 2.0  # dip direction

            # azimuth convention: x = East, y = North
            dx = horiz * math.sin(dip_az)
            dy = horiz * math.cos(dip_az)
            return dx, dy

        # Global offset for bottom surfaces (same z_min/z_max for all features)
        dx_dip, dy_dip = _dip_offset(z_max, z_min, strike, dip)

        # Helper to safely get x,y from a coordinate (supports 2D or 3D coords)
        def _xy(coord):
            return coord[0], coord[1]

        geoms3 = []
        for geom in gdf2.geometry:
            if geom.is_empty:
                continue

            # Clip to XY box first (works for 2D or 3D, z is dropped by intersection)
            geom_clipped = geom.intersection(xy_box)
            if geom_clipped.is_empty:
                continue

            # Lines → fault walls
            if geom_clipped.geom_type in ("LineString", "MultiLineString"):
                parts = (
                    geom_clipped.geoms
                    if geom_clipped.geom_type == "MultiLineString"
                    else [geom_clipped]
                )
                for line in parts:
                    coords = [_xy(c) for c in line.coords]

                    # Top trace at z_max
                    top = [(x, y, z_max) for x, y in coords]

                    # Bottom trace at z_min, shifted along dip direction
                    bot = [
                        (x + dx_dip, y + dy_dip, z_min)
                        for x, y in reversed(coords)
                    ]

                    ring = top + bot
                    geoms3.append(shapely.geometry.Polygon(ring))

            # Polygons → 3D prisms (bottom offset along dip)
            elif geom_clipped.geom_type in ("Polygon", "MultiPolygon"):
                polys = (
                    geom_clipped.geoms
                    if geom_clipped.geom_type == "MultiPolygon"
                    else [geom_clipped]
                )
                for poly in polys:
                    ext = [_xy(c) for c in poly.exterior.coords]

                    top_ext = [(x, y, z_max) for x, y in ext]
                    bot_ext = [
                        (x + dx_dip, y + dy_dip, z_min)
                        for x, y in reversed(ext)
                    ]

                    holes3 = []
                    for hole in poly.interiors:
                        hc = [_xy(c) for c in hole.coords]
                        top_h = [(x, y, z_max) for x, y in hc]
                        bot_h = [
                            (x + dx_dip, y + dy_dip, z_min)
                            for x, y in reversed(hc)
                        ]
                        holes3.append(top_h + bot_h)

                    geoms3.append(
                        shapely.geometry.Polygon(
                            top_ext + bot_ext, holes=holes3
                        )
                    )

            # Skip points and unsupported geometry types
            else:
                continue

        gdf3 = gpd.GeoDataFrame(geometry=geoms3, crs=gdf2.crs)

        # Mark how it was extruded
        gdf3.attrs["extruded"] = True
        gdf3.attrs["z_min"] = z_min
        gdf3.attrs["z_max"] = z_max
        gdf3.attrs["nz"] = nz
        if strike is not None:
            gdf3.attrs["strike"] = strike
        if dip is not None:
            gdf3.attrs["dip"] = dip

        # Replace layer data with 3D solids
        layer_dict["data"] = gdf3
        layer_dict["data_col"] = "None"
        layer_dict["units"] = ""
        layer_dict["z_meas"] = target_z_meas

        return pfa

    @staticmethod
    def create_fault_surfaces_from_points(gdf_points, fault_number_col):
        """Create surfaces representing faults from point data.

        Parameters
        ----------
        gdf_points : GeoDataFrame
            GeoDataFrame containing point data with x, y, z coordinates and fault numbers.
        fault_number_col : str
            Column name in the GeoDataFrame that contains fault numbers.

        Returns
        -------
        gdf_surfaces : GeoDataFrame
            GeoDataFrame containing surfaces (Polygons or MultiPolygons) for each fault.
        """
        fault_surfaces = []

        # Group points by fault number
        grouped = gdf_points.groupby(fault_number_col)

        for fault_number, group in grouped:
            points = group.geometry
            coords = [(point.x, point.y, point.z) for point in points]

            # Create a surface (e.g., convex hull) for the fault
            try:
                surface = shapely.geometry.MultiPoint(coords).convex_hull
                fault_surfaces.append(
                    {"fault_number": fault_number, "geometry": surface}
                )
            except Exception as e:
                print(f"Error creating surface for fault {fault_number}: {e}")

        # Create GeoDataFrame for surfaces
        gdf_surfaces = gpd.GeoDataFrame(fault_surfaces)
        gdf_surfaces = gdf_surfaces.set_crs(gdf_points.crs)

        return gdf_surfaces

    @staticmethod
    def slice_geometry_at_z(geom3d, z, z_tol=1e-8):
        """
        Extract the 2D footprint of a 3D polygon geometry at a specific elevation.

        If the specified z-level lies outside the vertical extent of the geometry,
        returns None. Degenerate or invalid polygon slices are converted to their
        boundary LineString.

        Parameters
        ----------
        geom3d : shapely Polygon
            3D solid polygon geometry with z-coordinates in its vertices.
        z : float
            Target elevation for slicing the geometry.
        z_tol : float, optional
            Vertical tolerance for including slices near the target elevation.

        Returns
        -------
        shapely geometry or None
            2D Polygon or LineString footprint at elevation z, or None if outside range.
        """
        if not hasattr(geom3d, "exterior"):
            return None
        coords3 = list(geom3d.exterior.coords)
        zs = [c[2] for c in coords3]
        zlo, zhi = min(zs), max(zs)
        if z < zlo - z_tol or z > zhi + z_tol:
            return None
        xy = [(x, y) for (x, y, _) in coords3]

        # attempt full polygon
        try:
            fp = shapely.geometry.Polygon(xy)
        except Exception:
            fp = shapely.geometry.LineString(xy)

        # convert zero-area or invalid polygons to boundary
        if fp.geom_type == "Polygon" and (fp.area == 0 or not fp.is_valid):
            fp = fp.boundary

        return fp

    @staticmethod
    def convert_3d_to_2d(pfa, criteria, component, layer):
        """
        Convert a 3D GeoDataFrame layer to a 2D representation by collapsing along the Z-dimension.

        This function processes a 3D layer stored in the `pfa` dictionary structure, filtering out empty geometries,
        sorting the data by X, Y, and Z coordinates, and aggregating data by grouping on X and Y. The resulting
        2D GeoDataFrame replaces the original 3D data in the `pfa` dictionary.

        Parameters
        ----------
        pfa : dict
            A nested dictionary containing geospatial data organized by criteria, components, and layers.
            The specific layer to be processed is accessed using the given `criteria`, `component`, and `layer` keys.
        criteria : str
            The key in `pfa['criteria']` identifying the specific criteria containing the 3D layer.
        component : str
            The key in `pfa['criteria'][criteria]['components']` identifying the specific component containing the 3D layer.
        layer : str
            The key in `pfa['criteria'][criteria]['components'][component]['layers']` identifying the specific 3D layer.

        Returns
        -------
        dict
            The updated `pfa` dictionary with the 3D layer converted to a 2D representation. The resulting 2D GeoDataFrame
            is stored in `pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']`.

        Raises
        ------
        ValueError
            If the `geometry` column in the GeoDataFrame is not of type `Point` or contains invalid geometries.

        Notes
        -----
        - Empty geometries (e.g., `POINT EMPTY`) are filtered out before processing.
        - Sorting is performed by X, Y, and Z coordinates.
        - Aggregation sums the values in the specified data column while keeping a representative geometry for each (X, Y) pair.
        - The units of the data are updated to reflect the aggregation performed.
        """
        # Extract GeoDataFrame containing 3D data
        gdf_3d = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        col = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data_col"]
        units = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["units"]
        crs = gdf_3d.crs
        # Ensure the geometry column is of Point type
        if not gdf_3d.geometry.geom_type.isin(["Point"]).all():
            raise ValueError("Geometry column must contain Point geometries.")

        # Filter out empty geometries
        gdf_3d = (
            gdf_3d[~gdf_3d.geometry.is_empty].copy().reset_index(drop=True)
        )

        gdf_3d["x"] = gdf_3d.geometry.x
        gdf_3d["y"] = gdf_3d.geometry.y
        gdf_3d["z"] = gdf_3d.geometry.z

        # Sort geometries by X, Y, Z
        gdf_3d = gdf_3d.sort_values(by=["x", "y", "z"]).reset_index(drop=True)

        # Step 1: Aggregate by unique (X, Y, Z) points
        gdf_3d = gdf_3d.groupby(["x", "y", "z"], as_index=False).agg(
            {
                "geometry": "first",  # Keep the representative geometry
                col: "sum",  # Sum the data column for unique (X, Y, Z)
            }
        )
        gdf_3d = gdf_3d.set_geometry("geometry")

        # Step 2: Collapse Z dimension to (X, Y)
        gdf_2d = gdf_3d.groupby(["x", "y"], as_index=False).agg(
            {
                "geometry": "first",  # Keep the representative geometry
                col: "sum",  # Sum across Z for unique (X, Y)
            }
        )
        gdf_2d = gdf_2d.set_geometry("geometry")
        gdf_2d.set_crs(crs, inplace=True)

        # Update the pfa dictionary with the new layer
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "data"
        ] = gdf_2d
        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "units"
        ] = "summed " + units
        return pfa

    @staticmethod
    def extract_fault_traces_top(pfa, criteria, component, layer):
        """
        Create a 2D representation of 3D faults by extracting the trace at the top of each fault.

        Parameters
        ----------
        gdf_3d : GeoDataFrame
            A GeoDataFrame containing 3D fault data with geometries (Polygons or MultiPolygons).
        fault_id_col : str
            The name of the column in `gdf_3d` that uniquely identifies faults.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing 2D fault traces (LineStrings) for the top of each fault.

        Notes
        -----
        - The Z-coordinate is extracted for the "top" of each fault.
        - The resulting GeoDataFrame is 2D (ignoring Z-coordinates in the output geometries).
        """
        # Extract GeoDataFrame containing 3D data
        gdf_3d = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        fault_id_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["id_col"]

        # Check if the geometry is 3D
        if not gdf_3d.geometry.apply(lambda geom: geom.has_z).all():
            raise ValueError(
                "All geometries in the GeoDataFrame must have Z-coordinates."
            )
        # Extract Z-coordinates as a separate column
        gdf_3d["z"] = gdf_3d.geometry.apply(lambda geom: geom.z)

        # Extract the top trace for each fault
        fault_traces = []
        for fault_id, group in gdf_3d.groupby(fault_id_col):
            # Sort by Z to get the topmost points
            top_points = group.sort_values(by="z", ascending=False)

            # Create a LineString from the topmost points
            trace = shapely.geometry.LineString(
                [
                    (point.geometry.x, point.geometry.y)
                    for _, point in top_points.iterrows()
                ]
            )

            fault_traces.append({"fault_id": fault_id, "geometry": trace})

        # Create a new GeoDataFrame with the 2D fault traces
        gdf_2d = gpd.GeoDataFrame(
            fault_traces, geometry="geometry", crs=gdf_3d.crs
        )

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "data"
        ] = gdf_2d
        return pfa

    @staticmethod
    def extract_fault_traces_bottom(
        pfa, criteria, component, layer, min_z=None, tolerance=1e6
    ):
        """
        Extracts a 2D representation of faults by taking a slice at the bottom of the model.

        Parameters
        ----------
        gdf : GeoDataFrame
            A GeoDataFrame containing 3D fault geometries (Point geometries).
        fault_id_col : str
            The column name representing fault IDs to group points into separate faults.
        bottom_z_threshold : float
            A Z-value threshold used to select the bottom slice of the model. Points with
            Z-values within this threshold from the minimum Z will be included.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with LineString geometries representing the bottom traces of each fault.
        """
        # Extract GeoDataFrame containing 3D data
        gdf_3d = pfa["criteria"][criteria]["components"][component]["layers"][
            layer
        ]["data"]
        fault_id_col = pfa["criteria"][criteria]["components"][component][
            "layers"
        ][layer]["id_col"]

        # Ensure the geometry column contains 3D Points
        if not all(gdf_3d.geometry.geom_type == "Point"):
            raise ValueError("All geometries must be Point geometries.")
        if not all(gdf_3d.geometry.apply(lambda geom: geom.has_z)):
            raise ValueError("All geometries must be 3D (with Z-coordinates).")

        if min_z is None:
            # Find the minimum Z value in the dataset
            min_z = gdf_3d.geometry.apply(lambda geom: geom.z).min()
            print(min_z)

        # Filter points at exactly min_z (or within a small tolerance)
        bottom_points = gdf_3d[
            gdf_3d.geometry.apply(
                lambda geom: abs(geom.z - min_z) <= tolerance
            )
        ]

        # Extract fault traces (LineStrings) for each unique fault ID
        fault_traces = []
        for fault_id, group in bottom_points.groupby(fault_id_col):
            # Ensure there are enough points to create a LineString
            if len(group) > 1:
                # Sort points by X and Y for proper LineString creation
                group = group.sort_values(
                    by=["geometry"],
                    key=lambda col: col.apply(lambda geom: (geom.x, geom.y)),
                )
                trace = shapely.geometry.LineString(
                    group.geometry.apply(
                        lambda geom: (geom.x, geom.y, geom.z)
                    ).tolist()
                )
                fault_traces.append({"fault_id": fault_id, "geometry": trace})

        if not fault_traces:
            raise ValueError(
                "No valid fault traces found at the minimum Z value."
            )

        # Create a GeoDataFrame from the fault traces
        fault_traces_gdf = gpd.GeoDataFrame(
            fault_traces, geometry="geometry", crs=gdf_3d.crs
        )

        pfa["criteria"][criteria]["components"][component]["layers"][layer][
            "data"
        ] = fault_traces_gdf
        return pfa

    @staticmethod
    def interpolate_points(
        pfa,
        criteria,
        component,
        layer,
        nx,
        ny,
        nz=None,
        extent=None,
        interp_method="linear",
    ):
        """
        Function to interpolate, or go from points to either a 2D image
        or a 3D image (voxel grid), depending on user input.

        This function serves as a unified interface for interpolation.
        If `nz` is provided, 3D interpolation is performed.
        If `nz` is None, 2D interpolation is performed.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames.
        criteria : str
            Criteria associated with data to interpolate.
        component : str
            Component associated with data to interpolate.
        layer : str
            Layer associated with data to interpolate.
        nx : int
            Number of points in the x-direction.
        ny : int
            Number of points in the y-direction.
        nz : int, optional
            Number of points in the z-direction.
            If provided, 3D interpolation is performed.
            If None (default), 2D interpolation is performed.
        extent : list or tuple, optional
            Spatial extent defining the interpolation grid.

            - For 2D interpolation: length 4
              [x_min, y_min, x_max, y_max]
            - For 3D interpolation: length 6
              [x_min, y_min, z_min, x_max, y_max, z_max]

            If None, the extent is inferred from the input data.
        interp_method : str
            Method to use for interpolation.
            For 2D: supports 'nearest', 'linear', or 'cubic'.
            For 3D: supported methods depend on the underlying 3D function
            (typically 'nearest' or 'linear').

        Returns
        -------
        pfa : dict
            Updated pfa config which includes the interpolated model
            stored in the specified layer.

        Notes
        -----
        - This function dispatches internally to either
          `interpolate_points_2d()` or `interpolate_points_3d()`.
        - Dimensionality is determined solely by whether `nz` is provided.
        - For large 3D grids, consider using
          `fast_interpolate_points_3d()` directly for improved
          performance and memory efficiency.
        """

        if nz is not None:
            if extent is not None and len(extent) != 6:
                raise ValueError(
                    "3D interpolation requires extent of length 6 "
                    "[xmin, ymin, zmin, xmax, ymax, zmax]"
                )

            return Processing.interpolate_points_3d(
                pfa,
                criteria,
                component,
                layer,
                nx,
                ny,
                nz=nz,
                extent=extent,
                method=interp_method,
            )

        if extent is not None and len(extent) != 4:
            raise ValueError(
                "2D interpolation requires extent of length 4 "
                "[xmin, ymin, xmax, ymax]"
            )

        return Processing.interpolate_points_2d(
            pfa,
            criteria,
            component,
            layer,
            nx,
            ny,
            extent=extent,
            interp_method=interp_method,
        )
