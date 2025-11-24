"""
Set of interp_methods to process data from various formats into 2d images.
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

        Parameters:
            gdf (GeoDataFrame): GeoDataFrame containing point geometry and Z values in the geometry.
            z_meas (str): Current measurement reference (e.g., 'm-msl', 'epsg:####', or 'ft-msl').
            target_z_meas (str): Target measurement reference (e.g., 'm-msl', 'epsg:####', or 'ft-msl').

        Returns:
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
        data : Pandas Series
            Series of data values to filter
        quantile : int
            Number representing the quantile, which when exceeded, produces an outlier.

        Returns
        -------
        data : Pandas Series
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
        series : Pandas Series
            Series of data values to filter.
        quantile : float
            Number representing the quantile, which when exceeded, produces an outlier.

        Returns
        -------
        series : Pandas Series
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
        gdf : GeoDataFrame
            GeoDataFrame containing the data to filter.
        column : list of str
            Column name to apply the filter to.
        quantile : float
            Number representing the quantile, which when exceeded, produces an outlier.

        Returns
        -------
        gdf_filtered : GeoDataFrame
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
    def get_extent_3d(gdf: gpd.GeoDataFrame):
        """
        Get extent (i.e., bounding box) of a set of 3D points or polygons.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of Point, LineString, or Polygon geometry type to get extent from.
            Geometries should include z-coordinates for 3D bounding box calculation.

        Returns
        -------
        extent : list
            List of length 6 containing the extent (i.e., bounding box) of the gdf,
            in this order: [x_min, y_min, z_min, x_max, y_max, z_max]

        Notes
        -----
        - If the geometry type is Polygon or LineString, the function uses the total bounds and
        calculates the min/max z-coordinate separately.
        - If the geometry type is Point, the function uses the coordinates directly.
        """

        # Ensure the GeoDataFrame contains 3D geometries
        if not gdf.geometry.iloc[0].has_z:
            raise ValueError(
                "The provided GeoDataFrame does not contain 3D geometries."
            )

        # Extract bounds
        if gdf.geometry.iloc[0].geom_type == "Point":
            xmin = gdf.geometry.x.min()
            xmax = gdf.geometry.x.max()
            ymin = gdf.geometry.y.min()
            ymax = gdf.geometry.y.max()
            zmin = gdf.geometry.z.min()
            zmax = gdf.geometry.z.max()
        elif gdf.geometry.iloc[0].geom_type in {"Polygon", "LineString"}:
            xmin, ymin, xmax, ymax = gdf.total_bounds
            zmin = gdf.geometry.apply(
                lambda geom: min(coord[2] for coord in geom.coords)
            ).min()
            zmax = gdf.geometry.apply(
                lambda geom: max(coord[2] for coord in geom.coords)
            ).max()
        else:
            raise TypeError(
                "Unsupported geometry type. The GeoDataFrame should contain Points, Polygons, or LineStrings."
            )

        extent = [xmin, ymin, zmin, xmax, ymax, zmax]
        return extent

    @staticmethod
    def set_extent(gdf, extent):
        """Clip a GeoPandas DataFrame to a specified extent.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The GeoDataFrame to be clipped. It can contain any geometry type (e.g., Point, LineString, Polygon).
        extent : list or tuple
            The extent to clip to, specified as [xmin, ymin, xmax, ymax].

        Returns
        -------
        geopandas.GeoDataFrame
            A new GeoDataFrame clipped to the specified extent.
        """
        # Create a bounding box from the extent
        bbox = shapely.geometry.box(extent[0], extent[1], extent[2], extent[3])
        # Clip the GeoDataFrame using the bounding box
        gdf_clipped = gdf.clip(bbox)
        return gdf_clipped

    @staticmethod
    def set_extent_3d(gdf: gpd.GeoDataFrame, extent):
        """Clip 3D geometries in a GeoDataFrame to a 3D box extent.

        Points are clipped in true 3D. Other geometries are filtered
        by their XY and Z bounding boxes, but not sliced.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing 3D geometries.
        extent : list
            Length-6 list specifying the clip extent:
            [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with geometries filtered to the 3D extent.
        """
        xmin, ymin, zmin, xmax, ymax, zmax = extent

        clipped_geometries = []
        indices = []

        for idx, geom in gdf.geometry.items():
            if geom.is_empty:
                continue

            # Points
            if isinstance(geom, Point):
                z = geom.z if geom.has_z else 0
                if (
                    xmin <= geom.x <= xmax
                    and ymin <= geom.y <= ymax
                    and zmin <= z <= zmax
                ):
                    clipped_geometries.append(geom)
                    indices.append(idx)

            # everything else
            else:
                try:
                    xmin_geom, ymin_geom, xmax_geom, ymax_geom = geom.bounds

                    inside_xy = (
                        xmin_geom >= xmin
                        and xmax_geom <= xmax
                        and ymin_geom >= ymin
                        and ymax_geom <= ymax
                    )
                except:
                    inside_xy = False

                # compute z range
                zs = []
                try:
                    zs = [c[2] for c in geom.exterior.coords]
                    for ring in geom.interiors:
                        zs.extend([c[2] for c in ring.coords])
                except AttributeError:
                    try:
                        zs = [c[2] for c in geom.coords]
                    except AttributeError:
                        try:
                            for part in geom.geoms:
                                try:
                                    zs.extend(
                                        [c[2] for c in part.exterior.coords]
                                    )
                                except AttributeError:
                                    zs.extend([c[2] for c in part.coords])
                        except AttributeError:
                            zs = [0]

                if not zs:
                    zs = [0]

                min_z = min(zs)
                max_z = max(zs)

                # check if geometry is fully outside Z box
                if (max_z < zmin) or (min_z > zmax):
                    continue

                # test if geometry is entirely inside XY box
                if inside_xy:
                    clipped_geom = geom
                else:
                    continue

                if not clipped_geom.is_empty:
                    clipped_geometries.append(clipped_geom)
                    indices.append(idx)

        if indices:
            gdf_clipped = gdf.loc[indices].copy()
            gdf_clipped["geometry"] = clipped_geometries
        else:
            gdf_clipped = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

        return gdf_clipped


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
        gdf_points : GeoDataFrame
            GeoDataFrame containing point geometries and values.
        gdf_exclusion_areas : GeoDataFrame
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

        Parameters:
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

        Returns:
        -------
        dict
            The updated `pfa` dictionary, where `pfa['pr_excl']` contains the probability/favorability
            values after exclusion masks have been applied.

        Notes:
        ------
        - The exclusion areas are applied sequentially, with each subsequent exclusion potentially
        modifying the previously excluded points.
        - The exclusion areas are stored in shapefiles within the `pfa['exclusions']` structure,
        and each area is associated with a `set_to` value indicating what the probability/favorability
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
        gdf_points : GeoDataFrame
            GeoDataFrame containing point geometries and values.
        gdf_exclusion_areas : GeoDataFrame
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
        """Optimized 3D interpolation using Scipy's griddata (with timing checkpoints)."""

        start_total = time.time()

        t0 = time.time()
        gdf = pfa["criteria"][criteria]["components"][component]["layers"][layer]["data"]
        data_col = pfa["criteria"][criteria]["components"][component]["layers"][layer]["data_col"]
        print(f"[Time] Load layer data: {time.time() - t0:.4f} s")

        # Convert polygons → centroids if needed
        if gdf.geometry.type.iloc[0] == "Polygon":
            t1 = time.time()
            gdf.geometry = gdf.geometry.centroid
            print(
                f"[Time] Polygon → centroid conversion: {time.time() - t1:.4f} s"
            )

        # Extract coordinates + values
        t2 = time.time()
        x = gdf.geometry.x
        y = gdf.geometry.y
        z = gdf.geometry.apply(lambda geom: geom.z)
        values = gdf[data_col]
        print(f"[Time] Extract coordinates & values: {time.time() - t2:.4f} s")

        # Determine grid extents
        t3 = time.time()
        if extent is None:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = extent
        print(f"[Time] Compute extents: {time.time() - t3:.4f} s")

        # Build grid
        t4 = time.time()
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)
        xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        print(f"[Time] Build grid (meshgrid): {time.time() - t4:.4f} s")
        print(f"\t Grid resolution: {nx} x {ny} x {nz}")

        # Interpolation
        t5 = time.time()
        points = np.vstack((x.to_numpy(), y.to_numpy(), z.to_numpy())).T
        grid_values = griddata(
            points, values.values, (xv, yv, zv), method=method
        )
        print(f"[Time] Interpolation (griddata): {time.time() - t5:.4f} s")

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
        print(
            f"[Time] Build interpolated GeoDataFrame: {time.time() - t6:.4f} s"
        )

        # Store results
        t7 = time.time()
        pfa["criteria"][criteria]["components"][component]["layers"][layer]["model"] = interpolated_gdf
        pfa["criteria"][criteria]["components"][component]["layers"][layer]["model_data_col"] = "value_interpolated"
        pfa["criteria"][criteria]["components"][component]["layers"][layer]["model_units"] = \
            pfa["criteria"][criteria]["components"][component]["layers"][layer]["units"]
        print(f"[Time] Update PFA structure: {time.time() - t7:.4f} s")

        # Total runtime
        print(
            f"[TOTAL] interpolate_points_3d completed in {time.time() - start_total:.4f} s"
        )

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
        method="linear",             # "linear" or "nearest"
        build_gdf=True,              # set False to skip heavy GeoDataFrame creation
        chunk_points=2_000_000,      # max # of grid points per chunk to evaluate
        use_representative_point=True,  # faster/safer than centroid for polygons
        dtype=np.float32,            # memory saver vs float64
    ):
        """
        Faster 3D interpolation with lower memory usage.

        - method="nearest": uses cKDTree (very fast).
        - method="linear": uses LinearNDInterpolator + chunked evaluation.
        - build_gdf=False avoids creating millions of shapely Points (big speed win).
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

    # Define a helper function to classify each point
    @staticmethod
    def classify_point(args):
        """
        Classifies a point based on its location relative to a series of polygons and their corresponding buffers.

        This function checks if a point is inside a polygon or within the buffer surrounding the polygon.
        - If the point is inside the polygon, it returns the `polygon_value`.
        - If the point is not inside the polygon but is within the buffer area, it returns the `buffer_value`.
        - If the point is outside both the polygon and buffer, it returns a default value of 1.0.

        Parameters:
        ----------
        args : tuple
            A tuple containing the following elements:
            - point : shapely.geometry.Point
                The point to be classified.
            - polygons : list of shapely.geometry.Polygon
                A list of polygons to check for point containment.
            - buffers : list of shapely.geometry.Polygon
                A list of buffer polygons corresponding to each polygon in `polygons`.
            - polygon_value : float
                The value to return if the point is inside a polygon.
            - buffer_value : float
                The value to return if the point is inside a buffer but outside the polygon.

        Returns:
        -------
        float
            The classification value based on the point's location:
            - `polygon_value` if the point is inside a polygon.
            - `buffer_value` if the point is inside a buffer but outside the polygon.
            - 1.0 if the point is outside both the polygon and buffer.

        Notes:
        ------
        - The function assumes that the `polygons` and `buffers` lists are of the same length and that each buffer corresponds to the polygon at the same index.
        - It stops and returns a value as soon as the point is classified within a polygon or buffer.
        """

        point, polygons, buffers, polygon_value, buffer_value = args
        for polygon, buffer in zip(polygons, buffers):
            if polygon.contains(point):  # Inside the polygon
                return polygon_value
            if buffer.contains(
                point
            ):  # Inside the buffer but outside the polygon
                return buffer_value
        return 1.0  # Outside both polygon and buffer

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

        Parameters:
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

        Returns:
        -------
        dict
            The updated `pfa` dictionary, where the specified layer's grid points are classified based on their
            spatial relationship to the polygons and buffers. The classification is stored in the `model` attribute
            of the layer, with the 'classification' column representing the assigned values.

        Notes:
        ------
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
        ] = f"binary ({polygon_value}=polygon, {buffer_value}=buffer, 1=outside)"

        return pfa

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

        layer_dict = pfa["criteria"][criteria]["components"][component]["layers"][layer]
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
            fp = Polygon(xy)
        except Exception:
            fp = LineString(xy)

        # convert zero-area or invalid polygons to boundary
        if fp.geom_type == "Polygon" and (fp.area == 0 or not fp.is_valid):
            fp = fp.boundary

        return fp

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
            pts2 = [Point(x, y) for y in ys for x in xs]
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
                    all_pts.append(Point(xs[ix], ys[iy], zs[iz]))
                    all_d.append(D[iz, iy, ix])

        gdf_out = gpd.GeoDataFrame(
            {"distance": all_d}, geometry=all_pts, crs=gdf3.crs
        )

        layer_dict["model"] = gdf_out
        layer_dict["model_data_col"] = "distance"
        layer_dict["model_units"] = "m"
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
    def calculate_intersections(gdf_lines):
        """
        Calculate intersection points between line geometries in a GeoDataFrame.

        This function takes a GeoDataFrame of lines (such as faults or other linear features)
        and calculates the intersection points between them. If two lines overlap (instead
        of intersecting at a single point), the midpoint of the overlapping segment is taken
        as the intersection point.

        Parameters:
        -----------
        gdf_lines : geopandas.GeoDataFrame
            A GeoDataFrame containing line geometries (e.g., faults) stored in the 'geometry' column.
            It is assumed that these are LineString geometries.

        Returns:
        --------
        gdf_intersections : geopandas.GeoDataFrame
            A GeoDataFrame containing the points where the input lines intersect.
            If lines overlap, the midpoint of the overlapping segment is included.
            The returned GeoDataFrame will use the same CRS (coordinate reference system)
            as the input GeoDataFrame.
        """
        intersections = []
        # Extract the geometries from the GeoDataFrame
        lines = gdf_lines["geometry"].tolist()

        # Get all unique pairs of lines to check for intersections
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1 :]:
                if line1.intersects(line2):
                    # Find the intersection point (could be a Point or a LineString if overlapping)
                    intersection = line1.intersection(line2)
                    if isinstance(intersection, shapely.geometry.Point):
                        intersections.append(intersection)
                    elif isinstance(intersection, shapely.geometry.LineString):
                        # If lines overlap, get the mid-point of the overlapping segment
                        midpoint = shapely.geometry.Point(
                            intersection.centroid
                        )
                        intersections.append(midpoint)

        # Return a GeoDataFrame of intersection points
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

        Parameters:
        ----------
        gdf_points : GeoDataFrame
            A GeoDataFrame containing point geometries for which distances will be calculated.
        tree : STRtree
            A spatial index (STRtree) containing line geometries. This allows for efficient querying of the nearest line
            for each point.
        intersection_tree : STRtree, optional
            An optional spatial index (STRtree) containing intersection geometries. If provided, the function calculates
            the nearest distance to intersections as well. If not provided, the intersection distances are set to infinity.

        Returns:
        -------
        tuple
            A tuple containing two pandas Series:
            - nearest_line_distances: The nearest distance from each point to the nearest line.
            - nearest_intersection_distances: The nearest distance from each point to the nearest intersection (or infinity
            if no intersection tree is provided).

        Notes:
        ------
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

        Parameters:
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

        Returns:
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
            raise ValueError(
                "mode must be one of ['mean','sum','max','knearest']"
            )

        # store results
        gdf_grid = gpd.GeoDataFrame(
            geometry=list(starmap(Point, grid_xyz)),
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

    def convert_3d_to_2d(pfa, criteria, component, layer):
        """
        Convert a 3D GeoDataFrame layer to a 2D representation by collapsing along the Z-dimension.

        This function processes a 3D layer stored in the `pfa` dictionary structure, filtering out empty geometries,
        sorting the data by X, Y, and Z coordinates, and aggregating data by grouping on X and Y. The resulting
        2D GeoDataFrame replaces the original 3D data in the `pfa` dictionary.

        Parameters:
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

        Returns:
        -------
        dict
            The updated `pfa` dictionary with the 3D layer converted to a 2D representation. The resulting 2D GeoDataFrame
            is stored in `pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']`.

        Raises:
        ------
        ValueError
            If the `geometry` column in the GeoDataFrame is not of type `Point` or contains invalid geometries.

        Notes:
        ------
        - Empty geometries (e.g., `POINT EMPTY`) are filtered out before processing.
        - Sorting is performed by X, Y, and Z coordinates. Empty points are handled gracefully.
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
        gdf_3d = gdf_3d[~gdf_3d.geometry.is_empty].reset_index(drop=True)

        # Sort geometries by X, Y, Z, handling empty points gracefully
        gdf_3d = gdf_3d.sort_values(
            by=["geometry"],
            key=lambda col: col.apply(
                lambda geom: (geom.x, geom.y, geom.z)
                if not geom.is_empty
                else (float("nan"), float("nan"), float("nan"))
            ),
        ).reset_index(drop=True)

        # Step 1: Aggregate by unique (X, Y, Z) points
        gdf_3d = gdf_3d.groupby(
            gdf_3d.geometry.apply(lambda geom: (geom.x, geom.y, geom.z)),
            as_index=False,
        ).agg(
            {
                "geometry": "first",  # Keep the representative geometry
                col: "sum",  # Sum the data column for unique (X, Y, Z)
            }
        )
        gdf_3d = gdf_3d.set_geometry("geometry")
        gdf_3d.set_crs(crs, inplace=True)
        # Step 2: Collapse Z dimension to (X, Y)
        gdf_2d = gdf_3d.groupby(
            gdf_3d.geometry.apply(lambda geom: (geom.x, geom.y)),
            as_index=False,
        ).agg(
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

    def extract_fault_traces_top(pfa, criteria, component, layer):
        """
        Create a 2D representation of 3D faults by extracting the trace at the top of each fault.

        Parameters:
        ----------
        gdf_3d : GeoDataFrame
            A GeoDataFrame containing 3D fault data with geometries (Polygons or MultiPolygons).
        fault_id_col : str
            The name of the column in `gdf_3d` that uniquely identifies faults.

        Returns:
        -------
        GeoDataFrame
            A GeoDataFrame containing 2D fault traces (LineStrings) for the top of each fault.

        Notes:
        ------
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

    def extract_fault_traces_bottom(
        pfa, criteria, component, layer, min_z=None, tolerance=1e6
    ):
        """
        Extracts a 2D representation of faults by taking a slice at the bottom of the model.

        Parameters:
        ----------
        gdf : GeoDataFrame
            A GeoDataFrame containing 3D fault geometries (Point geometries).
        fault_id_col : str
            The column name representing fault IDs to group points into separate faults.
        bottom_z_threshold : float
            A Z-value threshold used to select the bottom slice of the model. Points with
            Z-values within this threshold from the minimum Z will be included.

        Returns:
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
