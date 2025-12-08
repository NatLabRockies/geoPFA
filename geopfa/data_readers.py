"""
Set of methods to read in data in various formats.
"""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
import shapely
import pyproj
import rasterio
import re
from itertools import starmap
from geopfa.processing import Processing

from geopfa.processing.Processing import convert_z_measurements


class GeospatialDataReaders:
    """Read geospatial data in various formats"""

    @staticmethod
    def read_shapefile(path):
        """Reads in a shapefile and returns a geopandas dataframe.

        Parameters
        ----------
        path : 'str'
            Path to shapefile

        Returns
        -------
        data : Geopandas DataFrame
            Geopandas DataFrame containing contents of shapefile
        """
        data = gpd.read_file(path)
        return data

    @staticmethod
    def read_csv(
        path,
        crs,
        x_col=None,
        y_col=None,
        z_col=None,
        geometry_column_name="geometry",
    ):
        """Reads in a CSV file and returns a geopandas dataframe.

        Parameters
        ----------
        path : 'str'
            Path to csv file
        crs : 'str' or 'int'
            String or integer version of coordinate reference system
            associated with the CSV file.
        x_col : 'str'
            Name of x geometry column if no combined geometry column
            is provided.
        y_col : 'str'
            Name of y geometry column if no combined geometry column
            is provided.
        z_col : 'str'
            Name of z geometry column if 3D, and if no combined geometry
            column is provided.
        geometry_column_name : str
            Name of column containing the geometry information. Defaults
            to 'geometry.'

        Returns
        -------
        gdf : Geopandas DataFrame
            Geopandas DataFrame containing contents of CSV file
        """
        # Read the CSV file
        df = pd.read_csv(path)

        # Validate input geometry columns
        if sum([(x_col is None), (y_col is None)]) == 1:
            raise ValueError(
                "Must specify both x_col and y_col, or a combined geometry column."
            )
        if (z_col is not None) and (x_col is None or y_col is None):
            raise ValueError(
                "Cannot specify z_col without also specifying x_col and y_col."
            )

        # Validate CRS
        try:
            crs = pyproj.CRS.from_user_input(crs) if crs else None
        except pyproj.exceptions.CRSError as e:
            raise ValueError(f"Invalid CRS provided: {crs}. Error: {e}")

        # Create geometry from a combined geometry column
        if x_col is None and y_col is None and z_col is None:
            df = df.rename(columns={geometry_column_name: "geometry"})
            df["geometry"] = df["geometry"].apply(wkt.loads)
            gdf = gpd.GeoDataFrame(df, crs=crs)

        # Create 2D geometry from x and y columns
        elif z_col is None:
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
            )
            gdf.set_crs(crs, inplace=True)

        # Create 3D geometry from x, y, and z columns
        else:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=list(
                    map(
                        shapely.geometry.Point, df[x_col], df[y_col], df[z_col]
                    )
                ),
            )
            gdf.set_crs(crs, inplace=True)

        return gdf

    @staticmethod
    def read_raster(path):
        """Reads in a raster file and returns a rasterio dataset object.

        Parameters
        ----------
        path : 'str'
            Path to raster file

        Returns
        -------
        data : Geopandas DataFrame
            Geopandas DataFrame containing contents of raster file
        """
        # TODO: Return geopandas dataframe instead of rasterio dataset object
        data = rasterio.open(path)
        return data

    @staticmethod
    def read_tif(tif_path):
        """Read a TIFF file and convert it to a GeoDataFrame with points.

        Parameters
        ----------
        tif_path : str
            Path to the TIFF file.

        Returns
        -------
        GeoDataFrame
            GeoDataFrame with points representing the raster data.
        """
        # Open the TIFF file
        with rasterio.open(tif_path) as src:
            # Read the raster data
            data = src.read(1)  # Read the first band
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        # Get all indices where data is not equal to nodata
        rows, cols = np.where(data != nodata)

        # Convert raster indices to spatial coordinates
        x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)

        # Create points and values arrays
        points = list(map(shapely.geometry.Point, x_coords, y_coords))
        values = data[rows, cols]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({"geometry": points, "value": values})
        gdf.set_crs(crs, inplace=True)

        return gdf

    @staticmethod
    def read_tec(
        path,
        crs,
        x_col=None,
        y_col=None,
        z_col=None,
        geometry_column_name="geometry",
    ):
        """Reads in a double-space-delimited TEC file and returns a
        geopandas dataframe.

        Assumes:
            - First row: title (ignored)
            - Second row: Tecplot-style VARIABLES declaration, e.g.
            VARIABLES = "X" "Y" "Z" "P(Pa)" ...
            - Third row: assumptions / zone line (ignored)
            - Data begins on row 4

        Parameters
        ----------
        path : 'str'
            Path to TEC file.
        crs : 'str' or 'int'
            String or integer version of coordinate reference system
            associated with the TEC file.
        x_col : 'str'
            Name of x geometry column if no combined geometry column
            is provided.
        y_col : 'str'
            Name of y geometry column if no combined geometry column
            is provided.
        z_col : 'str'
            Name of z geometry column if 3D, and if no combined geometry
            column is provided.
        geometry_column_name : str
            Name of column containing the geometry information. Defaults
            to 'geometry.'

        Returns
        -------
        gdf : Geopandas DataFrame
            Geopandas DataFrame containing contents of TEC file
        """

        # Read all lines
        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) < 4:
            raise ValueError(
                "File must contain at least title, header, assumptions, and data rows."
            )

        # ------------------------------------------------------------------
        # Parse column names from Tecplot-style header:
        #   VARIABLES = "X" "Y" "Z" "P(Pa)" ...
        # ------------------------------------------------------------------
        header_line = lines[1].strip()

        # Get everything inside quotes: ["X", "Y", "Z", "P(Pa)", ...]
        columns = re.findall(r'"([^"]+)"', header_line)

        # Fallback if we didn't find quoted names (more generic TEC)
        if not columns:
            tokens = header_line.split()
            # Drop leading VARIABLES token if present
            if tokens and tokens[0].lower().startswith("variables"):
                tokens = tokens[1:]
            columns = [t.strip('"') for t in tokens]

        # ------------------------------------------------------------------
        # Parse data from line 4 onward.
        # Use .split() to handle arbitrary whitespace (double spaces, tabs, etc.)
        # ------------------------------------------------------------------
        data = []
        for line in lines[3:]:
            if not line.strip():
                continue
            values = line.split()

            # If line is short/long, pad or truncate to match number of columns
            if len(values) < len(columns):
                values.extend([None] * (len(columns) - len(values)))
            elif len(values) > len(columns):
                values = values[: len(columns)]

            data.append(values)

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Clean column names: strip surrounding whitespace and quotes
        df.rename(columns=lambda c: c.strip().strip('"'), inplace=True)

        # Convert numeric columns where possible (avoid FutureWarning)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Leave non-numeric columns as-is
                pass

        # ------------------------------------------------------------------
        # Geometry handling (same pattern as read_csv)
        # ------------------------------------------------------------------

        # Validate geometry input
        if sum([(x_col is None), (y_col is None)]) == 1:
            raise ValueError(
                "Must specify both x_col and y_col, or a combined geometry column."
            )
        if (z_col is not None) and (x_col is None or y_col is None):
            raise ValueError(
                "Cannot specify z_col without also specifying x_col and y_col."
            )

        # Validate CRS
        try:
            crs = pyproj.CRS.from_user_input(crs) if crs else None
        except pyproj.exceptions.CRSError as e:
            raise ValueError(f"Invalid CRS provided: {crs}. Error: {e}")

        # Create geometry from a combined geometry column (rare for TEC, but supported)
        if x_col is None and y_col is None and z_col is None:
            df = df.rename(columns={geometry_column_name: "geometry"})
            df["geometry"] = df["geometry"].apply(wkt.loads)
            gdf = gpd.GeoDataFrame(df, crs=crs)

        # Create 2D geometry from x and y columns
        elif z_col is None:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[x_col], df[y_col]),
            )
            gdf.set_crs(crs, inplace=True)

        # Create 3D geometry from x, y, and z columns
        else:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=list(
                    map(
                        shapely.geometry.Point,
                        df[x_col],
                        df[y_col],
                        df[z_col],
                    )
                ),
            )
            gdf.set_crs(crs, inplace=True)

        return gdf

    @staticmethod
    def read_well_path_csv(
        csv_path,
        x_col,
        y_col,
        z_col=None,
        *,
        well_name_col=None,  # optional
        value_col=None,  # optional per-vertex values (temp/GR/etc.)
        source_crs=None,
        to_crs=None,
        sort_by=None,
        md_col=None,
        dropna_any=True,
        deduplicate_consecutive=True,
        z_meas=None,
        target_z_meas=None,
        convert_z_after=True,
    ):
        """Reads a well-path CSV and returns a point GeoDataFrame (ordered vertices) and optional values.

        Parameters
        ----------
        csv_path : 'str'
            Path to the CSV file containing well path vertices.
        x_col : 'str'
            Column name for X (or longitude).
        y_col : 'str'
            Column name for Y (or latitude).
        z_col : 'str', optional
            Column name for Z (depth or elevation). If omitted, Z=0.
        well_name_col : 'str', optional
            Column name for well name; stored in output.
        value_col : 'str', optional
            Column name for per-vertex values (e.g., Temperature, GR).
        source_crs : 'int or str', optional
            Input CRS for X/Y. Example: 32613 or 4326.
        to_crs : 'int or str', optional
            Output CRS for reprojection.
        sort_by : 'str', optional
            Column to sort by before building points (e.g., 'MD_m').
        md_col : 'str', optional
            If provided and sort_by is None, rows are sorted by this column.
        dropna_any : 'bool'
            If True, drop rows with any NaN in X/Y/Z; else only drop rows with all NaNs.
        deduplicate_consecutive : 'bool'
            If True, drop consecutive duplicate XY Z vertices.
        z_meas : 'str', optional
            Vertical reference of Z in the CSV (e.g., 'depth', 'm-msl', 'epsg:5703').
        target_z_meas : 'str', optional
            Desired vertical reference (e.g., 'm-msl', 'ft-msl', 'epsg:6360').
        convert_z_after : 'bool'
            If True and z_meas/target_z_meas provided, converts Z via convert_z_measurements().

        Returns
        -------
        well_gdf : GeoDataFrame
            Ordered vertices as 3D Point geometries (one row per vertex).
        values : np.ndarray or None
            Per-vertex values aligned with rows (or None if value_col not provided).
        """
        df = pd.read_csv(csv_path)

        # Pull coordinates
        x = df[x_col].astype(float).to_numpy()
        y = df[y_col].astype(float).to_numpy()
        if z_col is not None:
            z = df[z_col].astype(float).to_numpy()
        else:
            z = np.zeros_like(x, dtype=float)

        coords = np.column_stack([x, y, z])

        # Drop NaNs
        if dropna_any:
            mask = ~np.isnan(coords).any(axis=1)
        else:
            mask = ~(np.isnan(coords).all(axis=1))
        df = df.loc[mask].reset_index(drop=True)
        coords = coords[mask]

        if coords.shape[0] < 2:
            raise ValueError(
                "Not enough valid vertices to define a well path."
            )

        # Sorting
        if sort_by is None and md_col is not None and md_col in df.columns:
            sort_by = md_col
        if sort_by is not None and sort_by in df.columns:
            order = np.argsort(df[sort_by].to_numpy())
            df = df.iloc[order].reset_index(drop=True)
            coords = coords[order]

        # De-duplicate consecutive vertices
        if deduplicate_consecutive and len(coords) > 1:
            keep = np.ones(len(coords), dtype=bool)
            keep[1:] = (np.abs(np.diff(coords, axis=0)) > 0).any(axis=1)
            df = df.loc[keep].reset_index(drop=True)
            coords = coords[keep]
            if len(coords) < 2:
                raise ValueError(
                    "Well path collapsed to one vertex after de-duplication."
                )

        # Build point GeoDataFrame
        geometries = [
            shapely.geometry.Point(float(X), float(Y), float(Z))
            for X, Y, Z in coords
        ]
        well_name = (
            df[well_name_col].iloc[0]
            if (well_name_col and well_name_col in df.columns)
            else None
        )
        well_gdf = gpd.GeoDataFrame(
            {"name": [well_name] * len(geometries)},
            geometry=geometries,
            crs=source_crs,
        )

        # Reproject horizontal CRS if requested
        if to_crs is not None:
            well_gdf = well_gdf.to_crs(to_crs)

        # Optional per-vertex values
        values = None
        if value_col is not None:
            if value_col not in df.columns:
                raise ValueError(f"Column '{value_col}' not found for values.")
            values = df[value_col].to_numpy()
            # Align to vertex count
            if len(values) > len(well_gdf):
                values = values[: len(well_gdf)]
            elif len(values) < len(well_gdf):
                pad = np.full(len(well_gdf) - len(values), np.nan)
                values = np.concatenate([values, pad])

        if convert_z_after and z_meas is not None and target_z_meas is not None:
            well_gdf = Processing.convert_z_measurements(well_gdf, z_meas, target_z_meas)

        return well_gdf, values

    @classmethod
    def gather_data(cls, data_dir, pfa, file_types): 
        """Function to read in data layers associated with each component of each criteria.
        Note that data must be stored in a directory with the following structure which matches
        the config: criteria/component/layers. Criteria directory, component directory, and
        data file names must match the critera, components, and layers specified in the pfa,
        and file extensions must match those specified in file_types.

        Parameters
        ----------
        data_dir : str
            Path to directory where data is stored
        pfa : dictionary
            Config specifying criteria, components, and data layers'
            relationship to one another. Read in from json file.
        file_types : list
            List of file types to look for when gathering data. File
            types excluded from list will be ignored.

        Returns
        -------
        pfa : dictionary
            Updated pfa config which includes data
        """
        data_dir = Path(data_dir)

        for criteria in pfa["criteria"]:
            print("criteria: " + criteria)
            for component in pfa["criteria"][criteria]["components"]:
                print("\t component: " + component)
                COMPONENT_DIR = data_dir / criteria / component
                file_names = sorted(COMPONENT_DIR.iterdir())

                # --- Shapefiles -------------------------------------------------
                if ".shp" in file_types:
                    shapefile_names = [
                        x.name for x in file_names if (x.suffix == ".shp")
                    ]
                    for layer in pfa["criteria"][criteria]["components"][component][
                        "layers"
                    ]:
                        if f"{layer}.shp" in shapefile_names:
                            print("\t\t reading layer: " + layer)
                            pfa["criteria"][criteria]["components"][component][
                                "layers"
                            ][layer]["data"] = GeospatialDataReaders.read_shapefile(
                                COMPONENT_DIR / f"{layer}.shp"
                            )

                # --- CSV files ---------------------------------------------------
                if ".csv" in file_types:
                    csv_file_names = [
                        x.name for x in file_names if (x.suffix == ".csv")
                    ]
                    for layer in pfa["criteria"][criteria]["components"][component][
                        "layers"
                    ]:
                        if f"{layer}.csv" in csv_file_names:
                            print("\t\t reading layer: " + layer)
                            layer_config = pfa["criteria"][criteria]["components"][
                                component
                            ]["layers"][layer]
                            csv_crs = layer_config["crs"]

                            if (
                                "x_col" in layer_config
                                and "y_col" in layer_config
                            ):
                                x_col = layer_config["x_col"]
                                y_col = layer_config["y_col"]

                                if "z_col" in layer_config:
                                    z_col = layer_config["z_col"]
                                    data = GeospatialDataReaders.read_csv(
                                        COMPONENT_DIR / f"{layer}.csv",
                                        csv_crs,
                                        x_col,
                                        y_col,
                                        z_col,
                                    )
                                else:
                                    data = GeospatialDataReaders.read_csv(
                                        COMPONENT_DIR / f"{layer}.csv",
                                        csv_crs,
                                        x_col,
                                        y_col,
                                    )
                            else:
                                data = GeospatialDataReaders.read_csv(
                                    COMPONENT_DIR / f"{layer}.csv",
                                    csv_crs,
                                )

                            pfa["criteria"][criteria]["components"][component][
                                "layers"
                            ][layer]["data"] = data

                # --- TEC files ---------------------------------------------------
                if ".tec" in file_types:
                    tec_file_names = [
                        x.name for x in file_names if (x.suffix == ".tec")
                    ]
                    for layer in pfa["criteria"][criteria]["components"][component][
                        "layers"
                    ]:
                        if f"{layer}.tec" in tec_file_names:
                            print("\t\t reading layer: " + layer)
                            layer_config = pfa["criteria"][criteria]["components"][
                                component
                            ]["layers"][layer]
                            tec_crs = layer_config["crs"]

                            if (
                                "x_col" in layer_config
                                and "y_col" in layer_config
                            ):
                                x_col = layer_config["x_col"]
                                y_col = layer_config["y_col"]

                                if "z_col" in layer_config:
                                    z_col = layer_config["z_col"]
                                    data = GeospatialDataReaders.read_tec(
                                        COMPONENT_DIR / f"{layer}.tec",
                                        tec_crs,
                                        x_col,
                                        y_col,
                                        z_col,
                                    )
                                else:
                                    data = GeospatialDataReaders.read_tec(
                                        COMPONENT_DIR / f"{layer}.tec",
                                        tec_crs,
                                        x_col,
                                        y_col,
                                    )
                            else:
                                # Fall back to read_tec with only path + CRS
                                data = GeospatialDataReaders.read_tec(
                                    COMPONENT_DIR / f"{layer}.tec",
                                    tec_crs,
                                )
                            pfa["criteria"][criteria]["components"][component][
                                "layers"
                            ][layer]["data"] = data

                # --- Unknown file types -----------------------------------------
                for file_type in file_types:
                    if file_type not in {".shp", ".csv", ".tec"}:
                        print(
                            f"Warning: file type: {file_type} "
                            " not currently compatible with geoPFA."
                        )

        return pfa

    @classmethod
    def gather_processed_data(cls, data_dir, pfa, crs):
        """Function to read in processed data layers associated with each component of each criteria.
        Note that data must be stored in a directory with the following structure which matches
        the config: criteria/component/layers. Criteria directory, component directory, and
        data file names must match the critera, components, and layers specified in the pfa,
        and file extensions must match those specified in file_types.

        Parameters
        ----------
        data_dir : str
            Path to directory where data is stored
        pfa : dictionary
            Config specifying criteria, components, and data layers'
            relationship to one another. Read in from json file.

        Returns
        -------
        pfa : dictionary
            Updated pfa config which includes data
        """
        data_dir = Path(data_dir)

        for criteria in pfa["criteria"]:
            print("criteria: " + criteria)
            for component in pfa["criteria"][criteria]["components"]:
                print("\t component: " + component)
                COMPONENT_DIR = data_dir / criteria / component
                file_names = sorted(COMPONENT_DIR.iterdir())
                csv_file_names = [
                    x.name
                    for x in file_names
                    if x.name.endswith("_processed.csv")
                ]
                for layer in pfa["criteria"][criteria]["components"][
                    component
                ]["layers"]:
                    if layer + "_processed.csv" in csv_file_names:
                        print("\t\t reading layer: " + layer)
                        pfa["criteria"][criteria]["components"][component][
                            "layers"
                        ][layer]["model"] = GeospatialDataReaders.read_csv(
                            COMPONENT_DIR / f"{layer}_processed.csv",
                            crs,
                        )
        return pfa

    @classmethod
    def gather_exclusion_areas(cls, data_dir, pfa, target_crs):
        """Gathers/reads in exclusion area shapefiles for a given set of exclusion components and layers,
        transforming them into the target coordinate reference system (CRS) and storing them in the `pfa` dictionary.

        This function iterates over the exclusion components and their associated layers in the `pfa` dictionary,
        reads the corresponding shapefiles from the specified directory, filters the shapefile data based on the
        `DN` field (keeping only entries where `DN > 0`), and reprojects the geometries to the target CRS.
        The processed shapefiles are stored back into the `pfa` dictionary under the relevant component and layer.

        Parameters:
        ----------
        cls : class
            The class that the method belongs to. This is typically
            passed automatically in class methods.
        data_dir : str
            The directory path where the exclusion shapefiles are
            stored. The shapefiles are expected to be located under a
            subdirectory named 'exclusion' within this directory.
        pfa : dict
            A dictionary containing spatial data and exclusion
            components. The function reads exclusion areas for each
            component and layer and updates the dictionary with the
            processed shapefiles
        target_crs : str or dict
            The target Coordinate Reference System (CRS) to which the
            exclusion shapefiles will be reprojected. This can be a CRS
            string (e.g., 'EPSG:4326') or a CRS dictionary format.

        Returns:
        -------
        dict
            The updated `pfa` dictionary, where the processed exclusion
            areas are stored under
            `pfa['exclusions']['components'][exclusion_component]['layers'][layer]['model']`.

        Notes:
        ------
        - The function assumes that the exclusion shapefiles are stored
        in the `data_dir` under a subdirectory named 'exclusion' and
        that the filenames match the layer names.
        - Only shapefile records where the `DN` field has a value
        greater than 0 are retained for further processing.
        - The shapefile geometries are reprojected to the specified
        `target_crs` to ensure consistent spatial reference.
        - The processed shapefiles are stored in the `pfa` dictionary
        under their respective exclusion components and layers.
        """
        data_dir = Path(data_dir)

        for exclusion_component in pfa["exclusions"]["components"]:
            for layer in pfa["exclusions"]["components"][exclusion_component][
                "layers"
            ]:
                print("reading " + layer)
                path = data_dir / "exclusion" / f"{layer}.shp"
                shp = GeospatialDataReaders.read_shapefile(path)
                shp = shp[shp.DN > 0]
                shp = shp.to_crs(target_crs)
                pfa["exclusions"]["components"][exclusion_component]["layers"][
                    layer
                ]["model"] = shp
        return pfa
