"""
Geospatial plotting utilities for geoPFA.

This module provides both 2D and 3D plotting functions for geospatial
GeoDataFrames used throughout the geoPFA workflow.

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import contextily as ctx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import shapely
from shapely.ops import linemerge
from contextlib import suppress


class GeospatialDataPlotters:
    """
    Collection of static plotting utilities for geoPFA geospatial data.

    """

    @staticmethod
    def _coords3_from_point(pt):
        z = getattr(pt, "z", None)
        if z is None:
            c0 = pt.coords[0]
            z = c0[2] if len(c0) == 3 else 0.0  # noqa: PLR2004
        return (pt.x, pt.y, z)

    @staticmethod
    def _build_well_pts(_well):
        if _well is None:
            return None
        if hasattr(_well, "geometry"):
            if all(g.geom_type == "Point" for g in _well.geometry):
                return np.array(
                    [
                        GeospatialDataPlotters._coords3_from_point(p)
                        for p in _well.geometry
                    ],
                    dtype=float,
                )
            # fallback: lines in a GDF
            geoms = list(_well.geometry)
            merged = geoms[0]
            if len(geoms) > 1:
                with suppress(Exception):
                    merged = linemerge(geoms)
            if isinstance(
                merged, shapely.LineString | shapely.MultiLineString
            ):
                parts = (
                    merged.geoms
                    if isinstance(merged, shapely.MultiLineString)
                    else [merged]
                )
                arrs = []
                for ls in parts:
                    arr = np.asarray(ls.coords, dtype=float)
                    if arr.shape[1] == 2:  # noqa: PLR2004
                        arr = np.c_[arr, np.zeros(len(arr))]
                    arrs.append(arr)
                return np.vstack(arrs) if arrs else None
            return None
        # plain shapely lines
        if isinstance(_well, shapely.LineString | shapely.MultiLineString):
            parts = (
                _well.geoms
                if isinstance(_well, shapely.MultiLineString)
                else [_well]
            )
            arrs = []
            for ls in parts:
                arr = np.asarray(ls.coords, dtype=float)
                if arr.shape[1] == 2:  # noqa: PLR2004
                    arr = np.c_[arr, np.zeros(len(arr))]
                arrs.append(arr)
            return np.vstack(arrs) if arrs else None
        return None

    @staticmethod
    def geo_plot(  # noqa: PLR0913, PLR0917
        gdf,
        col,
        units,
        title,
        area_outline=None,
        overlay=None,
        xlabel="default",
        ylabel="default",
        cmap="jet",
        xlim=None,
        ylim=None,
        extent=None,
        basemap=False,
        markersize=15,
        figsize=(10, 10),
        vmin=None,
        vmax=None,
    ):
        """Plots data using gdf.plot(). Preserves geometry, but does not look
        smoothe.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Geodataframe containing data to plot, including a geometry column and crs.
        col : str
            Name of column containing data value to plot, if applicable.
        units : str
            Units of data to plot.
        title : str
            Title to add to plot.
        area_outline : geopandas.GeoDataFrame
            Optional, Geodataframe contatining outline of area to overlay on plot.
        overlay : geopandas.GeoDataFrame
            Optional, Geodataframe containing data locations to plot over map data.
        xlabel, ylabel : str
            Optional, label for x-axis and y-axis.
        cmap : str
            Optional, colormap to use instead of the default 'jet'.
        xlim, ylim : tuple
            Optional, limits to use for x and y axes.
        extent : list
            List of length 4 containing the extent (i.e., bounding box) to use in
            lieau of xlim and ylim, in this order: [x_min, y_min, x_max, y_max].
        basemap : bool
            Option to add a basemap, defaults to False.
        markersize : int
            Option to specify marker size to use in plot. Defaults to 15.
        figsize : tuple
            Option to specify figure size. Defaults to (10,10).
        vmin, vmax : float
            Optional minimum and maximum values to include in colorbar. If not provided,
            will use min and max value of data in the column to plot.

        """
        fig, ax = plt.subplots(figsize=figsize)
        if col is None or str(col).lower() == "none":
            gdf.plot(ax=ax)
        else:
            if vmin is None:
                norm = plt.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max())
            else:
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
            gdf.plot(
                ax=ax,
                marker="s",
                markersize=markersize,
                column=col,
                cmap=cmap,
                norm=norm,
                legend=False,
            )
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(units)
        if area_outline is not None:
            area_outline.boundary.plot(ax=ax, color="black")
        if overlay is not None:
            overlay.plot(ax=ax, color="gray", markersize=3, alpha=0.5)
        if xlabel == "default":
            xlabel = gdf.crs.axis_info[1].name if gdf.crs else "X-axis"
        if ylabel == "default":
            ylabel = gdf.crs.axis_info[0].name if gdf.crs else "Y-axis"
        if basemap:
            ctx.add_basemap(ax)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        elif extent is not None:
            plt.xlim(extent[0], extent[2])
            plt.ylim(extent[1], extent[3])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_zoom_in(  # noqa: PLR0913, PLR0917
        gdf,
        col,
        units,
        title,
        xlim,
        ylim,
        figsize,
        markersize,
        xlabel,
        ylabel,
        cmap,
    ):
        """Method to plot zoomed in version of geopfa maps, using xlim and ylim to determine the extent.
        Also adds a basemap."""
        fig, ax = plt.subplots(figsize=figsize)
        if col is None or str(col).lower() == "none":
            gdf.plot(ax=ax)
        else:
            gdf.plot(
                ax=ax,
                marker="s",
                markersize=markersize,
                column=col,
                cmap=cmap,
                legend=False,
                alpha=0.25,
            )
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max()),
            )
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(units)
        if xlabel == "default":
            xlabel = gdf.crs.axis_info[1].name if gdf.crs else "X-axis"
        if ylabel == "default":
            ylabel = gdf.crs.axis_info[0].name if gdf.crs else "Y-axis"
        # TODO: Basemap is causing problems. Fix at a later date.
        # Add the basemap
        # ctx.add_basemap(ax=ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def raster_plot(gdf, col, units, layer):
        """Plots data using pcolormesh. Creates a smoother plot, but does not
        preserve geometry in plot"""
        x = gdf.geometry.x
        y = gdf.geometry.y
        z = gdf[col]

        # grid coordinates
        xi = np.linspace(x.min(), x.max(), 500)
        yi = np.linspace(y.min(), y.max(), 500)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate
        zi = griddata((x, y), z, (xi, yi), method="linear")

        fig, ax = plt.subplots(figsize=(10, 10))
        c = ax.pcolormesh(xi, yi, zi, shading="auto", cmap="jet")
        fig.colorbar(c, ax=ax, label=units)

        plt.title(f"{layer}: heatmap")
        plt.xlabel("easting (m)")
        plt.ylabel("northing (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def geo_plot_3d(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
        gdf,
        col,
        units,
        title,
        area_outline=None,
        overlay=None,
        well_path=None,
        well_path_values=None,
        xlabel="default",
        ylabel="default",
        zlabel="Z-axis",
        cmap="jet",
        xlim=None,
        ylim=None,
        zlim=None,
        extent=None,
        markersize=15,
        figsize=(12, 10),
        vmin=None,
        vmax=None,
        filter_threshold=None,
        x_slice=None,
        y_slice=None,
        z_slice=None,
        well_units="Temperature (°C)",
        well_cmap="magma",
        show_well_colorbar=True,
        well_vmin=None,
        well_vmax=None,
        show_main_colorbar=True,
        view_nw=(20, 135),
        view_ne=(20, 45),
        view_sw=(20, -135),
        view_se=(20, -45),
    ):
        """
        Plot 3D geospatial data from a GeoDataFrame using one or more directional views.

        This function visualizes 3D point or polygon data across up to four user-defined
        view angles (NW, NE, SW, SE). Each active view is rendered as a separate subplot
        with a consistent spatial extent. A single main title is applied to the figure,
        and each subplot is labeled with its viewing direction.

        Color-mapped data (e.g., favorability) and optional well-path data can be displayed
        simultaneously, each with its own horizontal colorbar positioned below the plots.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input geospatial dataset containing 3D geometries (Point, Polygon, or MultiPolygon).
            Coordinates are expected to include Z values.

        col : str or None
            Column in `gdf` used for coloring the primary dataset. If None or "none",
            geometries are plotted in a uniform color.

        units : str
            Label for the primary dataset colorbar.

        title : str
            Main title for the entire figure (displayed once at the top).

        Overlay and Additional Geometry
        ------------------------------
        area_outline : geopandas.GeoDataFrame, optional
            Polygon geometry plotted as an outline above the data (e.g., study boundary).

        overlay : geopandas.GeoDataFrame, optional
            Additional point data plotted as a secondary scatter layer.

        well_path : geometry-like, optional
            Well trajectory input used to generate 3D well points.

        well_path_values : array-like, optional
            Values associated with well points (e.g., temperature). If provided, wells
            are colored by these values; otherwise plotted in black.

        well_units : str, default "Temperature (°C)"
            Label for the well data colorbar.

        well_cmap : str, default "magma"
            Colormap used for well data.

        well_vmin, well_vmax : float, optional
            Color scaling bounds for well data.

        show_well_colorbar : bool, default True
            Whether to display the well colorbar.

        Main Data Styling
        ----------------
        cmap : str, default "jet"
            Colormap for the primary dataset.

        vmin, vmax : float, optional
            Color scaling bounds for the primary dataset.

        markersize : float, default 15
            Marker size for scatter plots.

        show_main_colorbar : bool, default True
            Whether to display the main dataset colorbar.

        Axes, Limits, and Labels
        ------------------------
        xlabel, ylabel, zlabel : str or "default"
            Axis labels. If "default", labels are inferred from CRS when available.

        xlim, ylim, zlim : tuple, optional
            Axis limits for each dimension.

        extent : tuple, optional
            Combined spatial extent (xmin, ymin, zmin, xmax, ymax, zmax).
            Overrides individual axis limits if provided (except zlim if explicitly set).

        Slicing and Filtering
        ---------------------
        x_slice, y_slice, z_slice : float, optional
            Maximum coordinate thresholds used to filter data prior to plotting.
            Only points with coordinates <= these values are retained.

        filter_threshold : float, optional
            Minimum value of `col` required for a point to be plotted.

        Views / Camera Angles
        ---------------------
        view_nw, view_ne, view_sw, view_se : tuple(float, float) or None
            Optional camera angles for rendering directional 3D views, specified as
            (elevation, azimuth) and passed to matplotlib.axes.Axes.view_init.

            The suffixes indicate the approximate camera/viewpoint direction
            (for example, view_nw is a view from the northwest). Any view set to
            None is excluded. At least one view must be provided.

        Figure Layout
        -------------
        figsize : tuple, default (12, 10)
            Base figure size (width, height). Width scales with number of subplots.

        Notes
        -----
        - Subplots are dynamically generated based on active views (1-4).
        - All subplots share consistent axis scaling and styling.
        - Colorbars are rendered at the figure level (not per subplot) to ensure
          consistent sizing and avoid layout distortion.
        - The main colorbar is placed above the well colorbar.
        - If no valid data remains after filtering/slicing, the function exits early.
        """

        # ---------- active views ----------
        active_views = {
            "nw": view_nw,
            "ne": view_ne,
            "sw": view_sw,
            "se": view_se,
        }
        active_views = {k: v for k, v in active_views.items() if v is not None}
        if len(active_views) == 0:
            print("No active views provided.")
            return

        # ---------- helpers ----------
        def _apply_slice_pts(arr):
            if arr is None:
                return None
            mask = np.ones(len(arr), dtype=bool)
            if x_slice is not None:
                mask &= arr[:, 0] <= x_slice
            if y_slice is not None:
                mask &= arr[:, 1] <= y_slice
            if z_slice is not None:
                mask &= arr[:, 2] <= z_slice
            return arr[mask]

        # ---------- prep main dataset ----------
        gdf_copy = gdf.copy()

        # main colormap/norm
        if col is not None and str(col).lower() != "none":
            vmin_main = gdf_copy[col].min() if vmin is None else vmin
            vmax_main = gdf_copy[col].max() if vmax is None else vmax
            norm_main = plt.Normalize(vmin=vmin_main, vmax=vmax_main)
            cmap_main_obj = plt.get_cmap(cmap)
        else:
            norm_main = None
            cmap_main_obj = None

        # slicing on first coordinate (matches your original semantics)
        if x_slice is not None:
            gdf_copy = gdf_copy[
                gdf_copy.geometry.apply(
                    lambda geom: geom.coords[0][0] <= x_slice
                )
            ]
        if y_slice is not None:
            gdf_copy = gdf_copy[
                gdf_copy.geometry.apply(
                    lambda geom: geom.coords[0][1] <= y_slice
                )
            ]
        if z_slice is not None:
            gdf_copy = gdf_copy[
                gdf_copy.geometry.apply(
                    lambda geom: geom.coords[0][2] <= z_slice
                )
            ]

        # threshold filter
        if filter_threshold is not None and col != "None":
            gdf_filtered = gdf_copy[gdf_copy[col] >= filter_threshold]
        else:
            gdf_filtered = gdf_copy

        if gdf_filtered.empty and well_path is None:
            print("No data to plot after filtering and slicing.")
            return

        # color array for points (only created if col provided)
        if (
            col is not None and str(col).lower() != "none"
        ) and not gdf_filtered.empty:
            filtered_colors = cmap_main_obj(norm_main(gdf_filtered[col]))
        else:
            filtered_colors = "blue"

        # well data
        well_pts = _apply_slice_pts(
            GeospatialDataPlotters._build_well_pts(well_path)
        )
        well_vals = (
            None if well_path_values is None else np.asarray(well_path_values)
        )

        # check usable values
        has_well_values = (
            well_pts is not None
            and len(well_pts) > 0
            and well_vals is not None
            and np.isfinite(well_vals).any()
        )

        # ---------- figure ----------
        n = len(active_views)
        fig_w, fig_h = figsize
        _, axes = plt.subplots(
            1,
            n,
            figsize=(fig_w * n / 2, fig_h),
            subplot_kw={"projection": "3d"},
        )

        if n == 1:
            axes = [axes]

        main_mappable = None
        well_mappable = None

        # ---------- shared per-panel plotting ----------
        def _plot_on(  # noqa: PLR0912, PLR0914, PLR0915
            ax,
        ):
            sc = None
            # main geometries
            if not gdf_filtered.empty:
                gtype0 = gdf_filtered.geometry.iloc[0].geom_type
                if gtype0 == "Point":
                    xs, ys, zs = zip(
                        *[geom.coords[0] for geom in gdf_filtered.geometry]
                    )
                    if col is not None and str(col).lower() != "none":
                        sc = ax.scatter(
                            xs,
                            ys,
                            zs,
                            s=markersize,
                            c=gdf_filtered[col],
                            cmap=cmap_main_obj,
                            norm=norm_main,
                        )
                    else:
                        sc = ax.scatter(xs, ys, zs, s=markersize, color="blue")
                elif gtype0 in {"Polygon", "MultiPolygon"}:
                    for geom in gdf_filtered.geometry:
                        if geom.geom_type == "Polygon":
                            rings = [geom.exterior, *list(geom.interiors)]
                        elif geom.geom_type == "MultiPolygon":
                            rings = [
                                ring
                                for polygon in geom.geoms
                                for ring in [
                                    polygon.exterior,
                                    *list(polygon.interiors),
                                ]
                            ]
                        else:
                            rings = []
                        for ring in rings:
                            verts = [
                                (c[0], c[1], c[2] if len(c) == 3 else 0)  # noqa: PLR2004
                                for c in ring.coords
                            ]
                            ax.add_collection3d(
                                Poly3DCollection(
                                    [verts],
                                    alpha=0.5,
                                    edgecolor="grey",
                                    facecolor="lightblue",
                                )
                            )
            nonlocal main_mappable
            if main_mappable is None and sc is not None:
                main_mappable = sc

            # overlay
            if (
                overlay is not None
                and hasattr(overlay, "empty")
                and not overlay.empty
            ):
                ox, oy, oz = zip(
                    *[geom.coords[0] for geom in overlay.geometry]
                )
                ax.scatter(ox, oy, oz, color="gray", s=5, alpha=0.5)

            # well path scatter
            sc_well = None
            if well_pts is not None and len(well_pts):
                if not has_well_values:
                    # just draw the well in black if no values
                    sc_well = ax.scatter(
                        well_pts[:, 0],
                        well_pts[:, 1],
                        well_pts[:, 2],
                        s=markersize * 1.6,
                        color="k",
                        alpha=0.9,
                        zorder=5,
                    )
                else:
                    vals = well_vals
                    if len(vals) > len(well_pts):
                        vals = vals[: len(well_pts)]
                    elif len(vals) < len(well_pts):
                        vals = np.concatenate(
                            [vals, np.full(len(well_pts) - len(vals), np.nan)]
                        )

                    w_cmap = plt.get_cmap(well_cmap)
                    vmin_w = (
                        np.nanmin(vals) if well_vmin is None else well_vmin
                    )
                    vmax_w = (
                        np.nanmax(vals) if well_vmax is None else well_vmax
                    )
                    norm_w = plt.Normalize(vmin=vmin_w, vmax=vmax_w)
                    sc_well = ax.scatter(
                        well_pts[:, 0],
                        well_pts[:, 1],
                        well_pts[:, 2],
                        s=markersize * 1.6,
                        c=vals,
                        cmap=w_cmap,
                        norm=norm_w,
                        alpha=0.9,
                        zorder=5,
                    )

            nonlocal well_mappable
            if well_mappable is None and sc_well is not None:
                well_mappable = sc_well

            # area outline
            if (
                area_outline is not None
                and hasattr(area_outline, "empty")
                and not area_outline.empty
            ):
                if (
                    not gdf_copy.empty
                    and gdf_copy.geometry.iloc[0].geom_type == "Point"
                ):
                    zmax = max(geom.z for geom in gdf_copy.geometry)
                elif not gdf_copy.empty and gdf_copy.geometry.iloc[
                    0
                ].geom_type in {"Polygon", "MultiPolygon"}:
                    zmax = max(
                        max(
                            coord[2]
                            for coord in ring.coords
                            if len(coord) == 3  # noqa: PLR2004
                        )
                        for geom in gdf_copy.geometry
                        for ring in ([geom.exterior, *list(geom.interiors)])
                    )
                else:
                    zmax = 0
                for poly in area_outline.geometry:
                    xs, ys = zip(*[(c[0], c[1]) for c in poly.exterior.coords])
                    zs = [zmax + 1] * len(xs)
                    ax.plot(xs, ys, zs, color="black")

            # labels & limits
            xlabel_final = (
                xlabel
                if xlabel != "default"
                else (
                    gdf_copy.crs.axis_info[1].name
                    if gdf_copy.crs
                    else "X-axis"
                )
            )
            ylabel_final = (
                ylabel
                if ylabel != "default"
                else (
                    gdf_copy.crs.axis_info[0].name
                    if gdf_copy.crs
                    else "Y-axis"
                )
            )
            zlabel_final = zlabel or "Z-axis"
            ax.set_xlabel(xlabel_final)
            ax.set_ylabel(ylabel_final)
            ax.set_zlabel(zlabel_final)

            if extent is not None and zlim is None:
                ax.set_xlim(extent[0], extent[3])
                ax.set_ylim(extent[1], extent[4])
                ax.set_zlim(extent[2], extent[5])
            else:
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if zlim is not None:
                    ax.set_zlim(zlim)

            ax.grid(True)

        # apply views
        for ax, (name, view) in zip(axes, active_views.items()):
            ax.view_init(*view)
            _plot_on(ax)
            ax.set_title(f"View from {name.upper()}")

        fig = plt.gcf()

        # main title
        fig.suptitle(title, fontsize=14, y=0.95)

        # leave room at top and bottom
        plt.subplots_adjust(bottom=0.2, top=0.92)

        # main colorbar
        if main_mappable is not None and show_main_colorbar:
            cbar = fig.colorbar(
                main_mappable,
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.08,
            )
            cbar.ax.set_title(units, fontsize=10, pad=6)

        # well colorbar (below main)
        if well_mappable is not None and show_well_colorbar:
            cbar_w = fig.colorbar(
                well_mappable,
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.28,
            )
            cbar_w.ax.set_title(f"{well_units} (well)", fontsize=10, pad=6)

        plt.show()
