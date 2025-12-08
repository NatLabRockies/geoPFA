# -*- coding: utf-8 -*-
"""
Set of methods to read in data in various formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
import contextily as ctx
import shapely

class GeospatialDataPlotters:
    """Class of functions to plot geospatial data in various formats"""
    @staticmethod
    def geo_plot(gdf,col,units,title,area_outline=None,overlay=None,xlabel='default',ylabel='default',\
        cmap='jet',xlim=None,ylim=None,extent=None,basemap=False,markersize=15,figsize=(10, 10),vmin=None,vmax=None):
        """Plots data using gdf.plot(). Preserves geometry, but does not look 
        smoothe.
        
        Parameters
        ----------
        gdf : pandas geodataframe
            Geodataframe containing data to plot, including a geometry column and crs.
        col : str
            Name of column containing data value to plot, if applicable.
        units : str
            Units of data to plot.
        title : str
            Title to add to plot.
        area_outline : geodataframe
            Optional, Geodataframe contatining outline of area to overlay on plot.
        overlay : geodataframe
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
                norm=plt.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max())
            else:
                norm=plt.Normalize(vmin=vmin, vmax=vmax)
            gdf.plot(ax=ax, marker='s', markersize=markersize,
                    column=col,cmap=cmap,norm=norm,legend=False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(units)
        if area_outline is not None:
            area_outline.boundary.plot(ax=ax,color='black')
        if overlay is not None:
            overlay.plot(ax=ax,color='gray',markersize=3,alpha=0.5)
        if xlabel == 'default':
            xlabel = gdf.crs.axis_info[1].name
        if ylabel == 'default':
            ylabel = gdf.crs.axis_info[0].name
        if basemap is True:
            ctx.add_basemap(ax)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        elif extent is not None:
            plt.xlim(extent[0],extent[2])
            plt.ylim(extent[1],extent[3])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def geo_plot_3d(
        gdf, col, units, title,
        area_outline=None, overlay=None, well_path=None, well_path_values=None,
        xlabel='default', ylabel='default', zlabel='Z-axis',
        cmap='jet', xlim=None, ylim=None, zlim=None, extent=None, markersize=15, figsize=(12, 10),
        vmin=None, vmax=None, filter_threshold=None, x_slice=None, y_slice=None, z_slice=None,
        # Well-path colorbar settings
        well_units='Temperature (°C)',
        well_cmap='magma',
        show_well_colorbar=True,
        well_vmin=None,          # independent well-path vmin
        well_vmax=None,          # independent well-path vmax
        # Main (favorability) colorbar settings
        show_main_colorbar=True,
        # View controls: azimuths approximate "looking toward" each compass direction
        view_nw=(20, 135),       # from SE looking toward NW
        view_ne=(20, 45),        # from SW looking toward NE
        view_sw=(20, -135),      # from NE looking toward SW
        view_se=(20, -45),       # from NW looking toward SE
        # Layout control: relative width of colorbar column
        cbar_width=0.08
    ):
        """
        Plots 3D geospatial data with four directional views (NW, NE, SW, SE) in a 2×2 grid.

        The two colorbars live in a separate right-hand column:
            - Top:  main dataset colorbar (col / units)
            - Bottom: well-path colorbar (well_units), if well values exist

        Parameters are otherwise identical to your previous version.
        """

        # ---------- helpers ----------
        def _coords3_from_point(pt):
            z = getattr(pt, "z", None)
            if z is None:
                c0 = pt.coords[0]
                z = c0[2] if len(c0) == 3 else 0.0
            return (pt.x, pt.y, z)

        def _build_well_pts(_well):
            if _well is None:
                return None
            if hasattr(_well, "geometry"):
                if all(g.geom_type == "Point" for g in _well.geometry):
                    return np.array([_coords3_from_point(p) for p in _well.geometry], dtype=float)
                # fallback: lines in a GDF
                geoms = list(_well.geometry)
                merged = geoms[0]
                if len(geoms) > 1:
                    try:
                        from shapely.ops import linemerge
                        merged = linemerge(geoms)
                    except Exception:
                        pass
                if isinstance(merged, (LineString, MultiLineString)):
                    parts = merged.geoms if isinstance(merged, MultiLineString) else [merged]
                    arrs = []
                    for ls in parts:
                        arr = np.asarray(ls.coords, dtype=float)
                        if arr.shape[1] == 2:
                            arr = np.c_[arr, np.zeros(len(arr))]
                        arrs.append(arr)
                    return np.vstack(arrs) if arrs else None
                return None
            # plain shapely lines
            if isinstance(_well, (LineString, MultiLineString)):
                parts = _well.geoms if isinstance(_well, MultiLineString) else [_well]
                arrs = []
                for ls in parts:
                    arr = np.asarray(ls.coords, dtype=float)
                    if arr.shape[1] == 2:
                        arr = np.c_[arr, np.zeros(len(arr))]
                    arrs.append(arr)
                return np.vstack(arrs) if arrs else None
            return None

        def _apply_slice_pts(arr):
            if arr is None:
                return None
            mask = np.ones(len(arr), dtype=bool)
            if x_slice is not None: mask &= arr[:, 0] <= x_slice
            if y_slice is not None: mask &= arr[:, 1] <= y_slice
            if z_slice is not None: mask &= arr[:, 2] <= z_slice
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
            gdf_copy = gdf_copy[gdf_copy.geometry.apply(lambda geom: geom.coords[0][0] <= x_slice)]
        if y_slice is not None:
            gdf_copy = gdf_copy[gdf_copy.geometry.apply(lambda geom: geom.coords[0][1] <= y_slice)]
        if z_slice is not None:
            gdf_copy = gdf_copy[gdf_copy.geometry.apply(lambda geom: geom.coords[0][2] <= z_slice)]

        # threshold filter
        if filter_threshold is not None and col != "None":
            gdf_filtered = gdf_copy[gdf_copy[col] >= filter_threshold]
        else:
            gdf_filtered = gdf_copy

        if gdf_filtered.empty and well_path is None:
            print("No data to plot after filtering and slicing.")
            return

        # color array for points (only created if col provided)
        if (col is not None and str(col).lower() != "none") and not gdf_filtered.empty:
            filtered_colors = cmap_main_obj(norm_main(gdf_filtered[col]))
        else:
            filtered_colors = 'blue'

        # well points + values
        well_pts = _apply_slice_pts(_build_well_pts(well_path))
        well_vals = None if well_path_values is None else np.asarray(well_path_values)

        # do we actually have usable well values?
        has_well_values = (
            well_pts is not None
            and len(well_pts) > 0
            and well_vals is not None
            and np.isfinite(well_vals).any()
        )

        # ---------- figure layout: 2×2 views + right colorbar column ----------
        # Grid: 2 rows × 3 columns
        #   [ NW | NE | main_cbar ]
        #   [ SW | SE | well_cbar ]
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import MaxNLocator

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, cbar_width])

        ax_nw = fig.add_subplot(gs[0, 0], projection='3d')
        ax_ne = fig.add_subplot(gs[0, 1], projection='3d')
        ax_sw = fig.add_subplot(gs[1, 0], projection='3d')
        ax_se_ax = fig.add_subplot(gs[1, 1], projection='3d')

        cax_main = fig.add_subplot(gs[0, 2])  # main colorbar
        cax_well = fig.add_subplot(gs[1, 2])  # well colorbar

        # make cbar axes frameless but keep ticks/labels visible
        for cax in (cax_main, cax_well):
            for spine in cax.spines.values():
                spine.set_visible(False)

        # hide well cbar axis entirely if we know we won't use it
        if not (show_well_colorbar and has_well_values):
            cax_well.set_axis_off()

        # set view angles
        ax_nw.view_init(*view_nw)
        ax_ne.view_init(*view_ne)
        ax_sw.view_init(*view_sw)
        ax_se_ax.view_init(*view_se)

        # ---------- shared per-panel plotting ----------
        def _plot_on(ax, add_main_cbar=False, add_well_cbar=False):
            # main geometries
            if not gdf_filtered.empty:
                gtype0 = gdf_filtered.geometry.iloc[0].geom_type
                if gtype0 == 'Point':
                    xs, ys, zs = zip(*[geom.coords[0] for geom in gdf_filtered.geometry])
                    if isinstance(filtered_colors, str):
                        ax.scatter(xs, ys, zs, s=markersize, color=filtered_colors)
                    else:
                        ax.scatter(xs, ys, zs, s=markersize, c=filtered_colors)
                elif gtype0 in ['Polygon', 'MultiPolygon']:
                    for geom in gdf_filtered.geometry:
                        if geom.geom_type == 'Polygon':
                            rings = [geom.exterior] + list(geom.interiors)
                        elif geom.geom_type == 'MultiPolygon':
                            rings = [ring for polygon in geom.geoms
                                    for ring in [polygon.exterior] + list(polygon.interiors)]
                        else:
                            rings = []
                        for ring in rings:
                            verts = [(c[0], c[1], c[2] if len(c) == 3 else 0) for c in ring.coords]
                            ax.add_collection3d(
                                Poly3DCollection([verts], alpha=0.5, edgecolor='grey', facecolor='lightblue')
                            )

            # overlay
            if overlay is not None and hasattr(overlay, "empty") and not overlay.empty:
                ox, oy, oz = zip(*[geom.coords[0] for geom in overlay.geometry])
                ax.scatter(ox, oy, oz, color='gray', s=5, alpha=0.5)

            # well path scatter
            sc_well = None
            if well_pts is not None and len(well_pts):
                if not has_well_values:
                    # just draw the well in black if no values
                    sc_well = ax.scatter(
                        well_pts[:, 0], well_pts[:, 1], well_pts[:, 2],
                        s=markersize * 1.6, color='k', alpha=0.9, zorder=5
                    )
                else:
                    vals = well_vals
                    if len(vals) > len(well_pts):
                        vals = vals[:len(well_pts)]
                    elif len(vals) < len(well_pts):
                        vals = np.concatenate([vals, np.full(len(well_pts) - len(vals), np.nan)])

                    w_cmap = plt.get_cmap(well_cmap)
                    _vmin_w = np.nanmin(vals) if well_vmin is None else well_vmin
                    _vmax_w = np.nanmax(vals) if well_vmax is None else well_vmax
                    norm_w = plt.Normalize(vmin=_vmin_w, vmax=_vmax_w)
                    sc_well = ax.scatter(
                        well_pts[:, 0], well_pts[:, 1], well_pts[:, 2],
                        s=markersize * 1.6, c=vals, cmap=w_cmap, norm=norm_w, alpha=0.9, zorder=5
                    )

            # area outline
            if area_outline is not None and hasattr(area_outline, "empty") and not area_outline.empty:
                if not gdf_copy.empty and gdf_copy.geometry.iloc[0].geom_type == 'Point':
                    zmax = max(geom.z for geom in gdf_copy.geometry)
                elif not gdf_copy.empty and gdf_copy.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
                    zmax = max(
                        max(coord[2] for coord in ring.coords if len(coord) == 3)
                        for geom in gdf_copy.geometry
                        for ring in ([geom.exterior] + list(geom.interiors))
                    )
                else:
                    zmax = 0
                for poly in area_outline.geometry:
                    xs, ys = zip(*[(c[0], c[1]) for c in poly.exterior.coords])
                    zs = [zmax + 1] * len(xs)
                    ax.plot(xs, ys, zs, color='black')

            # labels & limits
            _xlabel = xlabel if xlabel != 'default' else (gdf_copy.crs.axis_info[1].name if gdf_copy.crs else 'X-axis')
            _ylabel = ylabel if ylabel != 'default' else (gdf_copy.crs.axis_info[0].name if gdf_copy.crs else 'Y-axis')
            _zlabel = zlabel if zlabel else 'Z-axis'
            ax.set_xlabel(_xlabel)
            ax.set_ylabel(_ylabel)
            ax.set_zlabel(_zlabel)

            if extent is not None and zlim is None:
                ax.set_xlim(extent[0], extent[3])
                ax.set_ylim(extent[1], extent[4])
                ax.set_zlim(extent[2], extent[5])
            else:
                if xlim is not None: ax.set_xlim(xlim)
                if ylim is not None: ax.set_ylim(ylim)
                if zlim is not None: ax.set_zlim(zlim)

            ax.grid(True)

            # main colorbar (only once, into dedicated axis)
            if add_main_cbar and (col is not None and str(col).lower() != "none") and not gdf_filtered.empty:
                sm = plt.cm.ScalarMappable(cmap=cmap_main_obj, norm=norm_main)
                cax_main.cla()
                cb = plt.colorbar(sm, cax=cax_main)
                cb.set_label(units)
                cb.locator = MaxNLocator(nbins=6); cb.update_ticks()
                cb.ax.tick_params(labelsize=9)

            # well colorbar (only once, if we truly have well values and user wants it)
            if add_well_cbar and show_well_colorbar and has_well_values and sc_well is not None:
                cax_well.cla()
                cbw = plt.colorbar(sc_well, cax=cax_well)
                cbw.set_label(well_units)
                cbw.locator = MaxNLocator(nbins=6); cbw.update_ticks()
                cbw.ax.tick_params(labelsize=9)
                cbw.ax.yaxis.set_ticks_position('left')
                cbw.ax.yaxis.set_label_position('left')

        # plot all panels; add colorbars only once (NW panel)
        _plot_on(ax_nw, add_main_cbar=show_main_colorbar, add_well_cbar=True)
        _plot_on(ax_ne, add_main_cbar=False,               add_well_cbar=False)
        _plot_on(ax_sw, add_main_cbar=False,               add_well_cbar=False)
        _plot_on(ax_se_ax, add_main_cbar=False,            add_well_cbar=False)

        # clearer view labels
        ax_nw.set_title(f"{title} — looking NW")
        ax_ne.set_title(f"{title} — looking NE")
        ax_sw.set_title(f"{title} — looking SW")
        ax_se_ax.set_title(f"{title} — looking SE")

        plt.show()

    @staticmethod
    def plot_zoom_in(gdf, col, units, title, xlim, ylim, figsize, markersize, xlabel, ylabel, cmap):
        """Method to plot zoomed in version of geopfa maps, using xlim and ylim to determine the extent. 
        Also adds a basemap."""
        fig, ax = plt.subplots(figsize=figsize)
        if col is None or str(col).lower() == "none":
            gdf.plot(ax=ax)
        else:
            gdf.plot(ax=ax, marker='s', markersize=markersize,
                    column=col,cmap=cmap,legend=False, alpha=0.25)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gdf[col].min(), vmax=gdf[col].max()))
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(units)
        if xlabel == 'default':
            xlabel = gdf.crs.axis_info[1].name
        if ylabel == 'default':
            ylabel = gdf.crs.axis_info[0].name
        ## TODO: Basemap is causing problems. Fix at a later date.
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
        zi = griddata((x, y), z, (xi, yi), method='linear')

        fig, ax = plt.subplots(figsize=(10, 10))
        c = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='jet') 
        fig.colorbar(c, ax=ax, label=units)

        plt.title(f'{layer}: heatmap')
        plt.xlabel('easting (m)')
        plt.ylabel('northing (m)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


