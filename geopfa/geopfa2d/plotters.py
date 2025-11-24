# -*- coding: utf-8 -*-
"""
Set of methods to read in data in various formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import contextily as ctx

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
