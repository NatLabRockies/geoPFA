"""3D conceptual model visualization for PFA outputs using PyVista.

All public methods in this module require Point-Z (3-D) geometries.
Passing 2-D geometries raises a ValueError.  For 2-D map views use
``GeospatialDataPlotters`` in ``geopfa.plotters``.  2-D conceptual
modeling may be implemented in a future release.
"""

import warnings
from contextlib import suppress

import numpy as np
import pyvista as pv
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _coords3_from_point(pt):
    try:
        z = pt.z
    except Exception:
        c0 = pt.coords[0]
        z = c0[2] if len(c0) == 3 else 0.0  # noqa: PLR2004
    return (pt.x, pt.y, z)


def _require_3d(gdf, fn_name):
    """Raise ValueError if any geometry in *gdf* lacks a Z coordinate."""
    if not all(getattr(g, "has_z", False) for g in gdf.geometry):
        raise ValueError(
            f"{fn_name}: all geometries must have Z coordinates. "
            "This module renders 3-D iso-surfaces and requires Point-Z data. "
            "For 2-D maps use GeospatialDataPlotters in geopfa.plotters."
        )


def _build_well_pts(well):  # noqa: PLR0911
    """Return an (N, 3) float array from a well-path GeoDataFrame or geometry."""
    if well is None:
        return None
    if hasattr(well, "geometry"):
        if len(well.geometry) == 0:
            return None
        if all(g.geom_type == "Point" for g in well.geometry):
            return np.array(
                [_coords3_from_point(p) for p in well.geometry], dtype=float
            )
        geoms = list(well.geometry)
        merged = geoms[0]
        if len(geoms) > 1:
            with suppress(Exception):
                merged = linemerge(geoms)
        if isinstance(merged, LineString | MultiLineString):
            parts = (
                merged.geoms
                if isinstance(merged, MultiLineString)
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
    if isinstance(well, LineString | MultiLineString):
        parts = well.geoms if isinstance(well, MultiLineString) else [well]
        arrs = []
        for ls in parts:
            arr = np.asarray(ls.coords, dtype=float)
            if arr.shape[1] == 2:  # noqa: PLR2004
                arr = np.c_[arr, np.zeros(len(arr))]
            arrs.append(arr)
        return np.vstack(arrs) if arrs else None
    return None


def _apply_slices(arr, x_slice=None, y_slice=None, z_slice=None):
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


def _clip_pts_to_bounds(arr, bounds):
    """Clip an (N, 3) point array to a (xmin, xmax, ymin, ymax, zmin, zmax) bounding box."""
    if arr is None or len(arr) == 0:
        return arr
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    mask = (
        (arr[:, 0] >= xmin)
        & (arr[:, 0] <= xmax)
        & (arr[:, 1] >= ymin)
        & (arr[:, 1] <= ymax)
        & (arr[:, 2] >= zmin)
        & (arr[:, 2] <= zmax)
    )
    return arr[mask]


def _infer_spacing(arr):
    unique = np.unique(np.sort(arr))
    if len(unique) <= 1:
        return 1.0
    return float(np.median(np.diff(unique)))


def _camera_from_view(bounds, elev_deg, azim_deg):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array(
        [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    )
    diag_xy = np.hypot(xmax - xmin, ymax - ymin) or 1.0
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    dir_vec = np.array(
        [-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), np.sin(el)]
    )
    dir_vec /= np.linalg.norm(dir_vec)
    return [
        (center + dir_vec * 2.2 * diag_xy).tolist(),
        center.tolist(),
        (0.0, 0.0, 1.0),
    ]


def _build_image_data(xs, ys, zs, extent=None):  # noqa: PLR0914
    """Build a PyVista ImageData grid from point cloud coordinates.

    Returns the grid plus the integer index arrays (ix, iy, iz) that map each
    input point to its nearest voxel.
    """
    if extent is not None:
        xmin, ymin, zmin, xmax, ymax, zmax = extent
    else:
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        zmin, zmax = float(zs.min()), float(zs.max())

    dx = _infer_spacing(xs)
    dy = _infer_spacing(ys)
    dz = _infer_spacing(zs)

    x = np.arange(xmin, xmax + 0.5 * dx, dx)
    y = np.arange(ymin, ymax + 0.5 * dy, dy)
    z = np.arange(zmin, zmax + 0.5 * dz, dz)
    nx, ny, nz = len(x), len(y), len(z)

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (float(x[0]), float(y[0]), float(z[0]))
    grid.spacing = (dx, dy, dz)

    ix = np.clip(np.rint((xs - x[0]) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((ys - y[0]) / dy).astype(int), 0, ny - 1)
    iz = np.clip(np.rint((zs - z[0]) / dz).astype(int), 0, nz - 1)

    return grid, ix, iy, iz, (nx, ny, nz)


def _build_outline_mesh(area_outline, zmax):
    """Build a PyVista line mesh from a polygon GeoDataFrame drawn at zmax."""
    all_points = []
    all_lines = []
    offset = 0
    for geom in area_outline.geometry:
        if geom is None:
            continue
        polys = (
            list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        )
        for poly in polys:
            coords = np.array(poly.exterior.coords, dtype=float)
            n = len(coords)
            pts = np.c_[coords[:, 0], coords[:, 1], np.full(n, zmax)]
            all_points.append(pts)
            all_lines.extend(
                [[2, offset + i, offset + i + 1] for i in range(n - 1)]
            )
            offset += n
    if not all_points:
        return None
    mesh = pv.PolyData()
    mesh.points = np.vstack(all_points)
    mesh.lines = np.array(all_lines, dtype=int).ravel()
    return mesh


def _add_well_to_plotter(  # noqa: PLR0913, PLR0917
    p,
    wp,
    well_path_values,
    well_units,
    well_cmap,
    well_vmin,
    well_vmax,
    markersize,
    show_colorbar,
    bar_x,
    bar_y,
    bar_width,
    bar_height,
    colorbar_title_font_size=14,
    colorbar_label_font_size=12,
):
    """Add a well-path point cloud to an existing PyVista plotter."""
    if wp is None or len(wp) == 0:
        return
    well_poly = pv.PolyData(wp)
    has_values = well_path_values is not None and np.any(
        np.isfinite(well_path_values)
    )
    if has_values:
        wv = np.asarray(well_path_values, float)[: len(wp)]
        well_poly[well_units] = wv
        w_vmin = np.nanmin(wv) if well_vmin is None else well_vmin
        w_vmax = np.nanmax(wv) if well_vmax is None else well_vmax
        sbar = (
            {
                "title": well_units,
                "n_labels": 6,
                "vertical": True,
                "position_x": bar_x,
                "position_y": bar_y,
                "width": bar_width,
                "height": bar_height,
                "title_font_size": colorbar_title_font_size,
                "label_font_size": colorbar_label_font_size,
            }
            if show_colorbar
            else {"title": ""}
        )
        p.add_mesh(
            well_poly,
            scalars=well_units,
            render_points_as_spheres=True,
            point_size=markersize,
            cmap=well_cmap,
            clim=(w_vmin, w_vmax),
            scalar_bar_args=sbar,
        )
    else:
        p.add_mesh(
            well_poly,
            render_points_as_spheres=True,
            point_size=markersize,
            color="black",
            show_scalar_bar=False,
        )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ConceptualModeling:
    """Visualization tools for 3D conceptual geothermal resource models.

    Methods
    -------
    plot_isosurface
        Render a single-component iso-surface from a 3D favorability model.
    plot_conceptual_model
        Render multi-component iso-surfaces for conceptual model visualization.
    """

    @staticmethod
    def plot_isosurface(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
        gdf,
        col,
        units,
        title,
        contour_level,
        *,
        area_outline=None,
        well_path=None,
        well_path_values=None,
        well_units="Temperature (°C)",
        well_cmap="magma",
        well_vmin=None,
        well_vmax=None,
        overlay_points=None,
        extent=None,
        x_slice=None,
        y_slice=None,
        z_slice=None,
        cmap="jet",
        opacity=0.6,
        markersize=10,
        vmin=None,
        vmax=None,
        view=(45, 45),
        off_screen=True,
        screenshot_path="isosurface.png",
        window_size=(1024, 768),
        title_font_size=12,
        colorbar_title_font_size=14,
        colorbar_label_font_size=12,
    ):
        """Render a single iso-surface of a 3D point-cloud favorability model.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Point-Z GeoDataFrame containing the scalar field.
        col : str
            Column name of the scalar field to contour.
        units : str
            Label for the main scalar colorbar.
        title : str
            Plot title.
        contour_level : float
            Iso-value at which to extract the surface.
        area_outline : geopandas.GeoDataFrame, optional
            Polygon GeoDataFrame drawn as a boundary line at the top of the
            volume (e.g. a project or study area boundary).
        well_path : geopandas.GeoDataFrame, optional
            Well-path GeoDataFrame (Point-Z geometries).
        well_path_values : array-like, optional
            Scalar values (e.g. temperature) along the well path.
        well_units : str, optional
            Colorbar label for the well-path scalars.
        well_cmap : str, optional
            Colormap for the well-path scalars.
        well_vmin, well_vmax : float, optional
            Colorbar limits for the well-path scalars.
        overlay_points : geopandas.GeoDataFrame, optional
            Point-Z GeoDataFrame rendered as small black spheres (e.g. earthquakes).
        extent : list, optional
            Bounding box ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
        x_slice, y_slice, z_slice : float, optional
            Clip data to coordinates <= the given value along each axis.
        cmap : str, optional
            Colormap for the iso-surface.
        opacity : float, optional
            Opacity of the iso-surface (0-1).
        markersize : float, optional
            Sphere size for well-path points.
        vmin, vmax : float, optional
            Colorbar limits for the iso-surface scalar.
        view : tuple, optional
            Camera ``(elevation_deg, azimuth_deg)``.
        off_screen : bool, optional
            If True, render off-screen and save to ``screenshot_path``.
        screenshot_path : str, optional
            File path for the saved screenshot when ``off_screen`` is True.

        Returns
        -------
        dict
            Keys: ``"grid"``, ``"iso"``, ``"plotter"``.
        """
        if gdf is None or len(gdf) == 0:
            warnings.warn("plot_isosurface: empty GeoDataFrame.", stacklevel=2)
            return None
        if col not in gdf.columns:
            raise ValueError(f"plot_isosurface: column '{col}' not found.")
        if not all(g.geom_type == "Point" for g in gdf.geometry):
            raise ValueError("plot_isosurface: expects Point-Z geometries.")
        _require_3d(gdf, "plot_isosurface")

        coords = np.array(
            [_coords3_from_point(p) for p in gdf.geometry], dtype=float
        )
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        vals = gdf[col].astype(float).to_numpy()

        mask = np.ones(len(xs), dtype=bool)
        if x_slice is not None:
            mask &= xs <= x_slice
        if y_slice is not None:
            mask &= ys <= y_slice
        if z_slice is not None:
            mask &= zs <= z_slice
        xs, ys, zs, vals = xs[mask], ys[mask], zs[mask], vals[mask]

        if len(xs) == 0:
            warnings.warn(
                "plot_isosurface: no data after slicing.", stacklevel=2
            )
            return None

        vmin_use = float(np.nanmin(vals)) if vmin is None else float(vmin)
        vmax_use = float(np.nanmax(vals)) if vmax is None else float(vmax)
        if (
            not np.isfinite(vmin_use)
            or not np.isfinite(vmax_use)
            or vmin_use == vmax_use
        ):
            vmin_use, vmax_use = 0.0, 1.0

        grid, ix, iy, iz, (nx, ny, nz) = _build_image_data(xs, ys, zs, extent)
        vol = np.full((nx, ny, nz), np.nan, dtype=float)
        for i, j, k, v in zip(ix, iy, iz, vals):
            vol[i, j, k] = v
        grid[col] = np.ascontiguousarray(vol).ravel(order="F")

        bar_y, bar_width, bar_height = 0.11, 0.028, 0.74

        p = pv.Plotter(off_screen=off_screen, window_size=window_size)
        p.add_text(title, position="upper_edge", font_size=title_font_size)

        iso = grid.contour(isosurfaces=[float(contour_level)], scalars=col)
        if iso.n_points > 0:
            p.add_mesh(
                iso,
                scalars=col,
                cmap=cmap,
                clim=(vmin_use, vmax_use),
                opacity=opacity,
                scalar_bar_args={
                    "title": units,
                    "n_labels": 6,
                    "vertical": True,
                    "position_x": 0.845,
                    "position_y": bar_y,
                    "width": bar_width,
                    "height": bar_height,
                    "title_font_size": colorbar_title_font_size,
                    "label_font_size": colorbar_label_font_size,
                },
            )

        wp = _apply_slices(
            _build_well_pts(well_path), x_slice, y_slice, z_slice
        )
        _add_well_to_plotter(
            p,
            wp,
            well_path_values,
            well_units,
            well_cmap,
            well_vmin,
            well_vmax,
            markersize,
            show_colorbar=True,
            bar_x=0.905,
            bar_y=bar_y,
            bar_width=bar_width,
            bar_height=bar_height,
            colorbar_title_font_size=colorbar_title_font_size,
            colorbar_label_font_size=colorbar_label_font_size,
        )

        if overlay_points is not None:
            op = _clip_pts_to_bounds(
                _apply_slices(
                    _build_well_pts(overlay_points), x_slice, y_slice, z_slice
                ),
                grid.bounds,
            )
            if op is not None and len(op) > 0:
                p.add_mesh(
                    pv.PolyData(op),
                    color="black",
                    render_points_as_spheres=True,
                    point_size=4,
                    show_scalar_bar=False,
                )

        if area_outline is not None:
            outline_mesh = _build_outline_mesh(area_outline, grid.bounds[5])
            if outline_mesh is not None:
                p.add_mesh(
                    outline_mesh,
                    color="black",
                    line_width=2,
                    show_scalar_bar=False,
                )

        p.add_mesh(
            grid.outline(), color="black", line_width=1, show_scalar_bar=False
        )
        with suppress(Exception):
            p.show_bounds(
                grid="front",
                location="outer",
                all_edges=True,
                ticks="outside",
                font_size=colorbar_label_font_size,
            )
        if hasattr(p, "add_axes"):
            p.add_axes()

        with suppress(Exception):
            p.camera_position = _camera_from_view(grid.bounds, *view)
        p.reset_camera()

        if off_screen:
            p.show(screenshot=screenshot_path)
            print(f"plot_isosurface: saved to {screenshot_path}")
        else:
            p.show()

        return {"grid": grid, "iso": iso, "plotter": p}

    @staticmethod
    def plot_conceptual_model(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
        gdf,
        cols,
        units,
        title,
        contour_levels,
        *,
        area_outline=None,
        well_path=None,
        well_path_values=None,
        well_units="Temperature (°C)",
        well_cmap="magma",
        well_vmin=None,
        well_vmax=None,
        show_well_colorbar=True,
        overlay_points=None,
        extent=None,
        x_slice=None,
        y_slice=None,
        z_slice=None,
        filter_threshold=None,
        component_colors=None,
        opacity=0.5,
        markersize=15,
        vmin=None,
        vmax=None,
        view=(15, 180),
        off_screen=True,
        screenshot_path="conceptual_model.png",
        window_size=(1024, 768),
        title_font_size=12,
        colorbar_title_font_size=14,
        colorbar_label_font_size=12,
    ):
        """Render multi-component iso-surfaces for a 3D conceptual resource model.

        Each column in ``cols`` is rendered as a separate iso-surface coloured by
        ``component_colors``.  When a single column is provided the surface is
        coloured by scalar value instead.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Point-Z GeoDataFrame containing one or more scalar fields.
        cols : str or list of str
            Column name(s) to render.  Each column produces one iso-surface.
        units : str
            Colorbar label (single-component mode only).
        title : str
            Plot title.
        contour_levels : float or list
            Iso-value(s) for each component.  If a scalar or 1-D array is
            provided it is broadcast to all components.  If a list of the
            same length as ``cols`` is provided each element is used for the
            corresponding component.
        area_outline : geopandas.GeoDataFrame, optional
            Polygon GeoDataFrame drawn as a boundary line at the top of the
            volume (e.g. a project or study area boundary).
        well_path : geopandas.GeoDataFrame, optional
            Well-path GeoDataFrame (Point-Z geometries).
        well_path_values : array-like, optional
            Scalar values (e.g. temperature) along the well path.
        well_units : str, optional
            Colorbar label for the well-path scalars.
        well_cmap : str, optional
            Colormap for the well-path scalars.
        well_vmin, well_vmax : float, optional
            Colorbar limits for the well-path scalars.
        show_well_colorbar : bool, optional
            Display the well-path scalar colorbar.
        overlay_points : geopandas.GeoDataFrame, optional
            Point-Z GeoDataFrame rendered as small black spheres (e.g. earthquakes).
        extent : list, optional
            Bounding box ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
        x_slice, y_slice, z_slice : float, optional
            Clip data to coordinates <= the given value along each axis.
        filter_threshold : float, optional
            Threshold for a semi-transparent volume overlay on the first column.
        component_colors : list of str, optional
            One colour per column.  Defaults to red / dodgerblue / goldenrod / …
        opacity : float, optional
            Opacity of iso-surfaces (0-1).
        markersize : float, optional
            Sphere size for well-path points.
        vmin, vmax : float, optional
            Colorbar limits (single-component mode only).
        view : tuple, optional
            Camera ``(elevation_deg, azimuth_deg)``.
        off_screen : bool, optional
            If True, render off-screen and save to ``screenshot_path``.
        screenshot_path : str, optional
            File path for the saved screenshot when ``off_screen`` is True.

        Returns
        -------
        dict
            Keys: ``"grid"``, ``"grid_clipped"``, ``"iso_components"``,
            ``"high_vol"``, ``"plotter"``.
        """
        if gdf is None or gdf.empty:
            warnings.warn(
                "plot_conceptual_model: empty GeoDataFrame.", stacklevel=2
            )
            return None

        cols = [cols] if isinstance(cols, str) else list(cols)
        if not cols:
            raise ValueError(
                "plot_conceptual_model: 'cols' must contain at least one column."
            )
        for c in cols:
            if c not in gdf.columns:
                raise ValueError(
                    f"plot_conceptual_model: column '{c}' not found."
                )

        single = len(cols) == 1
        main_col = cols[0]
        if not all(g.geom_type == "Point" for g in gdf.geometry):
            raise ValueError(
                "plot_conceptual_model: expects Point-Z geometries."
            )
        _require_3d(gdf, "plot_conceptual_model")

        coords_full = np.array(
            [_coords3_from_point(p) for p in gdf.geometry], dtype=float
        )
        xs_full, ys_full, zs_full = (
            coords_full[:, 0],
            coords_full[:, 1],
            coords_full[:, 2],
        )

        mask = np.ones(len(xs_full), dtype=bool)
        if x_slice is not None:
            mask &= xs_full <= x_slice
        if y_slice is not None:
            mask &= ys_full <= y_slice
        if z_slice is not None:
            mask &= zs_full <= z_slice

        if not mask.any():
            warnings.warn(
                "plot_conceptual_model: no data after slicing.", stacklevel=2
            )
            return None

        xs, ys, zs = xs_full[mask], ys_full[mask], zs_full[mask]
        grid, ix, iy, iz, (nx, ny, nz) = _build_image_data(xs, ys, zs, extent)

        for cname in cols:
            vals_full = gdf[cname].astype(float).to_numpy()
            vals = vals_full[mask]
            vol = np.full((nx, ny, nz), np.nan, dtype=float)
            for i, j, k, v in zip(ix, iy, iz, vals):
                vol[i, j, k] = v
            grid[cname] = np.ascontiguousarray(vol).ravel(order="F")

        grid_c = grid.clip_box(list(grid.bounds), invert=False)
        if grid_c.n_points == 0:
            warnings.warn(
                "plot_conceptual_model: no points after clipping.",
                stacklevel=2,
            )
            return None

        scalar_ranges = {}
        for cname in cols:
            cv = grid_c[cname]
            cv = cv[np.isfinite(cv)]
            scalar_ranges[cname] = (
                (float(cv.min()), float(cv.max()))
                if cv.size
                else (np.nan, np.nan)
            )

        vmin_main, vmax_main = scalar_ranges[main_col]
        if vmin is not None:
            vmin_main = float(vmin)
        if vmax is not None:
            vmax_main = float(vmax)
        if (
            not np.isfinite(vmin_main)
            or not np.isfinite(vmax_main)
            or vmin_main == vmax_main
        ):
            vmin_main, vmax_main = 0.0, 1.0

        # build per-component contour level arrays
        contour_by_comp = {}
        if isinstance(contour_levels, list | tuple) and len(
            contour_levels
        ) == len(cols):
            for cname, lev in zip(cols, contour_levels):
                vmin_c, vmax_c = scalar_ranges[cname]
                arr = np.atleast_1d(np.asarray(lev, dtype=float))
                if np.isfinite(vmin_c) and np.isfinite(vmax_c):
                    arr = arr[(arr >= vmin_c) & (arr <= vmax_c)]
                contour_by_comp[cname] = np.unique(np.round(arr, 6))
        else:
            gl = np.atleast_1d(np.asarray(contour_levels, dtype=float))
            for cname in cols:
                vmin_c, vmax_c = scalar_ranges[cname]
                arr = gl.copy()
                if np.isfinite(vmin_c) and np.isfinite(vmax_c):
                    arr = arr[(arr >= vmin_c) & (arr <= vmax_c)]
                contour_by_comp[cname] = np.unique(np.round(arr, 6))

        if component_colors is None:
            defaults = [
                "red",
                "dodgerblue",
                "goldenrod",
                "purple",
                "turquoise",
            ]
            component_colors = [
                defaults[i % len(defaults)] for i in range(len(cols))
            ]

        bar_y, bar_width, bar_height = 0.11, 0.028, 0.74

        p = pv.Plotter(off_screen=off_screen, window_size=window_size)
        p.add_text(title, position="upper_edge", font_size=title_font_size)

        iso_components = {}
        high_vol = None

        for idx, cname in enumerate(cols):
            levels = contour_by_comp.get(cname, np.array([], dtype=float))
            if levels.size == 0 or not np.isfinite(grid_c[cname]).any():
                continue
            iso = grid_c.contour(
                isosurfaces=list(map(float, levels)), scalars=cname
            )
            if iso.n_points > 0:
                if single:
                    p.add_mesh(
                        iso,
                        scalars=cname,
                        cmap="jet",
                        clim=(vmin_main, vmax_main),
                        opacity=opacity,
                        scalar_bar_args={
                            "title": units,
                            "n_labels": 6,
                            "vertical": True,
                            "position_x": 0.845,
                            "position_y": bar_y,
                            "width": bar_width,
                            "height": bar_height,
                            "title_font_size": colorbar_title_font_size,
                            "label_font_size": colorbar_label_font_size,
                        },
                    )
                else:
                    p.add_mesh(
                        iso,
                        color=component_colors[idx],
                        opacity=opacity,
                        show_scalar_bar=False,
                    )
                iso_components[cname] = iso

        if filter_threshold is not None and np.isfinite(vmin_main):
            hv = grid_c.threshold(
                filter_threshold, scalars=main_col, invert=False
            )
            if hv.n_cells > 0:
                high_vol = hv
                p.add_mesh(
                    high_vol,
                    opacity=0.20,
                    color="white",
                    show_scalar_bar=False,
                )

        wp = _apply_slices(
            _build_well_pts(well_path), x_slice, y_slice, z_slice
        )
        _add_well_to_plotter(
            p,
            wp,
            well_path_values,
            well_units,
            well_cmap,
            well_vmin,
            well_vmax,
            markersize,
            show_colorbar=show_well_colorbar,
            bar_x=0.875,
            bar_y=bar_y,
            bar_width=bar_width,
            bar_height=bar_height,
            colorbar_title_font_size=colorbar_title_font_size,
            colorbar_label_font_size=colorbar_label_font_size,
        )

        if overlay_points is not None:
            op = _clip_pts_to_bounds(
                _apply_slices(
                    _build_well_pts(overlay_points), x_slice, y_slice, z_slice
                ),
                grid.bounds,
            )
            if op is not None and len(op) > 0:
                p.add_mesh(
                    pv.PolyData(op),
                    color="black",
                    render_points_as_spheres=True,
                    point_size=4,
                    show_scalar_bar=False,
                )

        if area_outline is not None:
            outline_mesh = _build_outline_mesh(area_outline, grid_c.bounds[5])
            if outline_mesh is not None:
                p.add_mesh(
                    outline_mesh,
                    color="black",
                    line_width=2,
                    show_scalar_bar=False,
                )

        p.add_mesh(
            grid_c.outline(),
            color="black",
            line_width=1,
            show_scalar_bar=False,
        )
        with suppress(Exception):
            p.show_bounds(
                grid="front",
                location="outer",
                all_edges=True,
                ticks="outside",
                font_size=colorbar_label_font_size,
            )
        if hasattr(p, "add_axes"):
            p.add_axes()

        with suppress(Exception):
            p.camera_position = _camera_from_view(grid_c.bounds, *view)
        p.reset_camera()

        if off_screen:
            p.show(screenshot=screenshot_path)
            print(f"plot_conceptual_model: saved to {screenshot_path}")
        else:
            p.show()

        return {
            "grid": grid,
            "grid_clipped": grid_c,
            "iso_components": iso_components,
            "high_vol": high_vol,
            "plotter": p,
        }
