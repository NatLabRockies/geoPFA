"""PFAGrid protocol and adapter helpers for the probabilistic runner.

The PFA dict used throughout geoPFA has a nested structure
``pfa["criteria"][criteria]["components"][component]["layers"][layer]``,
with per-component ``pr_norm`` and per-layer ``model`` GeoDataFrames. The
probabilistic engine only consumes a small slice of this structure. This
module provides:

* a thin :class:`PFAGridAdapter` wrapper that exposes the slice we need,
* free functions for callers that prefer direct dict access, and
* a :func:`validate_pfa_for_probabilistic` precondition check that fails
  loudly with the field name that is missing or malformed.

Both 2D (``Point(x, y)``) and 3D (``Point(x, y, z)``) PFA dicts are supported
with the same surface. The only behavioural difference is what
``extract_grid_extent`` returns: ``(xmin, ymin, xmax, ymax)`` for 2D vs
``(xmin, ymin, zmin, xmax, ymax, zmax)`` for 3D.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import geopandas as gpd
import numpy as np

from geopfa.exceptions import GEOPFAValueError


def component_names(pfa: dict, *, criteria: str) -> list[str]:
    """Return the sorted component names under ``pfa['criteria'][criteria]``."""
    return sorted(pfa["criteria"][criteria]["components"].keys())


def iter_components(
    pfa: dict, *, criteria: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(component_name, component_data)`` for each component."""
    yield from pfa["criteria"][criteria]["components"].items()


def component_layers(
    component_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the ``layers`` mapping for a component dict."""
    return component_data["layers"]


def layer_model_gdf(layer_data: dict[str, Any]) -> gpd.GeoDataFrame:
    """Return the ``model`` GeoDataFrame for a single layer."""
    return layer_data["model"]


def layer_data_column(layer_data: dict[str, Any]) -> str:
    """Return the column name on the layer ``model`` that holds the evidence values."""
    return layer_data.get("model_data_col", "value_interpolated")


def component_grid(comp: dict[str, Any]) -> gpd.GeoDataFrame | None:
    """Return the spatial grid GDF for a component.

    Prefers ``pr_norm`` (populated after VoterVeto) but falls back to the
    first layer's ``model`` GDF when VoterVeto has not been run.  This lets
    ``run_probabilistic`` work on raw processed PFA dicts without requiring
    a prior VoterVeto pass — the two methods are independent pathways.
    """
    grid = comp.get("pr_norm")
    if grid is not None and len(grid) > 0:
        return grid
    # Fall back to the first available layer model.
    for layer_data in comp.get("layers", {}).values():
        model = layer_data.get("model")
        if model is not None and len(model) > 0:
            return model
    return None


def _validate_component_grid_dimensions(
    grid: gpd.GeoDataFrame,
    *,
    path: str,
    dimensions: str,
) -> None:
    """Require one selected component grid to match the declared dimension."""
    geometry = grid.geometry
    if geometry.isna().any() or geometry.is_empty.any():
        raise ValueError(f"{path} geometries must be non-empty points")
    if not geometry.geom_type.eq("Point").all():
        raise ValueError(f"{path} geometries must contain only points")
    has_z = geometry.has_z.to_numpy(dtype=bool)
    if dimensions == "2d":
        if has_z.any():
            raise ValueError(
                f"{path} geometries must not carry a Z coordinate for 2D runs"
            )
        coordinates = np.column_stack(
            (
                geometry.x.to_numpy(dtype=float),
                geometry.y.to_numpy(dtype=float),
            )
        )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"{path} geometries require finite X and Y")
        return
    if dimensions != "3d":
        raise ValueError("dimensions must be '2d' or '3d'")
    if not has_z.all():
        raise ValueError(
            f"{path} geometries must carry a Z coordinate on every row for 3D runs"
        )
    coordinates = np.column_stack(
        (
            geometry.x.to_numpy(dtype=float),
            geometry.y.to_numpy(dtype=float),
            geometry.z.to_numpy(dtype=float),
        )
    )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{path} geometries require finite X, Y, and Z")


def extract_grid_extent(
    pfa: dict, *, criteria: str, dimensions: str
) -> tuple[float, ...]:
    """Compute the bounding extent of the PFA grid across all components.

    Parameters
    ----------
    pfa
        The geoPFA dict.
    criteria
        Criteria key (e.g. ``"geologic"``).
    dimensions
        ``"2d"`` or ``"3d"``.

    Returns
    -------
    tuple of float
        ``(xmin, ymin, xmax, ymax)`` for 2D or
        ``(xmin, ymin, zmin, xmax, ymax, zmax)`` for 3D.
    """
    if dimensions not in {"2d", "3d"}:
        raise ValueError("dimensions must be '2d' or '3d'")
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for comp_name, comp in iter_components(pfa, criteria=criteria):
        grid = component_grid(comp)
        if grid is None or len(grid) == 0:
            continue
        _validate_component_grid_dimensions(
            grid,
            path=f"criteria/{criteria}/components/{comp_name} component grid",
            dimensions=dimensions,
        )
        xs.extend([float(grid.geometry.x.min()), float(grid.geometry.x.max())])
        ys.extend([float(grid.geometry.y.min()), float(grid.geometry.y.max())])
        if dimensions == "3d":
            zs.extend(
                [float(grid.geometry.z.min()), float(grid.geometry.z.max())]
            )
    if not xs:
        raise ValueError(
            "no components have a non-empty grid (pr_norm or layer model)"
        )
    if dimensions == "3d":
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    return (min(xs), min(ys), max(xs), max(ys))


def validate_pfa_for_probabilistic(
    pfa: dict, *, criteria: str, dimensions: str
) -> None:
    """Validate that ``pfa`` has the structure the probabilistic engine needs.

    Raises ``KeyError`` (missing field), ``TypeError`` (wrong type), or
    ``ValueError`` (wrong shape) on the first problem encountered, with the
    path that is broken. This is the precondition check the runner calls
    before doing any work.

    Notes
    -----
    ``pr_norm`` is NOT required.  It is populated by the VoterVeto layer-
    combination pathway, but ``run_probabilistic`` is an independent pathway
    that derives its spatial grid from layer ``model`` GeoDataFrames directly.
    When ``pr_norm`` is present it is used; when absent the first layer model
    is used as the grid (see ``component_grid``).
    """
    if criteria not in pfa.get("criteria", {}):
        raise KeyError(
            f"PFA dict is missing criteria block: criteria/{criteria}",
        )
    crit = pfa["criteria"][criteria]
    if "components" not in crit:
        raise KeyError(
            f"criteria/{criteria}/components is missing",
        )
    if not crit["components"]:
        raise ValueError(f"criteria/{criteria}/components is empty")
    for comp_name, comp_data in crit["components"].items():
        # Validate pr_norm only when present — it is optional.
        if "pr_norm" in comp_data:
            grid = comp_data["pr_norm"]
            if not isinstance(grid, gpd.GeoDataFrame):
                raise TypeError(
                    f"criteria/{criteria}/components/{comp_name}/pr_norm must be "
                    "a GeoDataFrame",
                )
        layers = comp_data.get("layers", {})
        if not layers:
            raise KeyError(
                f"criteria/{criteria}/components/{comp_name}/layers is missing "
                "or empty",
            )
        for layer_name, layer_data in layers.items():
            if "model" not in layer_data:
                raise KeyError(
                    f"criteria/{criteria}/components/{comp_name}/layers/"
                    f"{layer_name}/model is missing",
                )
            if not isinstance(layer_data["model"], gpd.GeoDataFrame):
                raise TypeError(
                    f"criteria/{criteria}/components/{comp_name}/layers/"
                    f"{layer_name}/model must be a GeoDataFrame",
                )
        grid = component_grid(comp_data)
        if grid is None:
            raise ValueError(
                f"criteria/{criteria}/components/{comp_name} has no non-empty "
                "component grid"
            )
        _validate_component_grid_dimensions(
            grid,
            path=f"criteria/{criteria}/components/{comp_name} component grid",
            dimensions=dimensions,
        )


class PFAGridAdapter:
    """Thin OO wrapper over a PFA dict for the probabilistic engine.

    Equivalent to using the free functions in this module, but keeps the
    ``(pfa, criteria, dimensions)`` triple bound so callers don't have to
    pass them around.
    """

    def __init__(
        self,
        pfa: dict,
        *,
        criteria: str = "geologic",
        dimensions: str = "2d",
    ) -> None:
        """Bind to a PFA dict and validate it for the requested mode."""
        validate_pfa_for_probabilistic(
            pfa, criteria=criteria, dimensions=dimensions
        )
        self._pfa = pfa
        self._criteria = criteria
        self._dimensions = dimensions

    @property
    def pfa(self) -> dict:
        """Underlying PFA dict (mutable; do not mutate via this property)."""
        return self._pfa

    @property
    def criteria(self) -> str:
        """Bound criteria key."""
        return self._criteria

    @property
    def dimensions(self) -> str:
        """``'2d'`` or ``'3d'``."""
        return self._dimensions

    def components(self) -> list[str]:
        """Sorted list of component names."""
        return component_names(self._pfa, criteria=self._criteria)

    def component_data(self, component: str) -> dict[str, Any]:
        """The component-level dict (with ``pr_norm``, ``layers``, ``pr0``, …)."""
        return self._pfa["criteria"][self._criteria]["components"][component]

    def pr_norm(self, component: str) -> gpd.GeoDataFrame:
        """The component's spatial grid GeoDataFrame.

        Returns ``pr_norm`` when present (populated by VoterVeto), otherwise
        falls back to the first layer's ``model`` GDF.  VoterVeto is not a
        prerequisite of ``run_probabilistic``; both are independent pathways.
        """
        comp = self.component_data(component)
        grid = component_grid(comp)
        if grid is None:
            raise ValueError(
                f"Component '{component}' has no usable grid: "
                "pr_norm is absent and no layer model GDFs are available."
            )
        return grid

    def layers(self, component: str) -> list[str]:
        """Layer names under the given component."""
        return list(self.component_data(component)["layers"].keys())

    def layer_data(self, component: str, layer: str) -> dict[str, Any]:
        """The per-layer dict (with ``model``, ``model_data_col``, …)."""
        return self.component_data(component)["layers"][layer]

    def layer_model(self, component: str, layer: str) -> gpd.GeoDataFrame:
        """The per-layer ``model`` GeoDataFrame."""
        return layer_model_gdf(self.layer_data(component, layer))

    def layer_value_column(self, component: str, layer: str) -> str:
        """The column name to read evidence values from on the layer ``model``."""
        return layer_data_column(self.layer_data(component, layer))

    def extent(self) -> tuple[float, ...]:
        """The grid extent across all components (see :func:`extract_grid_extent`)."""
        return extract_grid_extent(
            self._pfa, criteria=self._criteria, dimensions=self._dimensions
        )


def validate_declared_components(
    adapter: PFAGridAdapter,
    declared_components: set[str],
) -> None:
    """Require every configured analysis component to exist in the PFA tree."""
    missing = declared_components - set(adapter.components())
    if missing:
        raise GEOPFAValueError(
            "configured probabilistic component(s) are absent from the PFA "
            "criteria tree: " + ", ".join(sorted(missing))
        )


__all__ = [
    "PFAGridAdapter",
    "component_grid",
    "component_layers",
    "component_names",
    "extract_grid_extent",
    "iter_components",
    "layer_data_column",
    "layer_model_gdf",
    "validate_declared_components",
    "validate_pfa_for_probabilistic",
]
