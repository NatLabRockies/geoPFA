"""Synthetic 3D geoPFA-style dataset fixtures for unit tests.

3D analogue of ``synthetic_prob.make_synthetic_pfa``: a voxelised PFA dict
with depth-varying priors and 3D well points so the probabilistic engine can
be exercised end-to-end without depending on real geological datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from itertools import starmap


@dataclass
class SyntheticPFA3D:
    """Container for a synthetic 3D two-component PFA-like fixture."""

    pfa: dict
    wells: gpd.GeoDataFrame
    rng: np.random.Generator


def _make_voxel_grid(
    nx: int, ny: int, nz: int, *, crs: str = "EPSG:32611"
) -> gpd.GeoDataFrame:
    """Build a regular ``nx x ny x nz`` voxel grid as a 3D point GeoDataFrame."""
    xs = np.linspace(500_000.0, 600_000.0, nx)
    ys = np.linspace(4_300_000.0, 4_400_000.0, ny)
    zs = np.linspace(-3000.0, -500.0, nz)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    geom = list(starmap(Point, coords))
    return gpd.GeoDataFrame({"geometry": geom}, crs=crs)


def _depth_bump(  # noqa: PLR0913
    coords: np.ndarray,
    *,
    cx: float,
    cy: float,
    cz: float,
    sigma_xy: float,
    sigma_z: float,
    rng: np.random.Generator,
    noise: float = 0.05,
) -> np.ndarray:
    """3D smooth bump centred on ``(cx, cy, cz)`` plus Gaussian noise."""
    dx = (coords[:, 0] - cx) / sigma_xy
    dy = (coords[:, 1] - cy) / sigma_xy
    dz = (coords[:, 2] - cz) / sigma_z
    field = np.exp(-(dx * dx + dy * dy + dz * dz))
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    return field + rng.normal(0.0, noise, size=field.shape)


def make_synthetic_pfa_3d(  # noqa: PLR0914
    *,
    grid_n: int = 6,
    grid_nz: int = 4,
    n_wells: int = 40,
    seed: int = 0,
    crs: str = "EPSG:32611",
) -> SyntheticPFA3D:
    """Construct a deterministic two-component synthetic 3D PFA fixture.

    The fixture has:

    * a ``grid_n x grid_n x grid_nz`` voxel model grid in the given CRS,
    * two components named ``component_a`` and ``component_b``, each with a
      depth-varying ``prior_layer_*`` and a shared ``gradient`` 3D layer,
    * a ``pr_norm`` GeoDataFrame with a ``favorability`` column on the same
      voxel grid,
    * ``n_wells`` labelled wells with 3D point geometries and per-component
      label columns whose distributions correlate with the prior bumps.

    All numeric layers are framework-generic and unit-less.
    """
    rng = np.random.default_rng(seed)
    grid = _make_voxel_grid(grid_n, grid_n, grid_nz, crs=crs)
    coords = np.column_stack(
        [
            grid.geometry.x.to_numpy(dtype=float),
            grid.geometry.y.to_numpy(dtype=float),
            grid.geometry.z.to_numpy(dtype=float),
        ]
    )
    n_cells = len(grid)

    prior_a = _depth_bump(
        coords,
        cx=540_000.0,
        cy=4_360_000.0,
        cz=-1500.0,
        sigma_xy=25_000.0,
        sigma_z=1500.0,
        rng=rng,
    )
    prior_b = _depth_bump(
        coords,
        cx=570_000.0,
        cy=4_350_000.0,
        cz=-2000.0,
        sigma_xy=25_000.0,
        sigma_z=1500.0,
        rng=rng,
    )
    gradient = _depth_bump(
        coords,
        cx=560_000.0,
        cy=4_350_000.0,
        cz=-1750.0,
        sigma_xy=40_000.0,
        sigma_z=2000.0,
        rng=rng,
    )

    def _layer(values: np.ndarray) -> dict:
        df = grid.copy()
        df["value_interpolated"] = values
        return {
            "data_col": "value",
            "model_data_col": "value_interpolated",
            "model_units": "unitless",
            "units": "unitless",
            "model": df,
        }

    favorability = (prior_a + prior_b + 0.5 * gradient) / 3.0
    pr_norm = grid.copy()
    pr_norm["favorability"] = favorability

    pfa = {
        "criteria": {
            "geologic": {
                "weight": 1.0,
                "components": {
                    "component_a": {
                        "weight": 0.5,
                        "pr0": 0.55,
                        "layers": {
                            "prior_layer_a": _layer(prior_a),
                            "gradient": _layer(gradient.copy()),
                        },
                        "pr_norm": pr_norm.copy(),
                    },
                    "component_b": {
                        "weight": 0.5,
                        "pr0": 0.50,
                        "layers": {
                            "prior_layer_b": _layer(prior_b),
                            "gradient": _layer(gradient.copy()),
                        },
                        "pr_norm": pr_norm.copy(),
                    },
                },
                "pr_norm": pr_norm.copy(),
            }
        }
    }

    well_idx = rng.choice(n_cells, size=n_wells, replace=False)
    well_x = coords[well_idx, 0]
    well_y = coords[well_idx, 1]
    well_z = coords[well_idx, 2]
    well_prior_a = prior_a[well_idx]
    well_prior_b = prior_b[well_idx]
    base_logit = 4.0 * (0.5 * well_prior_a + 0.5 * well_prior_b - 0.5)
    label_prob = 1.0 / (1.0 + np.exp(-base_logit))
    labels = (rng.uniform(0.0, 1.0, size=n_wells) < label_prob).astype(int)

    wells_gdf = gpd.GeoDataFrame(
        {
            "well_id": [f"SYN3D_{i:04d}" for i in range(n_wells)],
            "longitude": well_x,
            "latitude": well_y,
            "depth_m": well_z,
            "heat_label": labels.astype(float),
            "reservoir_label": labels.astype(float),
            "barrier_label": np.full(n_wells, np.nan),
            "label_source": "synthetic",
            "label_quality": "proxy",
            "geometry": list(map(Point, well_x, well_y, well_z)),
        },
        crs=crs,
    )

    return SyntheticPFA3D(pfa=pfa, wells=wells_gdf, rng=rng)


__all__ = ["SyntheticPFA3D", "make_synthetic_pfa_3d"]
