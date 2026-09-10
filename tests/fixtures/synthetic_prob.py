"""Synthetic geoPFA-style dataset fixtures for unit tests.

These fixtures build deterministic, region-agnostic test data so the Stage-1
probabilistic modules can be exercised without depending on the real Nevada or
Newberry datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


@dataclass
class SyntheticPFA:
    """Container for a synthetic, two-component, gridded PFA-like fixture."""

    pfa: dict
    wells: gpd.GeoDataFrame
    rng: np.random.Generator


def _make_grid(n: int, *, crs: str = "EPSG:32611") -> gpd.GeoDataFrame:
    """Square `n × n` grid of point geometries on the given CRS, in metres."""
    xs = np.linspace(500_000.0, 600_000.0, n)
    ys = np.linspace(4_300_000.0, 4_400_000.0, n)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    geometry = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame({"geometry": geometry}, crs=crs)


def _layer_value(
    coords: np.ndarray,
    *,
    cx: float,
    cy: float,
    sigma: float,
    rng: np.random.Generator,
    noise: float = 0.05,
) -> np.ndarray:
    """Smooth bump centred on `(cx, cy)` plus Gaussian noise."""
    dx = (coords[:, 0] - cx) / sigma
    dy = (coords[:, 1] - cy) / sigma
    field = np.exp(-(dx * dx + dy * dy))
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    field = field + rng.normal(0.0, noise, size=field.shape)
    return field


def make_synthetic_pfa(
    *,
    grid_n: int = 12,
    n_wells: int = 60,
    seed: int = 0,
    crs: str = "EPSG:32611",
) -> SyntheticPFA:
    """Construct a deterministic two-component synthetic PFA fixture.

    The fixture has:

    * a `grid_n × grid_n` model grid,
    * two components named `component_a` and `component_b`, each with three
      evidence layers: a component-specific "prior" layer (`prior_layer_a` or
      `prior_layer_b`), a shared `gradient` layer, and a sparse-binary
      `sparse_indicator` layer that should be auto-excluded from regression,
    * a `pr_norm` GeoDataFrame with a `favorability` column,
    * `n_wells` labelled wells with distinct `heat_label` / `reservoir_label`
      outcomes generated from the same offset-plus-evidence model fitted by the
      probabilistic workflow.

    All numeric layers are framework-generic and unit-less.
    """
    rng = np.random.default_rng(seed)
    grid = _make_grid(grid_n, crs=crs)
    coords = np.column_stack(
        [
            grid.geometry.x.to_numpy(dtype=float),
            grid.geometry.y.to_numpy(dtype=float),
        ]
    )
    n_cells = len(grid)

    cx_a, cy_a = 540_000.0, 4_360_000.0
    cx_b, cy_b = 570_000.0, 4_350_000.0

    prior_a = _layer_value(coords, cx=cx_a, cy=cy_a, sigma=25_000.0, rng=rng)
    prior_b = _layer_value(coords, cx=cx_b, cy=cy_b, sigma=25_000.0, rng=rng)
    gradient = _layer_value(
        coords, cx=560_000.0, cy=4_350_000.0, sigma=40_000.0, rng=rng
    )
    sparse_indicator = (rng.uniform(0.0, 1.0, size=n_cells) < 0.03).astype(
        float
    )  # ~3% positive => sparse-binary

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
                            "sparse_indicator": _layer(
                                sparse_indicator.copy()
                            ),
                        },
                        "pr_norm": pr_norm.copy(),
                    },
                    "component_b": {
                        "weight": 0.5,
                        "pr0": 0.50,
                        "layers": {
                            "prior_layer_b": _layer(prior_b),
                            "gradient": _layer(gradient.copy()),
                            "sparse_indicator": _layer(
                                sparse_indicator.copy()
                            ),
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
    well_prior_a = prior_a[well_idx]
    well_prior_b = prior_b[well_idx]

    # Matched logistic data-generating process. The per-component prior layers
    # are transformed exactly as AlphaModeConfig's default layer_logit mode,
    # while the shared gradient enters as a linear evidence effect. Outcomes
    # are conditionally independent given those predictors, so their joint
    # event probability is the product modeled by the GBLK workflow. The old
    # fixture duplicated one Bernoulli draw into both endpoints, making that
    # product mathematically false and invalidating joint-recovery tests.
    def prior_probability(values: np.ndarray) -> np.ndarray:
        scaled = (values - values.min()) / (values.max() - values.min())
        return 0.2 + 0.6 * scaled

    gradient_scaled = (gradient - gradient.mean()) / gradient.std()
    well_gradient = gradient_scaled[well_idx]
    heat_prior = prior_probability(prior_a)[well_idx]
    reservoir_prior = prior_probability(prior_b)[well_idx]
    heat_logit = np.log(heat_prior / (1.0 - heat_prior)) + 0.8 * well_gradient
    reservoir_logit = (
        np.log(reservoir_prior / (1.0 - reservoir_prior)) + 0.8 * well_gradient
    )
    heat_probability = 1.0 / (1.0 + np.exp(-heat_logit))
    reservoir_probability = 1.0 / (1.0 + np.exp(-reservoir_logit))
    heat_labels = rng.binomial(1, heat_probability, size=n_wells)
    reservoir_labels = rng.binomial(1, reservoir_probability, size=n_wells)

    wells_gdf = gpd.GeoDataFrame(
        {
            "well_id": [f"SYN_{i:04d}" for i in range(n_wells)],
            "longitude": well_x,
            "latitude": well_y,
            "depth_m": np.full(n_wells, np.nan),
            "heat_label": heat_labels.astype(float),
            "reservoir_label": reservoir_labels.astype(float),
            "barrier_label": np.full(n_wells, np.nan),
            "label_source": "synthetic",
            "label_quality": "proxy",
            "geometry": [Point(x, y) for x, y in zip(well_x, well_y)],
        },
        crs=crs,
    )

    return SyntheticPFA(pfa=pfa, wells=wells_gdf, rng=rng)


__all__ = ["SyntheticPFA", "make_synthetic_pfa"]
