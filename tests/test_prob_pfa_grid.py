"""Tests for the PFAGrid protocol and adapter helpers (2D + 3D)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from geopfa.prob.pfa_grid import (
    PFAGridAdapter,
    component_layers,
    component_names,
    extract_grid_extent,
    iter_components,
    layer_data_column,
    layer_model_gdf,
    validate_pfa_for_probabilistic,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# ---------------------------------------------------------------------------
# 2D fixture round-trip
# ---------------------------------------------------------------------------


def test_synthetic_2d_pfa_satisfies_protocol() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=20, seed=0)
    pfa = fixture.pfa
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    assert adapter.dimensions == "2d"
    comps = adapter.components()
    assert set(comps) == {"component_a", "component_b"}
    layers_a = list(adapter.layers("component_a"))
    assert "prior_layer_a" in layers_a
    assert "gradient" in layers_a
    grid_a = adapter.layer_model("component_a", "gradient")
    assert isinstance(grid_a, gpd.GeoDataFrame)
    assert "value_interpolated" in grid_a.columns
    assert grid_a.crs is not None


def test_iter_components_yields_all_components() -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=10, seed=1)
    comps = list(iter_components(fixture.pfa, criteria="geologic"))
    assert {name for name, _ in comps} == {"component_a", "component_b"}
    for _, data in comps:
        assert "layers" in data
        assert "pr_norm" in data


def test_component_names_returns_sorted_list() -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=2)
    names = component_names(fixture.pfa, criteria="geologic")
    assert names == ["component_a", "component_b"]


def test_layer_data_column_returns_model_data_col() -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=3)
    col = layer_data_column(
        fixture.pfa["criteria"]["geologic"]["components"]["component_a"]["layers"][
            "gradient"
        ]
    )
    assert col == "value_interpolated"


def test_layer_model_gdf_returns_gdf() -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=4)
    gdf = layer_model_gdf(
        fixture.pfa["criteria"]["geologic"]["components"]["component_a"]["layers"][
            "gradient"
        ]
    )
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_component_layers_returns_layers_dict() -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=5)
    layers = component_layers(
        fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    )
    assert "gradient" in layers
    assert "prior_layer_a" in layers
    assert "sparse_indicator" in layers


def test_extract_grid_extent_2d() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=10, seed=6)
    extent = extract_grid_extent(fixture.pfa, criteria="geologic", dimensions="2d")
    # SyntheticPFA: linspace(500_000, 600_000) × linspace(4_300_000, 4_400_000)
    xmin, ymin, xmax, ymax = extent
    assert xmin == pytest.approx(500_000.0)
    assert ymin == pytest.approx(4_300_000.0)
    assert xmax == pytest.approx(600_000.0)
    assert ymax == pytest.approx(4_400_000.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_passes_for_synthetic_2d() -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=7)
    validate_pfa_for_probabilistic(fixture.pfa, criteria="geologic", dimensions="2d")


def test_validate_raises_on_missing_criteria() -> None:
    pfa = {"criteria": {}}
    with pytest.raises(KeyError, match="geologic"):
        validate_pfa_for_probabilistic(pfa, criteria="geologic", dimensions="2d")


def test_validate_raises_on_missing_components() -> None:
    pfa = {"criteria": {"geologic": {"weight": 1.0}}}
    with pytest.raises(KeyError, match="components"):
        validate_pfa_for_probabilistic(pfa, criteria="geologic", dimensions="2d")


def test_validate_accepts_component_without_pr_norm() -> None:
    """pr_norm is optional — VoterVeto is an independent pathway from probabilistic."""
    geom = [Point(0, 0), Point(1, 0)]
    model_gdf = gpd.GeoDataFrame({"value_interpolated": [0.3, 0.7], "geometry": geom}, crs="EPSG:32611")
    pfa = {
        "criteria": {
            "geologic": {
                "components": {
                    "heat": {"layers": {"layer_a": {"model": model_gdf}}}
                }
            }
        }
    }
    # Should not raise — pr_norm is absent but layers/model are present.
    validate_pfa_for_probabilistic(pfa, criteria="geologic", dimensions="2d")


def test_validate_raises_on_layer_missing_model() -> None:
    geom = [Point(0, 0), Point(1, 0)]
    grid = gpd.GeoDataFrame({"favorability": [0.1, 0.2], "geometry": geom}, crs="EPSG:32611")
    pfa = {
        "criteria": {
            "geologic": {
                "components": {
                    "heat": {
                        "pr_norm": grid,
                        "layers": {"layer_a": {"data_col": "value"}},
                    }
                }
            }
        }
    }
    with pytest.raises(KeyError, match="model"):
        validate_pfa_for_probabilistic(pfa, criteria="geologic", dimensions="2d")


# ---------------------------------------------------------------------------
# 3D adapter (minimal — full coverage in Phase J 3D fixture)
# ---------------------------------------------------------------------------


def _make_minimal_3d_pfa() -> dict:
    rng = np.random.default_rng(0)
    nx, ny, nz = 4, 4, 3
    xs = np.linspace(0.0, 100.0, nx)
    ys = np.linspace(0.0, 100.0, ny)
    zs = np.linspace(-300.0, -100.0, nz)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    geom = [Point(x, y, z) for x, y, z in pts]
    crs = "EPSG:32611"
    layer_vals = rng.uniform(0.0, 1.0, size=len(pts))
    layer_gdf = gpd.GeoDataFrame(
        {"value_interpolated": layer_vals, "geometry": geom}, crs=crs
    )
    pr_norm = gpd.GeoDataFrame(
        {"favorability": layer_vals * 5.0, "geometry": geom}, crs=crs
    )
    return {
        "criteria": {
            "geologic": {
                "weight": 1.0,
                "components": {
                    "heat": {
                        "pr0": 0.5,
                        "weight": 1.0,
                        "layers": {
                            "layer_3d": {
                                "model_data_col": "value_interpolated",
                                "model": layer_gdf,
                            }
                        },
                        "pr_norm": pr_norm,
                    }
                },
                "pr_norm": pr_norm,
            }
        }
    }


def test_3d_pfa_satisfies_protocol() -> None:
    pfa = _make_minimal_3d_pfa()
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="3d")
    assert adapter.dimensions == "3d"
    assert adapter.components() == ["heat"]
    grid = adapter.layer_model("heat", "layer_3d")
    # Z coord present
    assert hasattr(grid.geometry.iloc[0], "z")


def test_3d_extract_grid_extent_returns_xyz_bounds() -> None:
    pfa = _make_minimal_3d_pfa()
    extent = extract_grid_extent(pfa, criteria="geologic", dimensions="3d")
    xmin, ymin, zmin, xmax, ymax, zmax = extent
    assert xmin == pytest.approx(0.0)
    assert xmax == pytest.approx(100.0)
    assert zmin == pytest.approx(-300.0)
    assert zmax == pytest.approx(-100.0)


def test_3d_validation_raises_when_geometry_lacks_z() -> None:
    geom = [Point(0, 0), Point(1, 0)]  # 2D
    grid = gpd.GeoDataFrame({"favorability": [0.1, 0.2], "geometry": geom}, crs="EPSG:32611")
    layer_gdf = gpd.GeoDataFrame(
        {"value_interpolated": [0.5, 0.5], "geometry": geom}, crs="EPSG:32611"
    )
    pfa = {
        "criteria": {
            "geologic": {
                "components": {
                    "heat": {
                        "pr_norm": grid,
                        "layers": {"layer_a": {"model": layer_gdf, "model_data_col": "value_interpolated"}},
                    }
                }
            }
        }
    }
    with pytest.raises(ValueError, match="Z coordinate"):
        validate_pfa_for_probabilistic(pfa, criteria="geologic", dimensions="3d")
