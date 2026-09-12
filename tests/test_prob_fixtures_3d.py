"""Tests for the 3D synthetic PFA fixture."""

from __future__ import annotations

import geopandas as gpd
import pytest

from geopfa.prob.pfa_grid import (
    PFAGridAdapter,
    extract_grid_extent,
    validate_pfa_for_probabilistic,
)
from tests.fixtures.synthetic_prob_3d import (
    SyntheticPFA3D,
    make_synthetic_pfa_3d,
)


def test_make_synthetic_pfa_3d_returns_typed_container() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=10, seed=0)
    assert isinstance(fixture, SyntheticPFA3D)
    assert isinstance(fixture.pfa, dict)
    assert isinstance(fixture.wells, gpd.GeoDataFrame)


def test_3d_fixture_grid_size() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=10, seed=1)
    pr_norm = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ]
    assert len(pr_norm) == 4 * 4 * 3


def test_3d_fixture_geometries_have_z() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=10, seed=2)
    pr_norm = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ]
    assert pr_norm.geometry.iloc[0].has_z
    assert fixture.wells.geometry.iloc[0].has_z


def test_3d_fixture_passes_validation() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=10, seed=3)
    validate_pfa_for_probabilistic(
        fixture.pfa, criteria="geologic", dimensions="3d"
    )


def test_3d_fixture_extent_returns_six_floats() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=10, seed=4)
    extent = extract_grid_extent(
        fixture.pfa, criteria="geologic", dimensions="3d"
    )
    xmin, _ymin, zmin, xmax, _ymax, zmax = extent
    assert xmin == pytest.approx(500_000.0)
    assert xmax == pytest.approx(600_000.0)
    assert zmin == pytest.approx(-3000.0)
    assert zmax == pytest.approx(-500.0)


def test_3d_fixture_components_accessible_via_adapter() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=10, seed=5)
    adapter = PFAGridAdapter(fixture.pfa, criteria="geologic", dimensions="3d")
    assert set(adapter.components()) == {"component_a", "component_b"}
    assert "prior_layer_a" in adapter.layers("component_a")
    assert "gradient" in adapter.layers("component_a")


def test_3d_fixture_wells_have_labels_and_depth() -> None:
    n_wells = 12
    fixture = make_synthetic_pfa_3d(
        grid_n=4, grid_nz=3, n_wells=n_wells, seed=6
    )
    assert len(fixture.wells) == n_wells
    assert "heat_label" in fixture.wells.columns
    assert "depth_m" in fixture.wells.columns
    # well depths must fall within the grid Z extent
    z_min, z_max = -3000.0, -500.0
    assert float(fixture.wells["depth_m"].min()) >= z_min
    assert float(fixture.wells["depth_m"].max()) <= z_max


def test_3d_fixture_is_deterministic_under_same_seed() -> None:
    fixture_a = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=8, seed=42)
    fixture_b = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=8, seed=42)
    assert (
        fixture_a.wells["heat_label"].to_numpy()
        == fixture_b.wells["heat_label"].to_numpy()
    ).all()
