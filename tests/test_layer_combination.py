"""Tests for layer_combination.py

This suit of tests covers the layer_combination module.

The specific 2D & 3D features must be covered to guarantee the
transition process to a unified module.
"""

from datetime import timedelta
from hypothesis import given, settings, strategies as st
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pytest

import geopfa.geopfa2d.layer_combination as layer_combination_2D
import geopfa.geopfa3d.layer_combination as layer_combination_3D
from geopfa.layer_combination import VoterVeto
from geopfa.layer_combination import (
    detect_geom_dimension,
    detect_pfa_dimension,
)
from tests.fixtures.pfa_builders import make_pfa_with_layers


# ==== Transition tets ====
# Tests to secure the transition from the 2D & 3D modules to the
# unified module. Eventually, this might be unecessary.


def test_2D_get_w0():
    """Test some special cases for `get_w0`"""
    Voter = layer_combination_2D.VoterVeto()

    assert np.isneginf(Voter.get_w0(0))

    assert Voter.get_w0(0.5) == 0.0

    with pytest.raises(ZeroDivisionError):
        Voter.get_w0(1)


def test_3D_get_w0():
    """Test some special cases for `get_w0`"""
    Voter = layer_combination_3D.VoterVeto()

    assert np.isneginf(Voter.get_w0(0))

    assert Voter.get_w0(0.5) == 0.0

    with pytest.raises(ZeroDivisionError):
        Voter.get_w0(1)


@given(
    st.floats(
        min_value=0.0,
        max_value=1.0,
        exclude_max=True,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_validate_2D_and_3D_get_w0(Pr0):
    ans = VoterVeto.get_w0(Pr0)
    assert ans == layer_combination_2D.VoterVeto().get_w0(Pr0)
    assert ans == layer_combination_3D.VoterVeto().get_w0(Pr0)


@given(
    st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10, deadline=timedelta(milliseconds=500))
def test_validate_2D_voter(Pr0, n_layers, ni, nj):
    """Confirm that the 2D voter is consistent with the unified module"""
    Voter = layer_combination_2D.VoterVeto()

    w = np.random.random(n_layers)
    z = np.random.random((n_layers, ni, nj))
    w0 = Voter.get_w0(Pr0)

    assert np.allclose(VoterVeto.voter(w, z, w0), Voter.voter(w, z, w0))


@given(
    st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10, deadline=timedelta(milliseconds=500))
def test_validate_3D_voter(Pr0, n_layers, ni, nj, nk):
    """Confirm that the 3D voter is consistent with the unified module"""
    Voter = layer_combination_3D.VoterVeto()

    w = np.random.random(n_layers)
    z = np.random.random((n_layers, ni, nj, nk))
    w0 = Voter.get_w0(Pr0)

    assert np.allclose(VoterVeto.voter(w, z, w0), Voter.voter(w, z, w0))


# ==== Unified module tests ====


def test_get_w0():
    """Test some special cases for `get_w0`"""
    assert np.isneginf(VoterVeto.get_w0(0))

    assert VoterVeto.get_w0(0.5) == 0.0

    with pytest.raises(ZeroDivisionError):
        VoterVeto.get_w0(1)


def test_voter():
    """Test a simple case for `voter`"""
    w = np.array([0.1])
    z = np.array(
        [
            [
                [1, 2],
                [10, 20],
            ]
        ]
    )
    w0 = VoterVeto.get_w0(0.5)
    PrX = VoterVeto.voter(w, z, w0)

    assert np.allclose(
        PrX, np.array([[0.52497919, 0.549834], [0.73105858, 0.88079708]])
    )


@given(
    st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_voter_properties(Pr0, n_layers, ni, nj, nk):
    w = np.random.random(n_layers)
    z = np.random.random((n_layers, ni, nj, nk))
    w0 = VoterVeto.get_w0(Pr0)

    PrX = VoterVeto.voter(w, z, w0)

    assert PrX.shape == (ni, nj, nk)
    assert np.all(PrX >= 0) and np.all(PrX <= 1)


# ==== Dimensionality detection function tests ====


def test_detect_geom_dimension_2d():
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
        ]
    )

    assert detect_geom_dimension(gdf) == 2


def test_detect_geom_dimension_3d():
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0, 0),
            Point(1, 1, 1),
            Point(2, 2, 2),
        ]
    )

    assert detect_geom_dimension(gdf) == 3


def test_detect_geom_dimension_mixed_raises():
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0),
            Point(1, 1, 1),
        ]
    )

    with pytest.raises(ValueError, match="Mixed 2D and 3D geometries"):
        detect_geom_dimension(gdf)


def test_detect_geom_dimension_empty_raises():
    gdf = gpd.GeoDataFrame(geometry=[])

    with pytest.raises(ValueError, match="empty GeoDataFrame"):
        detect_geom_dimension(gdf)


def test_detect_geom_dimension_all_none_raises():
    gdf = gpd.GeoDataFrame(geometry=[None, None])

    with pytest.raises(ValueError, match="all geometries are None"):
        detect_geom_dimension(gdf)


def test_detect_pfa_dimension_single_layer_2d():
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])

    pfa = make_pfa_with_layers({"layer1": gdf})

    assert detect_pfa_dimension(pfa) == 2


def test_detect_pfa_dimension_multiple_layers_same_dim():
    gdf1 = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])
    gdf2 = gpd.GeoDataFrame(geometry=[Point(2, 2), Point(3, 3)])

    pfa = make_pfa_with_layers(
        {
            "layer1": gdf1,
            "layer2": gdf2,
        }
    )

    assert detect_pfa_dimension(pfa) == 2


def test_detect_pfa_dimension_mixed_layers_raises():
    gdf2d = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    gdf3d = gpd.GeoDataFrame(geometry=[Point(0, 0, 1)])

    pfa = make_pfa_with_layers(
        {
            "layer2d": gdf2d,
            "layer3d": gdf3d,
        }
    )

    with pytest.raises(ValueError, match="mixes 2D and 3D geometries"):
        detect_pfa_dimension(pfa)


def test_detect_pfa_dimension_all_layers_empty_raises():
    pfa = make_pfa_with_layers(
        {
            "layer1": gpd.GeoDataFrame(geometry=[]),
            "layer2": gpd.GeoDataFrame(geometry=[]),
        }
    )

    with pytest.raises(
        ValueError, match="No non-empty layer models were found"
    ):
        detect_pfa_dimension(pfa)


def test_detect_pfa_dimension_all_layers_none_raises():
    pfa = make_pfa_with_layers(
        {
            "layer1": None,
            "layer2": None,
        }
    )

    with pytest.raises(ValueError):
        detect_pfa_dimension(pfa)
