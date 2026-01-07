"""Tests for layer_combination.py

This suit of tests covers the layer_combination module.

The specific 2D & 3D features must be covered to guarantee the
transition process to a unified module.
"""

from datetime import timedelta
from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

import geopfa.geopfa2d.layer_combination as layer_combination_2D
import geopfa.geopfa3d.layer_combination as layer_combination_3D
from geopfa.layer_combination import VoterVeto


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
