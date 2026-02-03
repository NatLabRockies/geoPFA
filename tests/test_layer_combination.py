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
from tests.fixtures.gdf_builders import gdf_from_xy_value, extract_xy_from_gdf


# ==== Transition tests ====
# Tests to secure the transition from the 2D & 3D modules to the
# unified module. Eventually, this might be unecessary.


def test_2D_get_w0():
    """
    Test some special cases for `get_w0`
    """
    Voter = layer_combination_2D.VoterVeto()

    assert np.isneginf(Voter.get_w0(0))

    assert Voter.get_w0(0.5) == 0.0

    with pytest.raises(ZeroDivisionError):
        Voter.get_w0(1)


def test_3D_get_w0():
    """
    Test some special cases for `get_w0`
    """
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
    """
    Confirm that get_w0 is consistent with the unified module
    """
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
    """
    Confirm that the 2D voter is consistent with the unified module
    """
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
    """
    Confirm that the 3D voter is consistent with the unified module
    """
    Voter = layer_combination_3D.VoterVeto()

    w = np.random.random(n_layers)
    z = np.random.random((n_layers, ni, nj, nk))
    w0 = Voter.get_w0(Pr0)

    assert np.allclose(VoterVeto.voter(w, z, w0), Voter.voter(w, z, w0))


# ==== Unified module tests ====
# ==== detect_geom_dimension tests ====
# The function detects whether a GeoDataFrame geometry is 2D or 3D and
# raises a ValueError if gdf is empty or has mixed dimensionality.


def test_detect_geom_dimension_2d():
    """
    Verify that purely 2D geometries are correctly identified as 2D.
    """
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
        ]
    )

    assert detect_geom_dimension(gdf) == 2


def test_detect_geom_dimension_3d():
    """
    Verify that purely 3D geometries (with Z coordinates) are correctly
    identified as 3D.
    """
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0, 0),
            Point(1, 1, 1),
            Point(2, 2, 2),
        ]
    )

    assert detect_geom_dimension(gdf) == 3


def test_detect_geom_dimension_mixed_raises():
    """
    Ensure an error is raised when 2D and 3D geometries are mixed within
    a single GeoDataFrame.
    """
    gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0),
            Point(1, 1, 1),
        ]
    )

    with pytest.raises(ValueError, match="Mixed 2D and 3D geometries"):
        detect_geom_dimension(gdf)


def test_detect_geom_dimension_empty_raises():
    """
    Ensure an informative error is raised for empty GeoDataFrames.
    """
    gdf = gpd.GeoDataFrame(geometry=[])

    with pytest.raises(ValueError, match="empty GeoDataFrame"):
        detect_geom_dimension(gdf)


def test_detect_geom_dimension_all_none_raises():
    """
    Ensure an error is raised when all geometries are None.
    """
    gdf = gpd.GeoDataFrame(geometry=[None, None])

    with pytest.raises(ValueError, match="all geometries are None"):
        detect_geom_dimension(gdf)


# ==== detect_pfa_dimension tests ====
# The function detects whether the PFA uses 2D or 3D geometries and enforces
# that non-empty layer GeoDataFrames must share the same dimensionality.


def test_detect_pfa_dimension_single_layer_2d():
    """
    Infer overall PFA dimensionality from a single 2D layer.
    """
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])

    pfa = make_pfa_with_layers({"layer1": gdf})

    assert detect_pfa_dimension(pfa) == 2


def test_detect_pfa_dimension_multiple_layers_same_dim():
    """
    Infer overall PFA dimensionality when multiple layers consistently share
    the same dimension.
    """
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
    """
    Reject PFAs that mix 2D and 3D layers across components or criteria.
    """
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
    """
    Ensure an error is raised when no non-empty layer models are available.
    """
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
    """
    Ensure an error is raised when all layer models are set to None.
    """
    pfa = make_pfa_with_layers(
        {
            "layer1": None,
            "layer2": None,
        }
    )

    with pytest.raises(ValueError):
        detect_pfa_dimension(pfa)


# ==== get_w0 tests ====
# The function derives w0 value using logit function.


def test_get_w0():
    """
    Validate special and boundary cases for the unified get_w0 implementation.”
    """
    assert np.isneginf(VoterVeto.get_w0(0))

    assert VoterVeto.get_w0(0.5) == 0.0

    with pytest.raises(ZeroDivisionError):
        VoterVeto.get_w0(1)


# ==== voter tests ====
# The voter function combines processed, transformed, and scaled data layers into a
# 'favorability' grid for a specific required resource component using a
# generalized linear model.


def test_voter():
    """
    Verify the voter produces expected favorability values for a simple,
    deterministic input.
    """
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
    """
    Ensure voter outputs preserve expected shape and remain bounded.
    """
    w = np.random.random(n_layers)
    z = np.random.random((n_layers, ni, nj, nk))
    w0 = VoterVeto.get_w0(Pr0)

    PrX = VoterVeto.voter(w, z, w0)

    assert PrX.shape == (ni, nj, nk)
    assert np.all(PrX >= 0) and np.all(PrX <= 1)


# ==== veto tests ====
# The veto function performs element-wise multiplication of component
# favorabilities.


def test_veto_elementwise_multiplication():
    """
    Confirm that the original veto combines component favorabilities via
    element-wise multiplication, including NaN propagation.
    """
    PrXs = np.array(
        [
            [[0.0, 0.2], [1.0, 0.8]],
            [[0.4, 0.5], [0.5, np.nan]],
        ]
    )

    result = VoterVeto.veto(PrXs)

    expected = np.array([[0.0 * 0.4, 0.2 * 0.5], [1.0 * 0.5, np.nan]])

    assert np.allclose(result, expected, equal_nan=True)


# ==== modified_veto tests ====
# modified_veto combines component 'favorability' grids into a resource 'favorability'
# model, optionally vetoing areas where any one component is not present (0% 'favorability').
# This method combines component 'favorability' grids using a weighted sum, and then normalizing.


def test_modified_veto_weighted_no_veto():
    """
    Verify weighted aggregation behavior of modified_veto when hard vetoing
    is disabled.
    """
    PrXs = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]
    )
    w = np.array([0.5, 0.5])

    result = VoterVeto.modified_veto(PrXs, w, veto=False)

    # Basic sanity checks
    assert result.shape == (2, 2)
    assert np.nanmax(result) > 0
    assert result[1, 1] > result[0, 0]  # ordering preserved


# ==== prepare_for_combination tests ====
# The prepare_for_combination function prepare a stack of layers/components for
# combination by handling NaNs.


def test_prepare_propagate_shared_partial_nans():
    """
    Ensure that NaNs propagate only when all contributing inputs are NaN at a
    pixel under propagate_shared mode.
    """
    arr = np.array(
        [
            [[1.0, np.nan], [2.0, 3.0]],
            [[4.0, 5.0], [np.nan, 6.0]],
        ]
    )

    filled, mask = VoterVeto.prepare_for_combination(
        arr, nan_mode="propagate_shared"
    )

    # Only pixels where *all* inputs are NaN should be masked
    assert mask.shape == (2, 2)
    assert not mask.any()

    # Filled array should contain no NaNs
    assert not np.isnan(filled).any()


def test_prepare_propagate_any_masks_any_nan():
    """
    Ensure that any NaN in contributing inputs masks the output pixel under
    propagate_any mode.
    """
    arr = np.array(
        [
            [[1.0, np.nan], [2.0, 3.0]],
            [[4.0, 5.0], [np.nan, 6.0]],
        ]
    )

    filled, mask = VoterVeto.prepare_for_combination(
        arr, nan_mode="propagate_any"
    )

    expected_mask = np.array(
        [
            [False, True],
            [True, False],
        ]
    )

    assert np.array_equal(mask, expected_mask)

    # Filled array itself should be finite (mask controls NaNs)
    assert not np.isnan(filled).any()


def test_prepare_all_nan_input():
    """
    Confirm that fully missing inputs remain fully masked and are not
    neutral-filled.
    """
    arr = np.array(
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ]
    )

    filled, mask = VoterVeto.prepare_for_combination(
        arr, nan_mode="propagate_shared"
    )

    # All pixels have no data
    assert mask.all()

    # Filled array should remain NaN everywhere
    assert np.isnan(filled).all()


def test_prepare_invalid_inputs_raise():
    """
    Ensure invalid input shapes or unsupported NaN handling modes raise
    informative errors
    """
    # Too few dimensions
    with pytest.raises(ValueError):
        VoterVeto.prepare_for_combination(np.array([1.0, 2.0]))

    # Invalid nan_mode
    arr = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        VoterVeto.prepare_for_combination(arr, nan_mode="not_a_mode")


# ==== do_voter_veto tests ====
# The do_voter_veto function combines individual data layers into a resource 'favorability'
# model, using the unified 2D/3D voter-veto implementation with NaN handling.


def test_do_voter_veto_minimal_2d():
    """
    Run a minimal end-to-end 2D voter-veto pipeline to ensure basic wiring,
    rasterization, and aggregation succeed.
    """
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])

    Z = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    gdf = gdf_from_xy_value(xs, ys, Z)

    pfa = make_pfa_with_layers({"layer1": gdf})

    layer_cfg = pfa["criteria"]["crit1"]["components"]["comp1"]["layers"][
        "layer1"
    ]
    layer_cfg["model_data_col"] = "value"
    layer_cfg["weight"] = 1.0
    layer_cfg["transformation_method"] = "none"

    pfa["criteria"]["crit1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["pr0"] = 0.5

    result = VoterVeto.do_voter_veto(
        pfa,
        normalize_method="minmax",
        normalize=True,
    )

    assert "pr" in result
    assert len(result["pr"]) == 4
    assert "favorability" in result["pr"].columns


def test_do_voter_veto_minimal_3d():
    """
    Run a minimal end-to-end 3D voter-veto pipeline to ensure basic wiring,
    rasterization, and aggregation succeed.
    """
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([0.0, 1.0])  # two depth levels

    pts = []
    vals = []

    Z = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    for k, z in enumerate(zs):
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                pts.append(Point(x, y, z))
                vals.append(Z[k, i, j])

    gdf = gpd.GeoDataFrame(
        {"geometry": pts, "value": vals},
        crs=None,
    )

    pfa = make_pfa_with_layers({"layer1": gdf})

    layer_cfg = pfa["criteria"]["crit1"]["components"]["comp1"]["layers"][
        "layer1"
    ]
    layer_cfg["model_data_col"] = "value"
    layer_cfg["weight"] = 1.0
    layer_cfg["transformation_method"] = "none"

    pfa["criteria"]["crit1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["pr0"] = 0.5

    result = VoterVeto.do_voter_veto(
        pfa,
        normalize_method="minmax",
        normalize=True,
    )

    assert "pr" in result
    assert len(result["pr"]) == 8  # 2×2×2 voxels
    assert "favorability" in result["pr"].columns
    assert result["pr"].geometry.iloc[0].has_z


def test_do_voter_veto_nan_propagate_shared():
    """
    Verify that shared NaN regions propagate correctly through the full
    voter-veto workflow under the propagate_shared mode
    """
    # nan_mode="propagate_shared"

    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])

    Z1 = np.array(
        [
            [np.nan, 1.0],
            [2.0, 3.0],
        ]
    )

    Z2 = np.array(
        [
            [np.nan, 4.0],
            [5.0, 6.0],
        ]
    )

    gdf1 = gdf_from_xy_value(xs, ys, Z1)
    gdf2 = gdf_from_xy_value(xs, ys, Z2)

    pfa = make_pfa_with_layers(
        {
            "layer1": gdf1,
            "layer2": gdf2,
        }
    )

    for name in ["layer1", "layer2"]:
        layer_cfg = pfa["criteria"]["crit1"]["components"]["comp1"]["layers"][
            name
        ]
        layer_cfg["model_data_col"] = "value"
        layer_cfg["weight"] = 0.5
        layer_cfg["transformation_method"] = "none"

    pfa["criteria"]["crit1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["pr0"] = 0.5

    result = VoterVeto.do_voter_veto(
        pfa,
        normalize_method="minmax",
        nan_mode="propagate_shared",
    )

    vals = result["pr"]["favorability"].values
    assert np.isnan(vals).sum() == 1  # only shared-NaN pixel


def test_do_voter_veto_shape_mismatch_raises():
    """
    Ensure the pipeline fails fast when layer grids have incompatible shapes.
    """
    xs1 = np.array([0.0, 1.0])
    ys1 = np.array([0.0, 1.0])
    Z1 = np.ones((2, 2))

    xs2 = np.array([0.0, 1.0, 2.0])
    ys2 = np.array([0.0, 1.0])
    Z2 = np.ones((2, 3))

    gdf1 = gdf_from_xy_value(xs1, ys1, Z1)
    gdf2 = gdf_from_xy_value(xs2, ys2, Z2)

    pfa = make_pfa_with_layers(
        {
            "layer1": gdf1,
            "layer2": gdf2,
        }
    )

    for name in ["layer1", "layer2"]:
        layer_cfg = pfa["criteria"]["crit1"]["components"]["comp1"]["layers"][
            name
        ]
        layer_cfg["model_data_col"] = "value"
        layer_cfg["weight"] = 0.5
        layer_cfg["transformation_method"] = "none"

    pfa["criteria"]["crit1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["weight"] = 1.0
    pfa["criteria"]["crit1"]["components"]["comp1"]["pr0"] = 0.5

    with pytest.raises(ValueError, match="Layer grid shape mismatch"):
        VoterVeto.do_voter_veto(
            pfa,
            normalize_method="minmax",
        )
